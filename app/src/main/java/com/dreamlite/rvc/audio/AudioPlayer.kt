package com.dreamlite.rvc.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlin.math.max

/**
 * Mono playback. The pipeline emits blocks at the voice's own rate (40 kHz
 * for RVC v2), but the device only honours a small low-latency AudioTrack
 * buffer at 48 kHz: requesting 10 ms at 48 kHz + PERFORMANCE_MODE_LOW_LATENCY
 * gives a 120 ms buffer here, while 40 kHz (or the non-low-latency path)
 * clamps to the 601 ms deep-buffer minimum. That 480 ms is pure output
 * latency, so the player resamples the voice stream to 48 kHz.
 *
 * The inference loop must never block on the speaker. AudioTrack.write() is
 * a real-time-paced call; called from the feeder it would serialise compute
 * with buffer drain, and the capture channel would silently fill up. Writes
 * therefore go to a small bounded queue and a dedicated writer thread.
 *
 * The queue holds at most [BLOCKS_QUEUED] blocks: if the speaker falls
 * behind we drop the oldest block instead of letting the delay grow.
 */
class AudioPlayer(
    voiceRate: Int = DEFAULT_SR,
    private val muted: Boolean = false,
    private val preferredDevice: android.media.AudioDeviceInfo? = null,
) {

    private val inRate = voiceRate

    /** Fast-path output rate; the deep-buffer alternative is 601 ms here. */
    private val outRate = TARGET_SR

    companion object {
        private const val TAG = "rvc_play"

        /** Blocks kept ahead of the speaker. */
        private const val BLOCKS_QUEUED = 2
        const val DEFAULT_SR = 40_000
        private const val TARGET_SR = 48_000

        private val AUDIO_ATTRIBUTES = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()
    }

    @Volatile
    private var running = false

    @Volatile
    private var track: AudioTrack? = null

    private val queue = ArrayDeque<FloatArray>()
    private val qLock = Object()
    private var writer: Thread? = null

    /** Frames handed to AudioTrack; used by [finish] to drain the tail. */
    @Volatile
    private var writtenFrames = 0L

    /** Blocks discarded because the speaker was already 2 blocks behind. */
    @Volatile
    var dropped: Long = 0
        private set

    /* ---- streaming linear resampler (voiceRate -> 48 kHz) ---- */

    private val resampleLock = Object()
    private var resamplePhase = 0.0
    private var resamplePrev = 0.0f

    private fun resample(src: FloatArray): FloatArray {
        if (inRate == outRate) return src
        if (src.isEmpty()) return src
        val step = inRate.toDouble() / outRate
        synchronized(resampleLock) {
            val n = ((((src.size - 1).toDouble() - resamplePhase) / step).toInt() + 1)
                .coerceAtLeast(0)
            val out = FloatArray(n)
            var p = resamplePhase
            val last = src[src.size - 1]
            for (i in 0 until n) {
                val i0 = kotlin.math.floor(p).toInt()
                val frac = p - i0
                val a = if (i0 < 0) resamplePrev else src[i0]
                val b = if (i0 + 1 < src.size) src[i0 + 1] else last
                out[i] = (a + (b - a) * frac).toFloat()
                p += step
            }
            resamplePhase = p - src.size
            resamplePrev = last
            return out
        }
    }

    /* ---- lifecycle ---- */

    fun start(): Boolean {
        if (running) return true
        var t = newTrack(lowLatency = true)
        if (t == null) {
            Log.w(TAG, "low-latency track rejected, falling back to deep buffer")
            t = newTrack(lowLatency = false)
        }
        if (t == null) {
            Log.e(TAG, "AudioTrack not initialized")
            return false
        }
        track = t
        running = true
        dropped = 0
        writtenFrames = 0
        synchronized(qLock) { queue.clear() }
        synchronized(resampleLock) {
            resamplePhase = 0.0
            resamplePrev = 0f
        }
        t.play()
        t.setVolume(if (muted) 0f else 1f)
        preferredDevice?.let { dev ->
            if (!t.setPreferredDevice(dev)) Log.w(TAG, "preferred device rejected: ${dev.type}")
        }
        writer = Thread({ writeLoop(t) }, "rvc-play").also {
            it.priority = Thread.NORM_PRIORITY + 1
            it.start()
        }
        return true
    }

    private fun newTrack(lowLatency: Boolean): AudioTrack? {
        val fmt = AudioFormat.Builder()
            .setSampleRate(outRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .build()
        // 10 ms request: the framework rounds up to its fast-path size (120 ms
        // here). Clamping to getMinBufferSize() would force the 601 ms deep
        // buffer, which is exactly the latency we are removing.
        val want = if (lowLatency) outRate / 100 * 4 else {
            max(
                AudioTrack.getMinBufferSize(
                    outRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
                ),
                outRate / 4 * 4,
            )
        }
        val t = AudioTrack(
            AUDIO_ATTRIBUTES,
            fmt,
            want,
            AudioTrack.MODE_STREAM,
            if (lowLatency) AudioTrack.PERFORMANCE_MODE_LOW_LATENCY
            else AudioTrack.PERFORMANCE_MODE_NONE,
        )
        if (t.state != AudioTrack.STATE_INITIALIZED) {
            t.release()
            return null
        }
        Log.i(
            TAG,
            "track: ${outRate} Hz ${if (lowLatency) "low-latency" else "deep"} " +
                "alloc=${t.bufferSizeInFrames} frames " +
                "(${t.bufferSizeInFrames * 1000 / outRate} ms)",
        )
        return t
    }

    /* ---- queue ---- */

    /** Queues one block. Never blocks: the speaker thread drains the queue. */
    fun write(block: FloatArray) = enqueue(resample(block), wait = false)

    /** For file preview, where dropping blocks would cut the audio. */
    fun writeBlocking(block: FloatArray) = enqueue(resample(block), wait = true)

    private fun enqueue(block: FloatArray, wait: Boolean) {
        if (!running || block.isEmpty()) return
        synchronized(qLock) {
            if (wait) {
                while (running && queue.size >= BLOCKS_QUEUED) qLock.wait(20)
                if (!running) return
            } else {
                while (queue.size >= BLOCKS_QUEUED) {
                    queue.removeFirst()
                    dropped++
                }
            }
            queue.addLast(block)
            qLock.notifyAll()
        }
    }

    private fun writeLoop(t: AudioTrack) {
        while (running) {
            val block = synchronized(qLock) {
                while (running && queue.isEmpty()) qLock.wait(20)
                if (!running) return
                queue.removeFirst()
            }
            var off = 0
            while (off < block.size && running) {
                val n = t.write(block, off, block.size - off, AudioTrack.WRITE_BLOCKING)
                if (n <= 0) return
                off += n
                writtenFrames += n
            }
            lastPresentMs = measurePresentMs()
        }
    }

    /* ---- latency readout ---- */

    /** Frames currently buffered ahead of the speaker, in milliseconds. */
    fun bufferedMs(): Int {
        val t = track ?: return 0
        if (!running) return 0
        val played = t.playbackHeadPosition.toLong() and 0xffffffffL
        val queued = (writtenFrames - played).coerceAtLeast(0L)
        return (queued * 1000 / outRate).toInt()
    }

    private val timestamp = android.media.AudioTimestamp()

    /** Last raw HAL measurement, refreshed by the writer after every write. */
    @Volatile
    private var lastPresentMs = 0

    private fun measurePresentMs(): Int {
        val t = track ?: return 0
        val written = writtenFrames
        if (t.getTimestamp(timestamp)) {
            val played = timestamp.framePosition
            val framesLeft = (written - played).coerceAtLeast(0L)
            val presentNs = timestamp.nanoTime + framesLeft * 1_000_000_000L / outRate
            val ns = presentNs - System.nanoTime()
            if (ns > 0) return (ns / 1_000_000L).toInt()
        }
        return bufferedMs()
    }

    /**
     * Milliseconds until the most recently produced block is presented by the
     * HAL. Includes the AudioTrack buffer *and* the device pipeline (on this
     * device the latter is ~470 ms with Bluetooth A2DP, ~50 ms on the
     * speaker). Blocks still queued for the writer count 250 ms each.
     */
    fun presentationDelayMs(): Int {
        if (!running) return 0
        val pendingBlocks = synchronized(qLock) { queue.size }
        return lastPresentMs + pendingBlocks * 250
    }

    /** One-line state for the latency measurement: written/head/timestamp. */
    fun debugLatency(): String {
        val t = track ?: return "no-track"
        val tsLocal = android.media.AudioTimestamp()
        val has = t.getTimestamp(tsLocal)
        val head = t.playbackHeadPosition.toLong() and 0xffffffffL
        val wr = writtenFrames
        val pend = synchronized(qLock) { queue.size }
        return "wr=${wr * 1000 / outRate} head=${head * 1000 / outRate} " +
            "ts=${if (has) tsLocal.framePosition * 1000 / outRate else -1} " +
            "tsAgeMs=${if (has) (System.nanoTime() - tsLocal.nanoTime) / 1_000_000 else -1} " +
            "last=$lastPresentMs pend=$pend"
    }

    /**
     * Waits until everything queued has been heard, then stops. File preview
     * must not truncate the last blocks; live capture end uses [stop] so the
     * mic can be released immediately.
     */
    fun finish(timeoutMs: Long = 10_000) {
        if (!running) return
        val deadline = android.os.SystemClock.elapsedRealtime() + timeoutMs
        synchronized(qLock) {
            while (running && queue.isNotEmpty() &&
                android.os.SystemClock.elapsedRealtime() < deadline
            ) {
                qLock.wait(20)
            }
        }
        val t = track
        if (t != null && running) {
            var spins = 0
            while (running && android.os.SystemClock.elapsedRealtime() < deadline && spins < 500) {
                val played = t.playbackHeadPosition.toLong() and 0xffffffffL
                if (writtenFrames - played <= 0) break
                try {
                    Thread.sleep(10)
                } catch (e: InterruptedException) {
                    break
                }
                spins++
            }
        }
        stop()
    }

    fun stop() {
        running = false
        synchronized(qLock) {
            queue.clear()
            qLock.notifyAll()
        }
        track?.let { t ->
            try { t.stop() } catch (e: IllegalStateException) { /* already stopped */ }
            try { t.release() } catch (e: IllegalStateException) { /* already released */ }
        }
        track = null
        writer?.let { runCatching { it.join(200) } }
        writer = null
    }
}
