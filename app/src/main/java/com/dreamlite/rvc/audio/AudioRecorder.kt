package com.dreamlite.rvc.audio

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow

/** One captured 250 ms block plus the wall clock at which it was completed. */
class CaptureBlock(val samples: FloatArray, val atMs: Long)

/**
 * 16 kHz mono capture. Emits 1600-sample chunks (100 ms). The pipeline can
 * consume any size, so a small chunk is pure latency win: it emits a block as
 * soon as the 3200/4800-sample threshold is crossed instead of waiting for
 * the next 250 ms read to return (75 ms less on average).
 *
 * The capture loop runs on a dedicated thread; [running] is the stop signal.
 * Level is published as a 0..1 amplitude envelope for the UI meter.
 */
class AudioRecorder {

    companion object {
        private const val TAG = "rvc_rec"
        private const val SAMPLE_RATE = 16_000
        private const val CHUNK = 1600
    }

    @Volatile
    private var running = false
    private var record: AudioRecord? = null

    /** Peak amplitude of the most recent block, 0..1. */
    val level = MutableStateFlow(0.0f)

    /**
     * Blocks discarded because the feeder was more than the channel capacity
     * behind. Every drop splices the pipeline's 2.8 s context, so a non-zero
     * count is audible (burrs / 电音) as well as a latency warning.
     */
    @Volatile
    var dropped: Long = 0
        private set

    val isRecording: Boolean get() = running

    @SuppressLint("MissingPermission")
    fun start(output: Channel<CaptureBlock>): Boolean {
        if (running) return true
        // The HAL minimum here is only 1280 B (40 ms), so asking for
        // BLOCK*2*2 = 500 ms of capture buffer buys nothing and costs a
        // quarter second of latency. One block worth is enough: the read is
        // blocking and is reissued immediately.
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(CHUNK * 2)
        val r = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf,
        )
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            r.release()
            return false
        }
        record = r
        running = true
        dropped = 0
        r.startRecording()

        Thread({
            val shortBuf = ShortArray(CHUNK)
            var peakHold = 0.0f
            while (running) {
                val n = r.read(shortBuf, 0, CHUNK, AudioRecord.READ_BLOCKING)
                if (n <= 0) continue
                val block = FloatArray(n)
                var peak = 0.0f
                for (i in 0 until n) {
                    val s = shortBuf[i] / 32768.0f
                    block[i] = s
                    val a = if (s < 0) -s else s
                    if (a > peak) peak = a
                }
                // level envelope: fast attack, slow release
                peakHold = if (peak > peakHold) peak else peakHold * 0.92f
                level.value = peakHold
                if (!running) break
                val captured = CaptureBlock(block, android.os.SystemClock.elapsedRealtime())
                val sent = output.trySend(captured)
                if (sent.isFailure) {
                    // feeder backed up: drop the oldest pending block to keep
                    // latency bounded rather than queueing unboundedly
                    output.tryReceive()
                    output.trySend(captured)
                    dropped++
                }
            }
            r.stop()
            r.release()
            record = null
        }, "rvc-capture").start()
        return true
    }

    fun stop() {
        running = false
        level.value = 0.0f
    }

    fun hasPermission(context: Context): Boolean =
        ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
}
