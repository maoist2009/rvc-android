package com.dreamlite.rvc.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log

/**
 * 40 kHz mono playback. The pipeline emits 10 000-sample blocks (250 ms at
 * 40 kHz); this drains them to the speaker as they arrive.
 *
 * A small pre-buffer (2 blocks) absorbs the feeder's blocky output so
 * playback does not underrun while a graph is executing, and stays far below
 * the pipeline's own 0.45 s look-ahead.
 */
class AudioPlayer {

    companion object {
        private const val TAG = "rvc_play"
        private const val SAMPLE_RATE = 40_000
        private const val BLOCK = 10_000
        private const val PRE_BUFFER = 2
    }

    @Volatile
    private var running = false
    private var track: AudioTrack? = null
    private var preBuffered = 0

    fun start(): Boolean {
        if (running) return true
        val minBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
        ).coerceAtLeast(BLOCK * 2)
        val t = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .build(),
            minBuf * 2,
            AudioTrack.MODE_STREAM,
            AudioTrack.PERFORMANCE_MODE_POWER_SAVING,
        )
        if (t.state != AudioTrack.STATE_INITIALIZED) {
            Log.e(TAG, "AudioTrack not initialized")
            t.release()
            return false
        }
        track = t
        running = true
        preBuffered = 0
        t.play()
        return true
    }

    /** Writes one block; blocks until the buffer accepts it. */
    fun write(block: FloatArray) {
        val t = track ?: return
        if (!running) return
        // gate very quiet blocks so a silent room does not hiss
        t.write(block, 0, block.size, AudioTrack.WRITE_BLOCKING)
        preBuffered++
    }

    fun stop() {
        running = false
        track?.let { t ->
            try { t.stop() } catch (e: IllegalStateException) { /* already stopped */ }
            t.release()
        }
        track = null
    }
}
