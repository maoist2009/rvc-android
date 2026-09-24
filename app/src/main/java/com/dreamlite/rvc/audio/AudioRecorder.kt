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

/**
 * 16 kHz mono capture. Emits 4000-sample chunks (250 ms), the exact block
 * stride the streaming pipeline consumes, so the feeder never has to buffer
 * or split input.
 *
 * The capture loop runs on a dedicated thread; [running] is the stop signal.
 * Level is published as a 0..1 amplitude envelope for the UI meter.
 */
class AudioRecorder {

    companion object {
        private const val TAG = "rvc_rec"
        private const val SAMPLE_RATE = 16_000
        private const val BLOCK = 4000
    }

    @Volatile
    private var running = false
    private var record: AudioRecord? = null

    /** Peak amplitude of the most recent block, 0..1. */
    val level = MutableStateFlow(0.0f)

    val isRecording: Boolean get() = running

    @SuppressLint("MissingPermission")
    fun start(output: Channel<FloatArray>): Boolean {
        if (running) return true
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(BLOCK * 2)
        val r = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 2,
        )
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            r.release()
            return false
        }
        record = r
        running = true
        r.startRecording()

        Thread({
            val shortBuf = ShortArray(BLOCK)
            var peakHold = 0.0f
            while (running) {
                val n = r.read(shortBuf, 0, BLOCK, AudioRecord.READ_BLOCKING)
                if (n <= 0) continue
                val block = FloatArray(n)
                var peak = 0.0f
                for (i in 0 until n) {
                    val s = shortBuf[i] / 32768.0f
                    block[i] = s
                    val a = if (s < 0) -s else s
                    if (a > peak) peak = a
                }
                // 250 ms envelope: fast attack, slow release
                peakHold = if (peak > peakHold) peak else peakHold * 0.92f
                level.value = peakHold
                if (!running) break
                val sent = output.trySend(block)
                if (sent.isFailure) {
                    // feeder backed up: drop the oldest pending block to keep
                    // latency bounded rather than queueing unboundedly
                    output.tryReceive()
                    output.trySend(block)
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
