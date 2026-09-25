package com.dreamlite.rvc.data

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decodes any media file Android understands (mp3, m4a, wav, ogg, flac…) into
 * the 16 kHz mono float32 buffer the pipeline consumes.
 *
 * MediaExtractor + MediaCodec is used rather than a bundled decoder: it keeps
 * the APK free of native codec libraries and handles every format the device
 * platform supports. Output is resampled to 16 kHz with a linear interpolator
 * — good enough for a feature-extraction front end (the pipeline's own mel
 * extraction resamples again internally).
 *
 * The class is a namespace of one function; instances are not needed.
 */
object AudioDecoder {

    private const val TAG = "rvc_decode"
    private const val TARGET_SR = 16_000

    /** Max decode duration in seconds, so a stray long file cannot OOM. */
    private const val MAX_SECONDS = 600

    /**
     * Decodes [src] to 16 kHz mono f32. Returns the samples, or null with a
     * reason logged.
     */
    fun decodeTo16kMono(src: File): FloatArray? {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var codecStarted = false
        return try {
            extractor.setDataSource(src.absolutePath)
            val track = pickAudioTrack(extractor) ?: run {
                Log.e(TAG, "no audio track in ${src.name}")
                return null
            }
            val inputFormat = extractor.getTrackFormat(track)
            extractor.selectTrack(track)

            val srcRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val srcChannels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)!!
            codec = try {
                MediaCodec.createDecoderByType(mime)
            } catch (e: Exception) {
                Log.e(TAG, "no decoder for $mime", e)
                return null
            }
            codec.configure(inputFormat, null, null, 0)
            codec.start()
            codecStarted = true

            val pcm = ArrayList<Float>(1 shl 18)
            val info = MediaCodec.BufferInfo()
            val timeoutUs = 10_000L
            val maxSamples = TARGET_SR * MAX_SECONDS
            var done = false
            var outRate = srcRate
            var outChannels = srcChannels

            while (!done) {
                val inIdx = codec.dequeueInputBuffer(timeoutUs)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val n = extractor.readSampleData(inBuf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    } else {
                        codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
                var sawEOS = false
                while (true) {
                    val outIdx = codec.dequeueOutputBuffer(info, timeoutUs)
                    when {
                        outIdx >= 0 -> {
                            val outBuf = codec.getOutputBuffer(outIdx)!!
                            appendPcm(outBuf, info, outChannels, pcm)
                            if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) sawEOS = true
                            codec.releaseOutputBuffer(outIdx, false)
                            if (pcm.size > maxSamples) done = true
                        }
                        outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val f = codec.outputFormat
                            outRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            outChannels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            Log.i(TAG, "output format: ${outRate}Hz $outChannels ch")
                        }
                        else -> break
                    }
                }
                if (sawEOS || (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) done = true
            }

            codec.stop()
            codec.release()
            codecStarted = false
            extractor.release()

            if (pcm.isEmpty()) {
                Log.e(TAG, "decoded no samples from ${src.name}")
                null
            } else {
                if (outRate == TARGET_SR) pcm.toFloatArray()
                else resampleLinear(pcm.toFloatArray(), outRate, TARGET_SR)
            }
        } catch (e: Exception) {
            Log.e(TAG, "decode failed for ${src.name}", e)
            try { extractor.release() } catch (_: Exception) {}
            // A leaked decoder keeps its hardware session and can poison the
            // next instance (observed: two concurrent decoders abort).
            if (codecStarted && codec != null) {
                try { codec.stop() } catch (_: Exception) {}
                try { codec.release() } catch (_: Exception) {}
            }
            null
        }
    }

    private fun pickAudioTrack(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return null
    }

    /** S16LE PCM → floats, deinterleaved to mono by channel averaging. */
    private fun appendPcm(
        buf: ByteBuffer, info: MediaCodec.BufferInfo, channels: Int, out: ArrayList<Float>
    ) {
        if (info.size <= 0) return
        buf.position(info.offset)
        buf.limit(info.offset + info.size)
        val frames = info.size / (2 * channels)
        var i = 0
        while (i < frames) {
            var acc = 0
            for (c in 0 until channels) {
                acc += buf.getShort().toInt()
            }
            out.add(acc / (channels * 32768.0f))
            i++
        }
    }

    private fun resampleLinear(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to) return x
        val n = (x.size.toLong() * to / from).toInt()
        val y = FloatArray(n)
        val step = from.toDouble() / to
        for (i in 0 until n) {
            val pos = i * step
            val i0 = pos.toInt()
            val i1 = minOf(i0 + 1, x.size - 1)
            val f = (pos - i0).toFloat()
            y[i] = x[i0] * (1f - f) + x[i1] * f
        }
        return y
    }

    /** Reads a raw LE float32 file directly (the pipeline's native exchange format). */
    fun readF32(src: File): FloatArray? = try {
        val b = ByteBuffer.wrap(src.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val f = FloatArray(b.limit() / 4)
        var i = 0
        while (b.hasRemaining() && i < f.size) { f[i++] = b.float }
        f
    } catch (e: Exception) {
        Log.e(TAG, "readF32 failed", e)
        null
    }
}
