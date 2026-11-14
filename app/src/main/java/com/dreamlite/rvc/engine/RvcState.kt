package com.dreamlite.rvc.engine

import android.util.Log
import org.json.JSONObject

/**
 * The RVC session state machine, mirroring ML-Sharp-QNN's PipelineState.
 *
 * The native pipeline is a single long-lived object; this wraps it with a
 * coroutine-friendly state flow and owns the audio plumbing. The states are
 * the UI's only view of the pipeline.
 */
sealed interface RvcState {

    /** Idle: no pipeline loaded. */
    data object Idle : RvcState

    /** Voice graphs are being staged and the QNN contexts are loading. */
    data class Loading(val label: String) : RvcState

    /** Running: mic capture → pipeline → playback, all live. */
    data class Running(
        val rtf: Float,
        val msPerBlock: Float,
        val msIndex: Float,
        val blocks: Long,
        val peak: Float,
    ) : RvcState

    /** A conversion finished; [samples] is the 40 kHz output. */
    data class Done(val samples: FloatArray, val rtf: Float) : RvcState

    data class Error(val message: String) : RvcState
}

/**
 * Decodes the native pipelineStats JSON into the numbers the UI shows.
 * Keep the keys in sync with rvc_jni.cpp::pipelineStats.
 */
data class RvcStats(
    val msTotal: Float,
    val rtf: Float,
    val nBlocks: Long,
    val msIndex: Float,
    val msF0: Float,
    val msEnc: Float,
    val msSynth: Float,
    val msC0: Float,
    val msGn: Float,
    val msR: Float,
    val msSola: Float,
    val graphCalls: Long,
    val peak: Float,
) {
    companion object {
        private const val TAG = "rvc_stats"

        fun parse(json: String): RvcStats? = try {
            val o = JSONObject(json)
            RvcStats(
                msTotal = o.optDouble("ms_total", 0.0).toFloat(),
                rtf = o.optDouble("rtf", 1.0).toFloat(),
                nBlocks = o.optLong("n_blocks", 0),
                msIndex = o.optDouble("ms_index", 0.0).toFloat(),
                msF0 = o.optDouble("ms_f0", 0.0).toFloat(),
                msEnc = o.optDouble("ms_enc", 0.0).toFloat(),
                msSynth = o.optDouble("ms_synth", 0.0).toFloat(),
                msC0 = o.optDouble("ms_c0", 0.0).toFloat(),
                msGn = o.optDouble("ms_gn", 0.0).toFloat(),
                msR = o.optDouble("ms_r", 0.0).toFloat(),
                msSola = o.optDouble("ms_sola", 0.0).toFloat(),
                graphCalls = o.optLong("graph_calls", 0),
                // kept for the offline path, not used in live stats
                peak = o.optDouble("peak", 0.0).toFloat(),
            )
        } catch (e: Exception) {
            Log.w(TAG, "bad stats json: $json")
            null
        }
    }
}
