package com.dreamlite.rvc

import android.util.Log

/**
 * Native interface to librvc_qnn.so.
 *
 * Every call that can fail returns "" on success and an error string
 * otherwise; [execGraph] returns null on failure with the reason available
 * from [lastError]. Heavy work runs on the calling thread — callers must
 * schedule off the main thread.
 */
object RvcJni {

    /** Initialise the QNN HTP backend. libDir/skelDir are the app's own native
     * library dir, which carries libQnnHtp.so and libQnnHtpV79Skel.so. */
    external fun nativeInit(libDir: String, skelDir: String): String

    external fun loadGraph(tag: String, binPath: String): String

    external fun freeGraph(tag: String): String

    /** Graph metadata as JSON: {"tag","inputs":[{"name","dtype","dims"}],...}. */
    external fun graphInfo(tag: String): String

    external fun shutdown()

    external fun lastError(): String

    /**
     * Execute a loaded graph. [data] entries are FloatArray for float32
     * tensors and IntArray for int32 tensors, matched to graph inputs by
     * [names]. Outputs come back in graph declaration order as FloatArray;
     * null means failure (see [lastError]).
     */
    external fun execGraph(
        tag: String,
        names: Array<String>,
        data: Array<Any>
    ): Array<FloatArray>?

    /* ---------- full pipeline ---------- */

    /** Create the pipeline handle. "" on success. */
    external fun pipelineCreate(): String

    /** Load the seven QNN contexts and the conv0 GroupNorm affine.
     * [libDir] is the app's nativeLibraryDir (the HTP stub resolves the skel
     * from there); [voiceJson] is the voice's *_gate.json content, with
     * missing keys falling back to the mygf_zbhy ctx-280 defaults. */
    external fun pipelineInit(modelsDir: String, libDir: String,
                              voiceJson: String): String

    /** Convert 16 kHz mono audio. null on failure (see [lastError]). */
    external fun pipelineConvert(input: FloatArray): FloatArray?

    /** Timing breakdown of the last convert as JSON. */
    external fun pipelineStats(): String

    external fun pipelineDestroy()

    /** Per-graph graphExecute wall-time stats as a JSON array. */
    external fun graphTimings(): String

    /* ---------- streaming pipeline ---------- */

    /** Reset the streaming state. "" on success. */
    external fun pipelineStreamStart(): String

    /** Append 16 kHz mono audio and return every block that just became
     * computable (10 000 samples @40 kHz per 250 ms of input). Empty means
     * either "nothing to emit yet" or failure — check [pipelineLastError]
     * when the input was long enough to emit. */
    external fun pipelineStreamFeed(input: FloatArray): FloatArray

    /** Pad the input so the final block emits and flush it. */
    external fun pipelineStreamEnd(): FloatArray

    external fun pipelineLastError(): String

    fun load(): Boolean = try {
        System.loadLibrary("rvc_qnn")
        true
    } catch (e: Throwable) {
        Log.e(TAG, "loadLibrary rvc_qnn failed", e)
        false
    }

    private const val TAG = "rvc_jni"
}
