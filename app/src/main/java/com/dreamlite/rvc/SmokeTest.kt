package com.dreamlite.rvc

import android.app.Application
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

private fun JSONArray.toInts() = IntArray(length()) { getInt(it) }

/**
 * End-to-end proof that the app process can drive the NPU: extracts the
 * bundled hubert_c0 graph (conv0 of the HuBERT feature extractor, 94 KB),
 * loads it as a QNN HTP context and executes it against a synthetic 440 Hz
 * sine. A non-degenerate output means the whole chain — setenv, dlopen,
 * backendCreate, contextCreateFromBinary, graphExecute — worked in-app.
 *
 * This mirrors run_rvc_npu.py's _run_qnn("hubert_c0", ...) on the smallest of
 * the split HuBERT graphs; the rest of the pipeline graphs use the same
 * executor path with bigger tensors.
 */
class SmokeTest(private val app: Application) {

    data class Result(val ok: Boolean, val lines: List<String>)

    suspend fun run(): Result = withContext(Dispatchers.IO) {
        val lines = mutableListOf<String>()
        fun log(msg: String) {
            Log.i(TAG, msg)
            lines.add(msg)
        }
        try {
            val modelsDir = File(app.filesDir, "models").apply { mkdirs() }
            val binName = "hubert_c0_s23.bin"
            val bin = File(modelsDir, binName)
            if (!bin.exists()) {
                app.assets.open("models/$binName").use { input ->
                    bin.outputStream().use { input.copyTo(it) }
                }
                log("extracted $binName (${bin.length()} bytes)")
            } else {
                log("using existing $binName (${bin.length()} bytes)")
            }

            val libDir = app.applicationInfo.nativeLibraryDir
            nativeInitAndLog(libDir, ::log) ?.let { return@withContext Result(false, lines + it) }

            RvcJni.loadGraph(TAG_GRAPH, bin.absolutePath).takeIf { it.isNotEmpty() }
                ?.let { return@withContext Result(false, lines + "loadGraph: $it") }
            log("loadGraph '$TAG_GRAPH' ok")

            val info = JSONObject(RvcJni.graphInfo(TAG_GRAPH))
            val inName = info.getJSONArray("inputs").getJSONObject(0).getString("name")
            val inDims = info.getJSONArray("inputs").getJSONObject(0).getJSONArray("dims")
            val outName = info.getJSONArray("outputs").getJSONObject(0).getString("name")
            val outDims = info.getJSONArray("outputs").getJSONObject(0).getJSONArray("dims")
            log("graph: in $inName ${inDims.toInts().joinToString("x")}, out $outName ${outDims.toInts().joinToString("x")}")

            val n = inDims.getInt(1)
            // Unit-amplitude 440 Hz sine at 16 kHz: real signal, so conv0's
            // output is far from the all-zero degenerate case.
            val input = FloatArray(n) { sin(2.0 * PI * 440.0 * it / 16000.0).toFloat() }

            val t0 = System.currentTimeMillis()
            val outs = RvcJni.execGraph(TAG_GRAPH, arrayOf(inName), arrayOf(input))
            val ms = System.currentTimeMillis() - t0
            if (outs == null) {
                return@withContext Result(false, lines + "execGraph failed: ${RvcJni.lastError()}")
            }
            val out = outs[0]
            var mn = Float.POSITIVE_INFINITY
            var mx = Float.NEGATIVE_INFINITY
            var sum = 0.0
            var absSum = 0.0
            for (v in out) {
                if (v < mn) mn = v
                if (v > mx) mx = v
                sum += v
                absSum += abs(v)
            }
            val mean = sum / out.size
            val first = out.take(8).joinToString(" ") { String.format("%.4f", it) }
            log("execGraph ok in ${ms}ms -> ${out.size} floats")
            log("  min=$mn max=$mx mean=${String.format("%.6f", mean)} mean|.|=${String.format("%.6f", absSum / out.size)}")
            log("  first: $first")

            val degenerate = absSum < 1e-6f
            if (degenerate) {
                Result(false, lines + "OUTPUT DEGENERATE (all zeros)")
            } else {
                log("SMOKE TEST PASSED — NPU executed $outName in-app")
                Result(true, lines)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "smoke test threw", t)
            Result(false, lines + "exception: ${t.javaClass.name}: ${t.message}")
        }
    }

    private fun nativeInitAndLog(libDir: String, log: (String) -> Unit): String? {
        val err = RvcJni.nativeInit(libDir, libDir)
        return if (err.isEmpty()) {
            log("nativeInit ok (libDir=$libDir)")
            null
        } else {
            "nativeInit: $err"
        }
    }

    companion object {
        private const val TAG = "rvc_smoke"
        private const val TAG_GRAPH = "hubert_c0"
    }
}
