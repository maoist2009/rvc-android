package com.dreamlite.rvc

import android.app.Application
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * End-to-end pipeline test: loads the seven staged QNN contexts plus the conv0
 * GroupNorm affine, converts a raw 16 kHz mono float32 file through the whole
 * run_pipeline_ctx port and reports the RTF breakdown.
 *
 * The models and the input are staged into filesDir by the host (root copy,
 * see stage_rvc_models.sh) — the bins are ~440 MB, far past any sane asset
 * budget, and the input must match the reference render's resample_poly path.
 */
class PipelineTest(private val app: Application) {

    data class Result(val ok: Boolean, val lines: List<String>)

    suspend fun run(): Result = withContext(Dispatchers.IO) {
        val lines = mutableListOf<String>()
        fun log(msg: String) {
            Log.i(TAG, msg)
            lines.add(msg)
        }
        try {
            val modelsDir = File(app.filesDir, "models")
            val voiceJson = File(modelsDir, "voice.json")
            val input = File(app.filesDir, "test_input.f32")
            for ((name, f) in listOf("models" to modelsDir, "voice.json" to voiceJson,
                                     "test_input.f32" to input)) {
                if (!f.exists()) {
                    return@withContext Result(false, lines +
                        "missing $name (expected ${f.absolutePath})")
                }
            }
            log("models dir: ${modelsDir.absolutePath} (${modelsDir.list()?.size} entries)")
            log("input: ${input.length() / 4} samples (${input.length() / 4 / 16_000.0}s)")

            RvcJni.pipelineCreate().takeIf { it.isNotEmpty() }?.let {
                return@withContext Result(false, lines + "pipelineCreate: $it")
            }

            val initErr = RvcJni.pipelineInit(
                modelsDir.absolutePath,
                app.applicationInfo.nativeLibraryDir,
                voiceJson.readText()
            )
            if (initErr.isNotEmpty()) {
                return@withContext Result(false, lines + "pipelineInit: $initErr")
            }
            log("pipelineInit ok — 7 graphs + conv0_gn loaded")

            val x = readFloat32LE(input)
            log("input samples: ${x.size}")

            val t0 = System.currentTimeMillis()
            val out = RvcJni.pipelineConvert(x)
            val wall = System.currentTimeMillis() - t0
            if (out == null) {
                return@withContext Result(false, lines +
                    "pipelineConvert failed: ${RvcJni.lastError()}")
            }
            log("pipelineConvert ok in ${wall}ms -> ${out.size} floats " +
                "(${out.size / 40_000.0}s @40k)")

            val stats = RvcJni.pipelineStats()
            Log.i(TAG, "stats json: $stats")
            lines.add("stats: $stats")
            val timings = RvcJni.graphTimings()
            Log.i(TAG, "graph timings: $timings")
            lines.add("graph_timings: $timings")

            val outFile = File(app.filesDir, "test_output.f32")
            writeFloat32LE(outFile, out)
            log("wrote ${outFile.absolutePath} (${outFile.length()} bytes)")

            var peak = 0f
            var absSum = 0.0
            for (v in out) {
                val a = Math.abs(v)
                if (a > peak) peak = a
                absSum += a
            }
            log("out peak=$peak mean|.|=${absSum / out.size}")
            val degenerate = peak < 1e-4f
            if (degenerate) {
                Result(false, lines + "OUTPUT DEGENERATE (near-silent)")
            } else {
                Result(true, lines)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "pipeline test threw", t)
            Result(false, lines + "exception: ${t.javaClass.name}: ${t.message}")
        } finally {
            try { RvcJni.pipelineDestroy() } catch (t: Throwable) { }
        }
    }

    companion object {
        private const val TAG = "rvc_pipeline_test"
    }
}

/* Raw float32 files, little-endian — the layout numpy's .tofile/.fromfile use,
 * so the host can stage inputs and read back outputs without any conversion. */
private fun readFloat32LE(f: File): FloatArray {
    java.io.FileInputStream(f).use { fis ->
        java.nio.channels.FileChannel.MapMode.READ_ONLY.let { mode ->
            fis.channel.use { ch ->
                val bb = ch.map(mode, 0, f.length())
                bb.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val fb = bb.asFloatBuffer()
                return FloatArray(fb.remaining()) { fb.get() }
            }
        }
    }
}

private fun writeFloat32LE(f: File, x: FloatArray) {
    val bb = java.nio.ByteBuffer.allocate(x.size * 4)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(x)
    java.io.FileOutputStream(f).use { it.write(bb.array()) }
}
