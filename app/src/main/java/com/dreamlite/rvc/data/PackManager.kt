package com.dreamlite.rvc.data

import android.content.Context
import android.util.Log
import com.dreamlite.rvc.RvcJni
import java.io.File

/**
 * Imports voice packs and the shared component pack from a user-visible
 * directory the file manager can reach.
 *
 * Convention — drop files under [packsDir]:
 * ```
 * packs/shared/         hubert_c0_s23.bin, hubert_r_2063_s23.bin,
 *                       hubert_enc_140_s23.bin, rmvpe_e2e_s23.bin,
 *                       conv0_gn.f32
 * packs/voices/<name>/  encflow*.bin, flowreverse*.bin, decnsf*.bin,
 *                       voice.json (or *_gate.json), optional *.index
 * ```
 * Graph files may carry a tag suffix (`encflow_mygf_wh_s23.bin`); they are
 * matched by role prefix and installed under their canonical name.
 */
class PackManager(
    private val context: Context,
    private val voiceManager: VoiceManager,
) {

    companion object {
        private const val TAG = "rvc_pack"

        /** Shared graphs live here after install; VoiceManager stages from it. */
        const val SHARED_VOICE_NAME = VoiceManager.SHARED_VOICE_NAME

        private val SHARED_ROLES = listOf(
            "hubert_c0_s23" to 94_208L,
            "hubert_r_2063_s23" to 10_645_504L,
            "hubert_enc_140_s23" to 189_980_672L,
            "rmvpe_e2e_s23" to 184_795_136L,
        )

        /** conv0_gn.f32 is exactly 512 weights + 512 biases, LE float32. */
        private const val CONV0_GN = "conv0_gn.f32"
        private const val CONV0_GN_BYTES = 4_096L

        private val VOICE_ROLES = listOf("encflow", "flowreverse", "decnsf_ext")

        /** Voice graphs vary with checkpoint size; only sanity-check the floor. */
        private const val VOICE_BIN_MIN = 1_000_000L
        private const val VOICE_BIN_MAX = 100_000_000L
    }

    /** Where the user drops packs. External so a file manager can reach it. */
    val packsDir: File get() = File(context.getExternalFilesDir(null), "packs")
    val sharedPacksDir: File get() = File(packsDir, "shared")
    val voicePacksDir: File get() = File(packsDir, "voices")

    private val _state = kotlinx.coroutines.flow.MutableStateFlow(ScanResult())
    val state: kotlinx.coroutines.flow.StateFlow<ScanResult> = _state

    /** Canonical install locations. */
    private val installedSharedDir: File
        get() = File(voiceManager.voicesDir, SHARED_VOICE_NAME)

    data class Problem(val file: String, val reason: String)

    data class SharedPack(
        val dir: File,
        val found: Map<String, File>,
        val problems: List<Problem>,
    ) {
        val ready: Boolean get() = problems.isEmpty() &&
            found.size >= SHARED_ROLES.size + 1
    }

    data class VoicePack(
        val name: String,
        val dir: File,
        val graphs: Map<String, File>,
        val manifest: File?,
        val index: File?,
        val problems: List<Problem>,
    ) {
        val ready: Boolean get() = problems.isEmpty() && graphs.size == VOICE_ROLES.size && manifest != null
    }

    data class ScanResult(
        val shared: SharedPack? = null,
        val voices: List<VoicePack> = emptyList(),
        val installedShared: Boolean = false,
        val installedVoices: List<String> = emptyList(),
    )

    /** Re-reads the packs directory. Call after the user drops files in. */
    fun scan() {
        _state.value = ScanResult(
            shared = scanShared(),
            voices = scanVoices(),
            installedShared = installedSharedDir.isDirectory &&
                SHARED_ROLES.all { File(installedSharedDir, it.first + ".bin").isFile } &&
                File(installedSharedDir, CONV0_GN).isFile,
            installedVoices = voiceManager.list().map { it.name }
                .filter { it != SHARED_VOICE_NAME },
        )
    }

    private fun scanShared(): SharedPack? {
        val dir = sharedPacksDir
        if (!dir.isDirectory) return null
        val found = mutableMapOf<String, File>()
        val problems = mutableListOf<Problem>()
        for ((role, size) in SHARED_ROLES) {
            val f = findRole(dir, role)
            when {
                f == null -> problems += Problem("$role.bin", "missing")
                f.length() < size / 2 -> problems += Problem(f.name, "truncated (${f.length()} B)")
                else -> found[role] = f
            }
        }
        val g = File(dir, CONV0_GN).takeIf { it.isFile }
        when {
            g == null -> problems += Problem(CONV0_GN, "missing")
            g.length() != CONV0_GN_BYTES ->
                problems += Problem(g.name, "${g.length()} B, expected $CONV0_GN_BYTES")
            else -> found[CONV0_GN] = g
        }
        return SharedPack(dir, found, problems)
    }

    private fun scanVoices(): List<VoicePack> {
        val dir = voicePacksDir
        if (!dir.isDirectory) return emptyList()
        return dir.listFiles { f -> f.isDirectory }
            ?.sortedBy { it.name }
            ?.map { scanVoicePack(it) }
            ?: emptyList()
    }

    private fun scanVoicePack(dir: File): VoicePack {
        val graphs = mutableMapOf<String, File>()
        val problems = mutableListOf<Problem>()
        for (role in VOICE_ROLES) {
            val f = findRole(dir, role)
            when {
                f == null -> problems += Problem("$role.bin", "missing")
                f.length() !in VOICE_BIN_MIN..VOICE_BIN_MAX ->
                    problems += Problem(f.name, "${f.length()} B is outside the sane range")
                else -> graphs[role] = f
            }
        }
        val manifest = pickManifest(dir)
        if (manifest == null) problems += Problem("voice.json", "missing")
        else problems += validateManifest(manifest)

        val index = dir.listFiles { f -> f.isFile && f.name.endsWith(".index") }
            ?.maxByOrNull { it.length() }
        return VoicePack(
            name = dir.name,
            dir = dir,
            graphs = graphs,
            manifest = manifest,
            index = index,
            problems = problems,
        )
    }

    /**
     * Installs the shared pack: copies the five components into the shared
     * voice directory under canonical names. Returns null on success.
     */
    suspend fun installShared(): String? {
        val pack = scanShared() ?: return "no shared pack under ${sharedPacksDir.absolutePath}"
        if (!pack.ready) return pack.problems.joinToString("; ") { "${it.file}: ${it.reason}" }
        return try {
            installInto(pack.found, installedSharedDir)
            scan()
            null
        } catch (e: Exception) {
            Log.e(TAG, "installShared failed", e)
            e.message ?: e.javaClass.simpleName
        }
    }

    /**
     * Installs one voice pack under [voiceManager]'s voices directory.
     * Graphs are renamed to the canonical names the pipeline expects.
     * Returns null on success.
     */
    suspend fun installVoice(pack: VoicePack): String? {
        if (!pack.ready) return pack.problems.joinToString("; ") { "${it.file}: ${it.reason}" }
        val target = File(voiceManager.voicesDir, pack.name)
        return try {
            val entries = buildMap {
                pack.graphs.forEach { (role, f) -> put("$role.bin", f) }
                put("voice.json", pack.manifest!!)
                pack.index?.let { put("retrieval.index", it) }
            }
            installInto(entries, target)
            voiceManager.rescan()
            scan()
            null
        } catch (e: Exception) {
            Log.e(TAG, "installVoice ${pack.name} failed", e)
            e.message ?: e.javaClass.simpleName
        }
    }

    private fun installInto(entries: Map<String, File>, target: File) {
        if (!target.isDirectory && !target.mkdirs())
            throw java.io.IOException("cannot create ${target.absolutePath}")
        for ((name, src) in entries) {
            val dst = File(target, name)
            if (!src.inputStream().use { i ->
                    dst.outputStream().use { o -> i.copyTo(o, 256 * 1024); true }
                }
            ) throw java.io.IOException("copy failed: $name")
        }
    }

    /**
     * Loads the pack's encflow context on the NPU and checks its IO against
     * the manifest's layout constants — the check that catches a T30 pack
     * staged against a ctx-280 voice.json before the pipeline ever runs.
     * Heavy (NPU + ~15 MB context); call off the main thread, once per pack.
     * Returns null when everything matches, otherwise what failed.
     */
    suspend fun validateDeep(pack: VoicePack): String? {
        val manifest = pack.manifest ?: return "no manifest"
        val o = try {
            org.json.JSONObject(manifest.readText())
        } catch (e: Exception) {
            return "manifest is not valid json"
        }
        val tPh = o.optInt("t_ph", 280)
        val tFlow = o.optInt("t_flow", 54)

        val libDir = context.applicationInfo.nativeLibraryDir
        val err = RvcJni.nativeInit(libDir, libDir)
        if (err.isNotEmpty()) return "nativeInit: $err"
        val bin = pack.graphs["encflow"] ?: return "encflow missing"
        val tag = "packcheck_encflow"
        val loadErr = RvcJni.loadGraph(tag, bin.absolutePath)
        if (loadErr.isNotEmpty()) {
            RvcJni.shutdown()
            return "loadGraph encflow: $loadErr"
        }
        val info = try {
            org.json.JSONObject(RvcJni.graphInfo(tag))
        } catch (e: Exception) {
            RvcJni.freeGraph(tag); RvcJni.shutdown()
            return "graphInfo: ${e.message}"
        }
        RvcJni.freeGraph(tag)
        RvcJni.shutdown()

        val dims = { side: String, idx: Int ->
            info.optJSONArray(side)?.optJSONObject(idx)?.optJSONArray("dims")
                ?.let { d -> List(d.length()) { d.optInt(it) } }
        }
        // encflow: phone (1, t_ph, 768), pitch (1, t_ph), sid (1,)
        val phone = dims("inputs", 0)
        if (phone != listOf(1, tPh, 768))
            return "phone dims $phone != [1, $tPh, 768] — pack layout does not match voice.json t_ph"
        // outputs: m_p (1,192,t_flow), logs_p (1,192,t_flow), g (1,256,1)
        val out0 = dims("outputs", 0)
        val out1 = dims("outputs", 1)
        val out2 = dims("outputs", 2)
        if (out0?.takeIf { it.size == 3 }?.last() != tFlow)
            return "output_0 dims $out0 — last dim must be t_flow=$tFlow"
        if (out1 != out0) return "output_1 $out1 must match output_0 (logs_p has the same shape)"
        if (out2 != listOf(1, 256, 1)) return "speaker embedding dims $out2 != [1, 256, 1]"
        return null
    }

    // ---------- helpers ----------

    /** Finds a file whose stem starts with [role], e.g. encflow_mygf_wh_s23. */
    private fun findRole(dir: File, role: String): File? =
        dir.listFiles { f -> f.isFile }?.firstOrNull { f ->
            f.name.substringBeforeLast('.').startsWith(role)
        }

    private fun pickManifest(dir: File): File? {
        val direct = File(dir, "voice.json")
        if (direct.isFile) return direct
        return dir.listFiles { f -> f.isFile }
            ?.firstOrNull { f ->
                f.name.endsWith("_gate.json") || f.name.endsWith("_manifest.json")
            }
    }

    private fun validateManifest(f: File): List<Problem> {
        val o = try {
            org.json.JSONObject(f.readText())
        } catch (e: Exception) {
            return listOf(Problem(f.name, "not valid json"))
        }
        val required = listOf(
            "sr", "t_ph", "t_out", "t_flow", "flow_head", "dec_head",
            "upp", "sine_amp", "dim", "l_w", "l_b", "n_spk",
        )
        return required.filter { !o.has(it) }
            .map { Problem(f.name, "missing key $it") }
            .ifEmpty { emptyList() }
    }
}
