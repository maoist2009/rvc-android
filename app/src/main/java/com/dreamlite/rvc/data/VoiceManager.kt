package com.dreamlite.rvc.data

import android.content.Context
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Voice discovery and staging.
 *
 * The native pipeline reads its graphs from a flat models directory
 * (filesDir/models), but users think in terms of voices. A voice is a folder
 * under filesDir/voices/&lt;name&gt;/ holding the seven .bin graphs plus voice.json
 * (plus an optional retrieval index). Selecting a voice symlinks/copies its
 * contents into models/ so the native side finds the expected flat layout.
 *
 * Symlinks are used when the filesystem allows them, because the graph set is
 * ~218 MB per voice and copying on every switch would be slow; the fallback is
 * a plain copy.
 */
class VoiceManager(private val context: Context) {

    companion object {
        /** Holds the shared components (HuBERT/RMVPE/conv0_gn) shared by all voices. */
        const val SHARED_VOICE_NAME = "shared"
    }

    data class Voice(
        val name: String,
        val dir: File,
        val hasIndex: Boolean,
    )

    val voicesDir: File get() = File(context.filesDir, "voices")
    val modelsDir: File get() = File(context.filesDir, "models")

    /** The graph files the native pipeline expects in the models directory. */
    private val requiredGraphs = listOf(
        "hubert_c0_s23.bin",
        "hubert_r_2063_s23.bin",
        "hubert_enc_140_s23.bin",
        "rmvpe_e2e_s23.bin",
        "encflow.bin",
        "flowreverse.bin",
        "decnsf_ext.bin",
        "conv0_gn.f32",
    )

    /** Only the voice-specific graphs; shared ones come from the shared dir. */
    private val voiceRoles = listOf(
        "encflow.bin",
        "flowreverse.bin",
        "decnsf_ext.bin",
    )

    private val indexNames = listOf("index.path", "retrieval.index")

    /** Files the pipeline needs but that are shared across voices. */
    private val sharedRoles = listOf(
        "hubert_c0_s23.bin",
        "hubert_r_2063_s23.bin",
        "hubert_enc_140_s23.bin",
        "rmvpe_e2e_s23.bin",
        "conv0_gn.f32",
    )

    private val _voicesFlow = MutableStateFlow<List<Voice>>(emptyList())
    /** The voice list as a flow; refreshed by [rescan] and on first access. */
    val voicesFlow: StateFlow<List<Voice>> = _voicesFlow.asStateFlow()

    init { rescan() }

    /** Re-reads the voices directory. Call after copying a voice in. */
    fun rescan() { _voicesFlow.value = list() }

    /**
     * Lists every voice folder that has all its OWN graphs and a manifest.
     * [SHARED_VOICE_NAME] is excluded — it holds the shared components only.
     * The shared graphs are checked separately by [sharedReady].
     */
    fun list(): List<Voice> {
        if (!voicesDir.isDirectory) return emptyList()
        return voicesDir.listFiles { f -> f.isDirectory }
            ?.sortedBy { it.name }
            ?.mapNotNull { dir ->
                if (dir.name == SHARED_VOICE_NAME) return@mapNotNull null
                val hasAll = voiceRoles.all { File(dir, it).isFile }
                if (!hasAll) null else Voice(
                    name = dir.name,
                    dir = dir,
                    hasIndex = indexNames.any { File(dir, it).isFile },
                )
            }
            ?: emptyList()
    }

    /** True when the shared component directory is complete. */
    fun sharedReady(): Boolean =
        sharedRoles.all { File(voicesDir, "$SHARED_VOICE_NAME/$it").isFile }

    /** A voice is playable when its own graphs AND the shared ones exist. */
    fun isPlayable(voice: Voice): Boolean = sharedReady() &&
        File(voice.dir, "voice.json").isFile()

    /** True when models/ already holds a complete, matching voice layout. */
    fun isStaged(): Boolean =
        requiredGraphs.all { File(modelsDir, it).isFile() } &&
            File(modelsDir, "voice.json").isFile()

    /** Which staged voice the models dir currently reflects, or null. */
    fun stagedVoiceName(): String? {
        if (!isStaged()) return null
        val marker = File(modelsDir, ".voice_name")
        if (marker.isFile) return marker.readText().trim().ifEmpty { null }
        return null
    }

    /**
     * Stages [voice] into models/. Links the shared components first (they are
     * the same for every voice), then the voice-specific graphs and manifest.
     * Removes only the files this manager owns (graphs, voice.json, index,
     * marker) so nothing else in models/ is touched.
     * Returns null on success, or the failure reason.
     */
    fun stage(voice: Voice): String? {
        return try {
            if (!modelsDir.isDirectory && !modelsDir.mkdirs())
                return "cannot create models dir"
            val sharedDir = File(voicesDir, SHARED_VOICE_NAME)
            // voices/shared/ is the pack layout; a self-contained voice folder
            // (one that carries the shared graphs itself) is equally valid.
            val sharedSrc = if (sharedReady()) sharedDir else voice.dir
            if (!sharedRoles.all { File(sharedSrc, it).isFile })
                return "shared components missing — import the shared pack first"
            cleanStaged()

            // Shared: link from the shared pack (relative symlink still works,
            // both live under filesDir).
            val sharedErr = linkOneOrCopy(sharedSrc, modelsDir, sharedRoles)
            if (sharedErr != null) return sharedErr
            // Voice-specific.
            val voiceErr = linkOneOrCopy(
                voice.dir, modelsDir,
                voiceRoles + indexNames.filter { File(voice.dir, it).isFile() })
            if (voiceErr != null) return voiceErr
            // Manifest, always named voice.json on the staged side.
            val manifest = pickManifest(voice.dir) ?: return "voice.json missing"
            val dst = File(modelsDir, "voice.json")
            dst.delete()
            if (!copyFile(manifest, dst)) return "cannot copy manifest"

            File(modelsDir, ".voice_name").writeText(voice.name)
            null
        } catch (e: Exception) {
            e.message ?: e.javaClass.simpleName
        }
    }

    private fun cleanStaged() {
        val owned = requiredGraphs + "voice.json" + indexNames + ".voice_name"
        for (name in owned) {
            val f = File(modelsDir, name)
            if (f.isFile) f.delete() else if (f.isDirectory) f.deleteRecursively()
        }
    }

    /**
     * Links the named files of [src] into [dst] (preferred) or copies them.
     * Files that do not exist in [src] are skipped — the caller passes only
     * the ones it knows are present. Returns null when every file landed.
     */
    private fun linkOneOrCopy(src: File, dst: File, names: List<String>): String? {
        val rel = try {
            src.relativeTo(context.filesDir).path
        } catch (e: RuntimeException) {
            null
        }
        for (name in names) {
            val f = File(src, name)
            if (!f.isFile) continue
            val target = File(dst, name)
            target.delete()
            val linked = if (rel != null) {
                try {
                    val ok = Runtime.getRuntime().exec(
                        arrayOf("ln", "-s", "../$rel/$name", target.absolutePath)
                    ).waitFor() == 0
                    if (ok) true else { target.delete(); false }
                } catch (e: Exception) {
                    target.delete(); false
                }
            } else false
            if (!linked) {
                if (!copyFile(f, target)) return "cannot link or copy $name"
            }
        }
        return null
    }

    private fun pickManifest(dir: File): File? {
        val direct = File(dir, "voice.json")
        if (direct.isFile) return direct
        return dir.listFiles { f -> f.isFile }
            ?.firstOrNull { f -> f.name.endsWith("_gate.json") || f.name.endsWith("_manifest.json") }
    }

    private fun copyFile(src: File, dst: File): Boolean = try {
        src.inputStream().use { input ->
            dst.outputStream().use { output -> input.copyTo(output, 64 * 1024) }
        }
        true
    } catch (e: Exception) {
        false
    }

    private fun File.isSymbolicLink(): Boolean = try {
        java.nio.file.Files.isSymbolicLink(toPath())
    } catch (e: Exception) {
        false
    }
}
