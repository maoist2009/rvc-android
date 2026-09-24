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

    private val indexNames = listOf("index.path", "retrieval.index")

    private val _voicesFlow = MutableStateFlow<List<Voice>>(emptyList())
    /** The voice list as a flow; refreshed by [rescan] and on first access. */
    val voicesFlow: StateFlow<List<Voice>> = _voicesFlow.asStateFlow()

    init { rescan() }

    /** Re-reads the voices directory. Call after copying a voice in. */
    fun rescan() { _voicesFlow.value = list() }

    /** Lists every folder under voices/ that has all required graphs. */
    fun list(): List<Voice> {
        if (!voicesDir.isDirectory) return emptyList()
        return voicesDir.listFiles { f -> f.isDirectory }
            ?.sortedBy { it.name }
            ?.mapNotNull { dir ->
                val hasAll = requiredGraphs.all { File(dir, it).isFile }
                if (!hasAll) null else Voice(
                    name = dir.name,
                    dir = dir,
                    hasIndex = indexNames.any { File(dir, it).isFile },
                )
            }
            ?: emptyList()
    }

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
     * Stages [voice] into models/. Removes only the files this manager owns
     * (graphs, voice.json, index, marker) so nothing else in models/ is touched.
     * Returns null on success, or the failure reason.
     */
    fun stage(voice: Voice): String? {
        return try {
            if (!modelsDir.isDirectory && !modelsDir.mkdirs())
                return "cannot create models dir"
            cleanStaged()
            val linked = linkOrCopy(voice.dir, modelsDir)
            if (linked == null) return "cannot link/copy graphs"
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
     * Links each file of [src] into [dst] (preferred) or copies it. Returns the
     * staging mode used, or null when both fail.
     */
    private fun linkOrCopy(src: File, dst: File): String? {
        val files = src.listFiles { f -> f.isFile }?.toList() ?: return null
        // A relative symlink from models/ back to voices/<name>/ — both live
        // under filesDir, so "../voices/<name>" is the same path from the
        // perspective of the models/ directory.
        val rel = try {
            src.relativeTo(context.filesDir).path
        } catch (e: RuntimeException) {
            null
        }
        for (f in files) {
            val target = File(dst, f.name)
            target.delete()
            val linked = if (rel != null) {
                try {
                    val ok = Runtime.getRuntime().exec(
                        arrayOf("ln", "-s", "../$rel/${f.name}", target.absolutePath)
                    ).waitFor() == 0
                    if (ok) true else { target.delete(); false }
                } catch (e: Exception) {
                    target.delete(); false
                }
            } else false
            if (!linked) {
                if (!copyFile(f, target)) return null
            }
        }
        return if (files.any { File(dst, it.name).isSymbolicLink() }) "link" else "copy"
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
