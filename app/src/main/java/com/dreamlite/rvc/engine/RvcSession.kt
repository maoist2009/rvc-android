package com.dreamlite.rvc.engine

import android.util.Log
import com.dreamlite.rvc.RvcJni
import com.dreamlite.rvc.audio.AudioPlayer
import com.dreamlite.rvc.audio.AudioRecorder
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.data.VoiceManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Orchestrates the whole conversion: mic capture -> streaming pipeline ->
 * playback, plus the recorded buffers kept for a "play converted" preview.
 *
 * Threading: capture is its own thread (AudioRecorder); the feeder coroutine
 * runs on Dispatchers.Default; all JNI calls happen off the main thread.
 * The native pipeline is single-consumer by construction.
 *
 * The class deliberately mirrors ML-Sharp-QNN's PipelineManager: one
 * app-level singleton, state exposed as a flow, all errors surfaced through
 * the state instead of exceptions.
 */
class RvcSession(
    private val voiceManager: VoiceManager,
    private val settings: SettingsRepository,
    private val libDir: String,
    private val filesDir: String,
) {

    companion object {
        private const val TAG = "rvc_session"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var runJob: Job? = null

    private val _state = MutableStateFlow<RvcState>(RvcState.Idle)
    val state: StateFlow<RvcState> = _state.asStateFlow()

    private val _level = MutableStateFlow(0.0f)
    val level: StateFlow<Float> = _level.asStateFlow()

    /** Last full converted output, kept for the preview button. */
    @Volatile
    var lastOutput: FloatArray? = null
        private set

    private val recorder = AudioRecorder()
    private var player: AudioPlayer? = null
    private var initialized = false

    /** True when the native pipeline is loaded for the current voice. */
    val isReady: Boolean get() = initialized

    /**
     * Loads the selected voice into the native pipeline. Idempotent: calling
     * it again with the same staged voice is a no-op. Returns null on success
     * or the error string.
     */
    suspend fun ensureInitialized(): String? = withContext(Dispatchers.Default) {
        if (initialized) return@withContext null
        val s = settings.settingsFlow.first()
        val voice = voiceManager.list().firstOrNull { it.name == s.voiceName }
            ?: voiceManager.list().firstOrNull()
            ?: return@withContext "no voice staged (add one under the voices directory)"
        _state.value = RvcState.Loading("Staging ${voice.name}…")
        val stageErr = voiceManager.stage(voice)
        if (stageErr != null) return@withContext "stage: $stageErr"

        _state.value = RvcState.Loading("Loading NPU graphs…")
        val voiceJson = readVoiceJson(voice.dir) ?: return@withContext "voice.json missing"
        val json = buildVoiceJson(voiceJson, s)
        val err = RvcJni.pipelineCreate().takeIf { it.isNotEmpty() }
            ?: RvcJni.pipelineInit(filesDir + "/models", libDir, json)
        if (err != null && err.isNotEmpty()) {
            _state.value = RvcState.Error("pipelineInit: $err")
            return@withContext err
        }
        initialized = true
        _state.value = RvcState.Idle
        null
    }

    /**
     * Starts a live conversion. Audio flows mic -> pipeline -> player as long
     * as [stop] is not called. [monitor] gates playback.
     */
    fun startLive(monitor: Boolean) {
        if (runJob?.isActive == true) return
        initialized = false // force reload in case the voice changed
        runJob = scope.launch {
            val initErr = ensureInitialized()
            if (initErr != null) {
                _state.value = RvcState.Error(initErr)
                return@launch
            }
            RvcJni.pipelineStreamStart()

            val input = Channel<FloatArray>(capacity = 8)
            if (!recorder.start(input)) {
                _state.value = RvcState.Error("AudioRecord init failed")
                return@launch
            }
            val monitorPlayer = if (monitor) {
                AudioPlayer().also { if (!it.start()) _state.value = RvcState.Error("AudioTrack init failed") }
            } else null
            player = monitorPlayer

            val out = ArrayList<FloatArray>()
            var outSamples = 0
            var primed = 0

            // feeder: consumes capture blocks, emits converted ones
            launch(Dispatchers.Default) {
                while (true) {
                    val block = input.receiveCatching().getOrNull() ?: break
                    val emitted = RvcJni.pipelineStreamFeed(block)
                    val stats = RvcStats.parse(RvcJni.pipelineStats())
                    if (emitted.isNotEmpty()) {
                        monitorPlayer?.write(emitted)
                        out.add(emitted)
                        outSamples += emitted.size
                    }
                    _state.value = RvcState.Running(
                        rtf = stats?.rtf ?: 1f,
                        msPerBlock = stats?.msTotal?.div((stats.nBlocks.coerceAtLeast(1))) ?: 0f,
                        msIndex = stats?.msIndex ?: 0f,
                        blocks = stats?.nBlocks ?: 0,
                        peak = computePeak(emitted),
                    )
                }
                // capture ended -> flush the tail
                val tail = RvcJni.pipelineStreamEnd()
                if (tail.isNotEmpty()) {
                    monitorPlayer?.write(tail)
                    out.add(tail)
                }
                monitorPlayer?.stop()
                val final = RvcStats.parse(RvcJni.pipelineStats())
                val merged = FloatArray(out.sumOf { it.size })
                var off = 0
                for (b in out) { System.arraycopy(b, 0, merged, off, b.size); off += b.size }
                lastOutput = merged
                _state.value = RvcState.Done(merged, final?.rtf ?: 1f)
                recorder.stop()
            }

            // the recorder owns its thread; nothing else to do here
        }
    }

    /** Stops capture and the pipeline; [lastOutput] is kept for preview. */
    fun stop() {
        recorder.stop()
        runJob?.cancel()
        runJob = null
        player?.stop()
        player = null
        if (_state.value is RvcState.Running) {
            _state.value = RvcState.Idle
        }
    }

    /** Plays back the last converted output. Returns null on success. */
    suspend fun playLast(): String? = withContext(Dispatchers.Default) {
        val out = lastOutput ?: return@withContext "no output yet"
        val p = AudioPlayer()
        if (!p.start()) return@withContext "AudioTrack init failed"
        val chunk = 10_000
        var i = 0
        while (i < out.size) {
            val n = minOf(chunk, out.size - i)
            p.write(out.copyOfRange(i, i + n))
            i += n
        }
        p.stop()
        null
    }

    private fun computePeak(block: FloatArray): Float {
        var p = 0.0f
        for (v in block) { val a = if (v < 0) -v else v; if (a > p) p = a }
        return p
    }

    private fun readVoiceJson(dir: java.io.File): String? {
        for (name in listOf("voice.json", "mygf_zbhy_gate.json")) {
            val f = java.io.File(dir, name)
            if (f.isFile) return f.readText()
        }
        // fall back to any *_gate.json / *_manifest.json
        dir.listFiles { f -> f.isFile && (f.name.endsWith("_gate.json") || f.name.endsWith("_manifest.json")) }
            ?.firstOrNull()?.let { return it.readText() }
        return null
    }

    /**
     * Merges the voice's own manifest with the user's retrieval settings,
     * because index_rate / nprobe are session-level, not checkpoint-level.
     */
    private fun buildVoiceJson(voiceJson: String, s: SettingsRepository.Settings): String {
        return try {
            val o = org.json.JSONObject(voiceJson)
            o.put("index_rate", s.indexRate.toDouble())
            o.put("nprobe", s.nprobe)
            o.put("sid", o.optInt("sid", 0))
            o.toString()
        } catch (e: Exception) {
            Log.w(TAG, "voice json not an object; using defaults")
            "{\"index_rate\":${s.indexRate},\"nprobe\":${s.nprobe}}"
        }
    }

    fun destroy() {
        stop()
        recorder.stop()
        RvcJni.pipelineDestroy()
        initialized = false
    }
}
