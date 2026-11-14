package com.dreamlite.rvc.engine

import android.util.Log
import com.dreamlite.rvc.RvcJni
import com.dreamlite.rvc.audio.AudioPlayer
import com.dreamlite.rvc.audio.AudioRecorder
import com.dreamlite.rvc.audio.CaptureBlock
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
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
    private val appContext: android.content.Context,
    private val voiceManager: VoiceManager,
    private val settings: SettingsRepository,
    private val libDir: String,
    private val filesDir: String,
) {

    companion object {
        private const val TAG = "rvc_session"

        /** Live preview keeps the most recent 120 s; the mic runs for hours. */
        private const val LIVE_PREVIEW_SAMPLES = 40_000 * 120
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var runJob: Job? = null

    private val _state = MutableStateFlow<RvcState>(RvcState.Idle)
    val state: StateFlow<RvcState> = _state.asStateFlow()

    private val _level = MutableStateFlow(0.0f)
    val level: StateFlow<Float> = _level.asStateFlow()

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    /** Last full converted output, kept for the preview button. */
    @Volatile
    var lastOutput: FloatArray? = null
        private set

    /** Sample rate of the currently staged voice, for playback. */
    @Volatile
    var voiceSr: Int = AudioPlayer.DEFAULT_SR
        private set

    /** Debug: log a per-block stage breakdown (set by the measure receiver). */
    @Volatile
    private var blockTrace = false

    fun setBlockTrace(on: Boolean) {
        blockTrace = on
    }

    private val recorder = AudioRecorder()
    private var player: AudioPlayer? = null
    private var previewPlayer: AudioPlayer? = null
    private var initialized = false

    /** Which settings the loaded pipeline was built for; a change forces a
     * reload, because index_rate/nprobe are baked into pipelineInit. */
    private var loadedSignature: String? = null

    /** The voice name models/ was staged to when the pipeline was loaded. */
    private var loadedVoice: String? = null

    /** Single-flights [ensureInitialized]; concurrent callers would stage the
     * same models/ directory against each other. */
    private val initMutex = Mutex()

    /** True when the native pipeline is loaded for the current voice. */
    val isReady: Boolean get() = initialized

    /**
     * Loads the selected voice into the native pipeline. Idempotent: calling
     * it again with the same staged voice is a no-op. Returns null on success
     * or the error string.
     */
    suspend fun ensureInitialized(): String? = initMutex.withLock {
        withContext(Dispatchers.Default) {
            val s = settings.settingsFlow.first()
            // A voice switch or a retrieval-settings change alters what
            // pipelineInit builds, so re-stage and reload rather than reusing
            // the contexts from the previous configuration. The staged marker
            // is also re-checked: if models/ no longer matches what we loaded
            // (crash mid-stage, files replaced), rebuild instead of trusting it.
            val signature = "${s.voiceName}|${s.indexRate}|${s.nprobe}"
            val stagedNow = loadedVoice?.let { voiceManager.stagedVoiceName() }
            if (initialized && signature == loadedSignature && stagedNow == loadedVoice)
                return@withContext null
            val voice = voiceManager.list().firstOrNull { it.name == s.voiceName }
                ?: voiceManager.list().firstOrNull()
                ?: return@withContext "no voice staged (add one under the voices directory)"
            _state.value = RvcState.Loading("Staging ${voice.name}…")
            val stageErr = voiceManager.stage(voice)
            if (stageErr != null) return@withContext "stage: $stageErr"

            _state.value = RvcState.Loading("Loading NPU graphs…")
            val voiceJson = readVoiceJson(voice.dir) ?: return@withContext "voice.json missing"
            voiceSr = parseVoiceSr(voiceJson)
            val json = buildVoiceJson(voiceJson, s)
            val err = RvcJni.pipelineCreate().takeIf { it.isNotEmpty() }
                ?: RvcJni.pipelineInit(filesDir + "/models", libDir, json)
            if (err != null && err.isNotEmpty()) {
                _state.value = RvcState.Error("pipelineInit: $err")
                return@withContext err
            }
            initialized = true
            loadedSignature = signature
            loadedVoice = voice.name
            // Pitch is runtime-configurable and cheap, so it rides outside the
            // reload signature: apply the stored value after every init.
            RvcJni.pipelineSetPitch(s.pitchSemitones)
            RvcJni.pipelineSetRmsMixRate(s.rmsMixRate)
            RvcJni.pipelineSetProtect(s.protect)
            RvcJni.pipelineSetFilterRadius(s.filterRadius)
            RvcJni.pipelineSetF0Margin(s.f0Margin)
            _state.value = RvcState.Idle
            null
        }
    }

    /**
     * Starts a live conversion. Audio flows mic -> pipeline -> player as long
     * as [stop] is not called. [monitor] gates playback.
     */
    fun startLive(monitor: Boolean, muteMonitor: Boolean = false, forceSpeaker: Boolean = false) {
        if (runJob?.isActive == true) return
        loadedSignature = null // force reload in case the voice changed
        runJob = scope.launch {
            val initErr = ensureInitialized()
            if (initErr != null) {
                _state.value = RvcState.Error(initErr)
                return@launch
            }
            RvcJni.pipelineStreamStart()

            // 4 x 100 ms chunks = 400 ms of capture backlog at most.
            val input = Channel<CaptureBlock>(capacity = 4)
            if (!recorder.start(input)) {
                _state.value = RvcState.Error("AudioRecord init failed")
                return@launch
            }
            val monitorPlayer = if (monitor) {
                val preferred = if (forceSpeaker) {
                    (appContext.getSystemService(android.content.Context.AUDIO_SERVICE)
                        as android.media.AudioManager)
                        .getDevices(android.media.AudioManager.GET_DEVICES_OUTPUTS)
                        .firstOrNull { it.type == android.media.AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
                } else null
                AudioPlayer(voiceSr, muted = muteMonitor, preferredDevice = preferred).also {
                    if (!it.start()) _state.value = RvcState.Error("AudioTrack init failed")
                }
            } else null
            player = monitorPlayer

            val out = ArrayDeque<FloatArray>()
            var outSamples = 0

            // feeder: consumes capture blocks, emits converted ones
            launch(Dispatchers.Default) {
                var lastWarnAt = 0L
                var lastStatusAt = 0L
                var lastDropped = 0L
                var fedTotal = 0L
                var prevStats: RvcStats? = null
                while (true) {
                    val captured = input.receiveCatching().getOrNull() ?: break
                    val block = captured.samples
                    val dequeuedAt = android.os.SystemClock.elapsedRealtime()
                    val queuedMs = dequeuedAt - captured.atMs
                    val t0 = System.nanoTime()
                    val emitted = RvcJni.pipelineStreamFeed(block)
                    fedTotal += block.size
                    val stats = RvcStats.parse(RvcJni.pipelineStats())
                    val nBlocks = stats?.nBlocks ?: 0L
                    // Block i covers input samples [4000i, 4000i+4000); the
                    // stream is real-time, so its first sample entered the
                    // pipeline (fedTotal - 4000i)/16 ms ago. Add the channel
                    // backlog to get the age at the microphone.
                    val blockAgeMs =
                        if (nBlocks > 0) (fedTotal - 4000L * (nBlocks - 1)) / 16.0 else 0.0
                    if (emitted.isNotEmpty()) {
                        monitorPlayer?.write(emitted)
                        out.addLast(emitted)
                        outSamples += emitted.size
                        // A live session can run for hours; keeping every block
                        // for the preview button would grow 160 kB/s and turn
                        // into GC pauses (audible as dropouts). Keep the tail.
                        while (outSamples > LIVE_PREVIEW_SAMPLES && out.size > 1) {
                            outSamples -= out.removeFirst().size
                        }
                    }
                    // The native RTF only covers the compute inside one block.
                    // Wall time here also contains the capture wait and the
                    // speaker drain, which is exactly what makes the audible
                    // latency grow while the RTF number stays below 1.
                    val wallMs = (System.nanoTime() - t0) / 1e6
                    val drops = recorder.dropped
                    val now = android.os.SystemClock.elapsedRealtime()
                    if (now - lastWarnAt > 2_000 && (wallMs > 300.0 || queuedMs > 300L || drops != lastDropped)) {
                        // Interpolation, not String.format: a concatenated
                        // format string binds .format() to the last literal
                        // only and silently misaligns the specifiers.
                        Log.w(
                            TAG,
                            "feed wall=${wallMs.toInt()} ms, queued=$queuedMs ms (input backlog), " +
                                "dropped=$drops (${(drops * 0.25).toInt()} s skipped), " +
                                "rtf=${"%.2f".format(stats?.rtf ?: 0f)}",
                        )
                        lastWarnAt = now
                        lastDropped = drops
                    }
                    if (emitted.isNotEmpty() && blockTrace && stats != null) {
                        val prev = prevStats
                        fun d(cur: Float, get: (RvcStats) -> Float) =
                            if (prev == null) cur else cur - get(prev)
                        val c0 = d(stats.msC0) { it.msC0 }
                        val gn = d(stats.msGn) { it.msGn }
                        val r = d(stats.msR) { it.msR }
                        val enc = d(stats.msEnc) { it.msEnc }
                        val idx = d(stats.msIndex) { it.msIndex }
                        val f0 = d(stats.msF0) { it.msF0 }
                        val syn = d(stats.msSynth) { it.msSynth }
                        val sola = d(stats.msSola) { it.msSola }
                        val outMs = monitorPlayer?.presentationDelayMs() ?: 0
                        Log.i(
                            TAG,
                            "BLK n=${stats.nBlocks} cap=${queuedMs}ms " +
                                "age=${blockAgeMs.toInt()}ms | " +
                                "c0=${c0.toInt()} gn=${gn.toInt()} r=${r.toInt()} " +
                                "enc=${enc.toInt()} idx=${idx.toInt()} f0=${f0.toInt()} " +
                                "syn=${syn.toInt()} sola=${sola.toInt()} " +
                                "tot=${(c0 + gn + r + enc + idx + f0 + syn + sola).toInt()}ms | " +
                                "out=${outMs}ms track=${monitorPlayer?.bufferedMs() ?: 0}ms " +
                                "wall=${wallMs.toInt()}ms drops=$drops " +
                                "| ${monitorPlayer?.debugLatency() ?: ""}",
                        )
                    }
                    if (stats != null) prevStats = stats
                    if (emitted.isNotEmpty() && now - lastStatusAt >= 5_000) {
                        lastStatusAt = now
                        val outMs = monitorPlayer?.presentationDelayMs() ?: 0
                        val inMs = queuedMs + blockAgeMs
                        val e2e = inMs + (outMs - 250.0)
                        Log.i(
                            TAG,
                            "LAT e2e=${e2e.toInt()}ms | in=${inMs.toInt()}ms " +
                                "(queued=$queuedMs, model=${blockAgeMs.toInt()}) | " +
                                "out=${outMs}ms (track=${monitorPlayer?.bufferedMs() ?: 0}) | " +
                                "wall=${wallMs.toInt()}ms " +
                                "rtf=${"%.2f".format(stats?.rtf ?: 0f)} " +
                                "drops=$drops blocks=$nBlocks",
                        )
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
                    out.addLast(tail)
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

    /**
     * Converts a complete 16 kHz mono buffer offline. Feeds the streaming
     * pipeline in 4 000-sample strides, collects every emitted 40 kHz block
     * and flushes the tail. Reports progress through the state while it runs.
     * Returns null on success or the error string.
     */
    suspend fun convertOffline(input: FloatArray): String? =
        withContext(Dispatchers.Default) {
            val initErr = ensureInitialized()
            if (initErr != null) {
                _state.value = RvcState.Error(initErr)
                return@withContext initErr
            }
            _state.value = RvcState.Loading("Converting…")
            val streamErr = RvcJni.pipelineStreamStart()
            if (streamErr.isNotEmpty()) {
                _state.value = RvcState.Error("streamStart: $streamErr")
                return@withContext streamErr
            }

            val stride = 4_000
            val out = ArrayList<FloatArray>()
            var outSamples = 0
            var off = 0
            while (off < input.size) {
                val n = minOf(stride, input.size - off)
                val block = if (n == stride) input.copyOfRange(off, off + n)
                else input.copyOfRange(off, off + n)
                off += n
                val emitted = RvcJni.pipelineStreamFeed(block)
                if (emitted.isNotEmpty()) {
                    out.add(emitted)
                    outSamples += emitted.size
                }
                val s = RvcStats.parse(RvcJni.pipelineStats())
                _state.value = RvcState.Running(
                    rtf = s?.rtf ?: 1f,
                    msPerBlock = s?.msTotal?.div(s.nBlocks.coerceAtLeast(1)) ?: 0f,
                    msIndex = s?.msIndex ?: 0f,
                    blocks = s?.nBlocks ?: 0,
                    peak = if (out.isEmpty()) 0f else computePeak(out.last()),
                )
            }
            val tail = RvcJni.pipelineStreamEnd()
            if (tail.isNotEmpty()) { out.add(tail); outSamples += tail.size }

            val merged = FloatArray(outSamples)
            var p = 0
            for (b in out) { System.arraycopy(b, 0, merged, p, b.size); p += b.size }
            lastOutput = merged
            val final = RvcStats.parse(RvcJni.pipelineStats())
            _state.value = RvcState.Done(merged, final?.rtf ?: 1f)
            null
        }

    /** Plays back the last converted output. Returns null on success. */
    suspend fun playLast(): String? = withContext(Dispatchers.Default) {
        val out = lastOutput ?: return@withContext "no output yet"
        if (_isPlaying.value) stopPlay()
        val p = AudioPlayer(voiceSr)
        if (!p.start()) return@withContext "AudioTrack init failed"
        previewPlayer = p
        _isPlaying.value = true
        try {
            val chunk = 10_000
            var i = 0
            while (i < out.size) {
                if (!_isPlaying.value) break // stopPlay() was called
                val n = minOf(chunk, out.size - i)
                p.writeBlocking(out.copyOfRange(i, i + n))
                i += n
            }
            p.finish()
        } finally {
            _isPlaying.value = false
            previewPlayer = null
        }
        null
    }

    /** Stops the preview playback started by [playLast]. */
    fun stopPlay() {
        _isPlaying.value = false
        previewPlayer?.stop()
        previewPlayer = null
    }

    /**
     * Applies a pitch transpose in semitones at runtime — the next emitted
     * block picks it up, no graph reload (unlike an index_rate change).
     * Returns null on success.
     */
    fun setPitch(semitones: Float): String? {
        val err = RvcJni.pipelineSetPitch(semitones)
        if (err.isNotEmpty()) {
            reportError("setPitch: $err")
            return err
        }
        return null
    }

    /** The three official tunables — same runtime contract as [setPitch]. */
    fun setRmsMixRate(rate: Float): String? {
        val err = RvcJni.pipelineSetRmsMixRate(rate)
        if (err.isNotEmpty()) { reportError("setRmsMixRate: $err"); return err }
        return null
    }

    fun setProtect(protect: Float): String? {
        val err = RvcJni.pipelineSetProtect(protect)
        if (err.isNotEmpty()) { reportError("setProtect: $err"); return err }
        return null
    }

    fun setFilterRadius(radius: Int): String? {
        val err = RvcJni.pipelineSetFilterRadius(radius)
        if (err.isNotEmpty()) { reportError("setFilterRadius: $err"); return err }
        return null
    }

    /**
     * Realtime f0 lookahead: 0 = earliest 100 ms window (lowest latency),
     * 1 = one window later (steadier block-tail f0, +100 ms). Same runtime
     * contract as [setPitch]; only the streaming path reads it.
     */
    fun setF0Margin(margin: Int): String? {
        val err = RvcJni.pipelineSetF0Margin(margin)
        if (err.isNotEmpty()) { reportError("setF0Margin: $err"); return err }
        return null
    }

    /** Surfaces a message to the UI without throwing. */
    fun reportError(message: String) {
        _state.value = RvcState.Error(message)
    }

    /**
     * Writes the last converted output as a 40 kHz mono WAV into the shared
     * media collection (Music/DreamLite/), so a file manager can pick it up.
     * Returns null on success.
     */
    suspend fun saveLastWavToMedia(context: android.content.Context): String? =
        withContext(Dispatchers.Default) {
            val out = lastOutput ?: return@withContext "no output yet"
            val name = "dreamlite_${System.currentTimeMillis()}.wav"
            try {
                val resolver = context.contentResolver
                val collection = android.provider.MediaStore.Audio.Media
                    .getContentUri(android.provider.MediaStore.VOLUME_EXTERNAL_PRIMARY)
                val values = android.content.ContentValues().apply {
                    put(android.provider.MediaStore.Audio.Media.DISPLAY_NAME, name)
                    put(android.provider.MediaStore.Audio.Media.MIME_TYPE, "audio/wav")
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                        put(
                            android.provider.MediaStore.Audio.Media.RELATIVE_PATH,
                            "Music/DreamLite",
                        )
                        put(android.provider.MediaStore.Audio.Media.IS_PENDING, 1)
                    }
                }
                val uri = resolver.insert(collection, values)
                    ?: return@withContext "MediaStore insert failed"
                resolver.openOutputStream(uri)?.use { os ->
                    writeWav(os, out)
                } ?: return@withContext "cannot open MediaStore stream"
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                    val done = android.content.ContentValues().apply {
                        put(android.provider.MediaStore.Audio.Media.IS_PENDING, 0)
                    }
                    resolver.update(uri, done, null, null)
                }
                null
            } catch (e: Exception) {
                e.message ?: e.javaClass.simpleName
            }
        }

    private fun writeWav(os: java.io.OutputStream, out: FloatArray) {
        val d = java.io.DataOutputStream(os)
        d.writeBytes("RIFF")
        d.writeInt(Integer.reverseBytes(36 + out.size * 2))
        d.writeBytes("WAVE")
        d.writeBytes("fmt ")
        d.writeInt(Integer.reverseBytes(16))
        d.writeShort(Integer.reverseBytes(1) ushr 16)   // PCM
        d.writeShort(Integer.reverseBytes(1) ushr 16)   // mono
        d.writeInt(Integer.reverseBytes(40_000))
        d.writeInt(Integer.reverseBytes(40_000 * 2))
        d.writeShort(Integer.reverseBytes(2) ushr 16)   // bytes per frame
        d.writeShort(Integer.reverseBytes(16) ushr 16)  // bits per sample
        d.writeBytes("data")
        d.writeInt(Integer.reverseBytes(out.size * 2))
        // s16 LE, written little-endian directly
        for (v in out) {
            val q = (v.coerceIn(-1f, 1f) * 32767f).toInt()
            d.write(q and 0xff)
            d.write((q shr 8) and 0xff)
        }
    }

    /** Writes the last converted output as a 40 kHz mono WAV. Returns the path. */
    suspend fun saveLastWav(dest: java.io.File): String? = withContext(Dispatchers.Default) {
        val out = lastOutput ?: return@withContext "no output yet"
        try {
            java.io.FileOutputStream(dest).use { writeWav(it, out) }
            dest.absolutePath
        } catch (e: Exception) {
            e.message ?: e.javaClass.simpleName
        }
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

    /** Sample rate the voice's synth runs at; playback must match it or the
     * pitch shifts by the ratio of the two rates. */
    private fun parseVoiceSr(voiceJson: String): Int = try {
        org.json.JSONObject(voiceJson).optInt("sr", AudioPlayer.DEFAULT_SR)
    } catch (e: Exception) {
        AudioPlayer.DEFAULT_SR
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
            o.put("rms_mix_rate", s.rmsMixRate.toDouble())
            o.put("protect", s.protect.toDouble())
            o.put("filter_radius", s.filterRadius)
            o.toString()
        } catch (e: Exception) {
            Log.w(TAG, "voice json not an object; using defaults")
            "{\"index_rate\":${s.indexRate},\"nprobe\":${s.nprobe}," +
                "\"rms_mix_rate\":${s.rmsMixRate},\"protect\":${s.protect}," +
                "\"filter_radius\":${s.filterRadius}}"
        }
    }

    fun destroy() {
        stop()
        stopPlay()
        recorder.stop()
        RvcJni.pipelineDestroy()
        initialized = false
        loadedSignature = null
    }
}
