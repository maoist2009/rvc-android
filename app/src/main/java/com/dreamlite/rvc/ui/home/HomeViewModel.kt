package com.dreamlite.rvc.ui.home

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.dreamlite.rvc.DreamLiteApp
import com.dreamlite.rvc.data.AudioDecoder
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.data.VoiceManager
import com.dreamlite.rvc.engine.RvcState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class HomeViewModel(
    app: Application,
) : AndroidViewModel(app) {

    private val context = app.applicationContext
    private val appInstance = app as DreamLiteApp
    private val session = appInstance.session
    private val settingsRepo = appInstance.settingsRepository

    /** Guards the decode+convert chain: a second pick while the first is
     * still running would race two MediaCodec instances and abort the
     * process (observed: libc++abi terminating -> SIGABRT). */
    private val converting = java.util.concurrent.atomic.AtomicBoolean(false)

    val state: StateFlow<RvcState> = session.state
    val level: StateFlow<Float> = session.level
    val isPlaying: StateFlow<Boolean> = session.isPlaying
    val settings: StateFlow<SettingsRepository.Settings> =
        settingsRepo.settingsFlow.stateIn(
            viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly,
            SettingsRepository.DEFAULTS,
        )

    /** Voices visible to the inline picker, so choosing one never requires a
     * trip through Settings. */
    val voices: StateFlow<List<VoiceManager.Voice>> =
        appInstance.voiceManager.voicesFlow.stateIn(
            viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, emptyList(),
        )

    /** True while a converted buffer exists for the result card. */
    fun hasOutput(): Boolean = session.lastOutput != null

    /** Sample rate of the staged voice, so the result card reports the real
     * duration instead of assuming 40 kHz. */
    fun sampleRate(): Int = session.voiceSr

    fun setVoice(name: String) {
        viewModelScope.launch { settingsRepo.setVoiceName(name) }
    }

    fun setMonitor(on: Boolean) {
        viewModelScope.launch { settingsRepo.setMonitor(on) }
    }

    init {
        // preload the pipeline as soon as the screen exists so the first
        // record press does not pay the graph-load latency
        viewModelScope.launch { session.ensureInitialized() }
    }

    fun startLive() {
        viewModelScope.launch {
            val s = settingsRepo.settingsFlow.first()
            session.startLive(s.monitor)
        }
    }

    fun stop() = session.stop()

    /**
     * Persists the transpose and pushes it to the live pipeline: the next
     * emitted block uses it, without the full reload an index change needs.
     */
    fun setPitchSemitones(semitones: Float) {
        viewModelScope.launch {
            settingsRepo.setPitchSemitones(semitones)
            session.setPitch(semitones)
        }
    }

    /** The three official RVC tunables — same persist + push contract. */
    fun setRmsMixRate(rate: Float) {
        viewModelScope.launch {
            settingsRepo.setRmsMixRate(rate)
            session.setRmsMixRate(rate)
        }
    }

    fun setProtect(protect: Float) {
        viewModelScope.launch {
            settingsRepo.setProtect(protect)
            session.setProtect(protect)
        }
    }

    fun setFilterRadius(radius: Int) {
        viewModelScope.launch {
            settingsRepo.setFilterRadius(radius)
            session.setFilterRadius(radius)
        }
    }

    /**
     * Realtime f0 lookahead: 0 = earliest window (low latency), 1 = one extra
     * 100 ms window (steadier block-tail f0). Persist + push, like pitch.
     */
    fun setF0Margin(margin: Int) {
        viewModelScope.launch {
            settingsRepo.setF0Margin(margin)
            session.setF0Margin(margin)
        }
    }

    fun playConverted() {
        viewModelScope.launch { session.playLast() }
    }

    fun stopPlaying() = session.stopPlay()

    fun setIndexRate(rate: Float) {
        viewModelScope.launch { settingsRepo.setIndexRate(rate) }
    }

    /**
     * Converts an imported/recorded audio file offline: decode to 16 kHz mono,
     * run the streaming pipeline over it, then report stats. The result lands
     * in [session.lastOutput] for playback and saving.
     */
    fun convertOffline(uri: android.net.Uri) {
        if (!converting.compareAndSet(false, true)) {
            session.reportError("a conversion is already running")
            return
        }
        viewModelScope.launch {
            try {
                val input = withContext(Dispatchers.IO) {
                    val tmp = File(context.cacheDir, "decode_in")
                    context.contentResolver.openInputStream(uri)?.use { i ->
                        tmp.outputStream().use { o -> i.copyTo(o, 256 * 1024) }
                    }
                    tmp
                } ?: run {
                    session.reportError("cannot open the picked file")
                    return@launch
                }
                val samples = withContext(Dispatchers.Default) {
                    AudioDecoder.decodeTo16kMono(input) ?: AudioDecoder.readF32(input)
                }
                if (samples == null || samples.isEmpty()) {
                    session.reportError("could not decode ${input.name} (unsupported format?)")
                    return@launch
                }
                session.convertOffline(samples)
            } finally {
                converting.set(false)
            }
        }
    }

    /** Saves the last converted output into the shared media collection. */
    fun saveWav() {
        viewModelScope.launch {
            val err = session.saveLastWavToMedia(context)
            if (err != null) session.reportError(err)
        }
    }

    @Suppress("UNCHECKED_CAST")
    class Factory(private val app: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            HomeViewModel(app) as T
    }
}
