package com.dreamlite.rvc.ui.home

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.dreamlite.rvc.DreamLiteApp
import com.dreamlite.rvc.data.AudioDecoder
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.engine.RvcState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class HomeViewModel(
    app: Application,
) : AndroidViewModel(app) {

    private val context = app.applicationContext
    private val session = (app as DreamLiteApp).session
    private val settings = (app as DreamLiteApp).settingsRepository

    val state: StateFlow<RvcState> = session.state
    val level: StateFlow<Float> = session.level

    init {
        // preload the pipeline as soon as the screen exists so the first
        // record press does not pay the graph-load latency
        viewModelScope.launch { session.ensureInitialized() }
    }

    fun startLive() {
        viewModelScope.launch {
            val s = settings.settingsFlow.let { it }
            val monitor = true
            session.startLive(monitor)
        }
    }

    fun stop() = session.stop()

    fun playConverted() {
        viewModelScope.launch { session.playLast() }
    }

    /**
     * Converts an imported/recorded audio file offline: decode to 16 kHz mono,
     * run the streaming pipeline over it, then report stats. The result lands
     * in [session.lastOutput] for playback and saving.
     */
    fun convertOffline(uri: android.net.Uri) {
        viewModelScope.launch {
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
