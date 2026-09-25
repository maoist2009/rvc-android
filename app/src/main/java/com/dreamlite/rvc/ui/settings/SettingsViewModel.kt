package com.dreamlite.rvc.ui.settings

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.dreamlite.rvc.DreamLiteApp
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.data.PackManager
import com.dreamlite.rvc.data.VoiceManager
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SettingsViewModel(
    app: Application,
) : AndroidViewModel(app) {

    private val app_ = app as DreamLiteApp

    val settings: StateFlow<SettingsRepository.Settings> =
        app_.settingsRepository.settingsFlow.stateIn(
            viewModelScope, SharingStarted.Eagerly, SettingsRepository.DEFAULTS
        )

    val voices: StateFlow<List<VoiceManager.Voice>> =
        app_.voiceManager.voicesFlow.stateIn(
            viewModelScope, SharingStarted.Eagerly, emptyList()
        )

    val packs: StateFlow<PackManager.ScanResult> =
        app_.packManager.state.stateIn(
            viewModelScope, SharingStarted.Eagerly, PackManager.ScanResult()
        )

    val filesDir: String = app_.filesDir.absolutePath

    fun setVoice(name: String) {
        viewModelScope.launch { app_.settingsRepository.setVoiceName(name) }
    }

    fun setIndexRate(rate: Float) {
        viewModelScope.launch { app_.settingsRepository.setIndexRate(rate) }
    }

    fun setNprobe(n: Int) {
        viewModelScope.launch { app_.settingsRepository.setNprobe(n) }
    }

    fun setMonitor(on: Boolean) {
        viewModelScope.launch { app_.settingsRepository.setMonitor(on) }
    }

    fun setNoiseGateDb(db: Float) {
        viewModelScope.launch { app_.settingsRepository.setNoiseGateDb(db) }
    }

    fun setJobLog(on: Boolean) {
        viewModelScope.launch { app_.settingsRepository.setJobLog(on) }
    }

    /** Re-scan the voices directory (after the user drops a folder in). */
    fun refreshVoices() {
        viewModelScope.launch { app_.voiceManager.rescan() }
    }

    /** Re-scan the external packs directory. */
    fun rescanPacks() {
        viewModelScope.launch { app_.packManager.scan() }
    }

    fun installShared() {
        viewModelScope.launch { app_.packManager.installShared() }
    }

    fun installVoice(pack: PackManager.VoicePack) {
        viewModelScope.launch { app_.packManager.installVoice(pack) }
    }

    /** Structural validation (files + manifest keys) — cheap, no NPU. */
    fun validateVoice(pack: PackManager.VoicePack) {
        // structural validation already ran during scan; re-scan to refresh
        viewModelScope.launch { app_.packManager.scan() }
    }

    /**
     * Loads encflow on the NPU and checks its IO against the manifest's
     * layout constants. Catches a T30 pack staged against a ctx-280 voice.
     */
    fun deepCheckVoice(pack: PackManager.VoicePack) {
        viewModelScope.launch { app_.packManager.validateDeep(pack) }
    }

    @Suppress("UNCHECKED_CAST")
    class Factory(private val app: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            SettingsViewModel(app) as T
    }
}
