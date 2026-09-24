package com.dreamlite.rvc.ui.home

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.dreamlite.rvc.DreamLiteApp
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.engine.RvcState
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class HomeViewModel(
    app: Application,
) : AndroidViewModel(app) {

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

    @Suppress("UNCHECKED_CAST")
    class Factory(private val app: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            HomeViewModel(app) as T
    }
}
