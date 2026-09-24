package com.dreamlite.rvc.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "rvc_settings"
)

/**
 * Persisted user configuration. Everything here is applied to the native
 * pipeline at conversion start, so a change only takes effect on the next run.
 */
class SettingsRepository(private val context: Context) {

    data class Settings(
        val voiceName: String = "",
        val indexRate: Float = 0.0f,
        val nprobe: Int = 0,
        val monitor: Boolean = false,
        val noiseGateDb: Float = -40.0f,
        val jobLog: Boolean = false,
    )

    companion object {
        val DEFAULTS = Settings()
    }

    private object Keys {
        val VOICE_NAME = stringPreferencesKey("voice_name")
        val INDEX_RATE = floatPreferencesKey("index_rate")
        val NPROBE = intPreferencesKey("nprobe")
        val MONITOR = booleanPreferencesKey("monitor")
        val NOISE_GATE_DB = floatPreferencesKey("noise_gate_db")
        val JOB_LOG = booleanPreferencesKey("job_log")
    }

    val settingsFlow: Flow<Settings> = context.settingsDataStore.data.map { p ->
        Settings(
            voiceName = p[Keys.VOICE_NAME] ?: "",
            indexRate = p[Keys.INDEX_RATE] ?: DEFAULTS.indexRate,
            nprobe = p[Keys.NPROBE] ?: DEFAULTS.nprobe,
            monitor = p[Keys.MONITOR] ?: DEFAULTS.monitor,
            noiseGateDb = p[Keys.NOISE_GATE_DB] ?: DEFAULTS.noiseGateDb,
            jobLog = p[Keys.JOB_LOG] ?: DEFAULTS.jobLog,
        )
    }

    suspend fun setVoiceName(name: String) =
        edit { it[Keys.VOICE_NAME] = name }

    suspend fun setIndexRate(rate: Float) =
        edit { it[Keys.INDEX_RATE] = rate.coerceIn(0.0f, 1.0f) }

    suspend fun setNprobe(n: Int) =
        edit { it[Keys.NPROBE] = n.coerceAtLeast(0) }

    suspend fun setMonitor(on: Boolean) =
        edit { it[Keys.MONITOR] = on }

    suspend fun setNoiseGateDb(db: Float) =
        edit { it[Keys.NOISE_GATE_DB] = db.coerceIn(-80.0f, 0.0f) }

    suspend fun setJobLog(on: Boolean) =
        edit { it[Keys.JOB_LOG] = on }

    private suspend inline fun edit(crossinline block: (androidx.datastore.preferences.core.MutablePreferences) -> Unit) {
        context.settingsDataStore.edit { block(it) }
    }
}
