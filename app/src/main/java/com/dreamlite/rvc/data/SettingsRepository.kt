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
        // Matches the reference pipeline's load_voice default. Official RVC
        // scripts use ~0.75; the checkpoint itself does not record the value.
        val indexRate: Float = 0.75f,
        val nprobe: Int = 0,
        val monitor: Boolean = false,
        val noiseGateDb: Float = -40.0f,
        val jobLog: Boolean = false,
        /** Pitch transpose in semitones; official RVC's f0_up_key, 0 = none. */
        val pitchSemitones: Float = 0.0f,
        /** Official rms_mix_rate: 0 matches the source loudness, 1 keeps the
         * converted output's own level. */
        val rmsMixRate: Float = 0.25f,
        /** Official protect: blends the pre-retrieval features back in on
         * voiced frames; 0.5 is the official's "off". */
        val protect: Float = 0.5f,
        /** Official filter_radius: median-filter window over the f0 track,
         * odd kernel sizes; 1 is a no-op. */
        val filterRadius: Int = 3,
        /** Realtime f0 lookahead in 100 ms windows: 0 = earliest window (low
         * latency), 1 = one window later (steadier block-tail f0, +100 ms).
         * Only the live/streaming path reads this. */
        val f0Margin: Int = 0,
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
        val PITCH_SEMITONES = floatPreferencesKey("pitch_semitones")
        val RMS_MIX_RATE = floatPreferencesKey("rms_mix_rate")
        val PROTECT = floatPreferencesKey("protect")
        val FILTER_RADIUS = intPreferencesKey("filter_radius")
        val F0_MARGIN = intPreferencesKey("f0_margin")
    }

    val settingsFlow: Flow<Settings> = context.settingsDataStore.data.map { p ->
        Settings(
            voiceName = p[Keys.VOICE_NAME] ?: "",
            indexRate = p[Keys.INDEX_RATE] ?: DEFAULTS.indexRate,
            nprobe = p[Keys.NPROBE] ?: DEFAULTS.nprobe,
            monitor = p[Keys.MONITOR] ?: DEFAULTS.monitor,
            noiseGateDb = p[Keys.NOISE_GATE_DB] ?: DEFAULTS.noiseGateDb,
            jobLog = p[Keys.JOB_LOG] ?: DEFAULTS.jobLog,
            pitchSemitones = p[Keys.PITCH_SEMITONES] ?: DEFAULTS.pitchSemitones,
            rmsMixRate = p[Keys.RMS_MIX_RATE] ?: DEFAULTS.rmsMixRate,
            protect = p[Keys.PROTECT] ?: DEFAULTS.protect,
            filterRadius = p[Keys.FILTER_RADIUS] ?: DEFAULTS.filterRadius,
            f0Margin = p[Keys.F0_MARGIN] ?: DEFAULTS.f0Margin,
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

    suspend fun setPitchSemitones(semitones: Float) =
        edit { it[Keys.PITCH_SEMITONES] = semitones.coerceIn(-12.0f, 12.0f) }

    suspend fun setRmsMixRate(rate: Float) =
        edit { it[Keys.RMS_MIX_RATE] = rate.coerceIn(0.0f, 1.0f) }

    suspend fun setProtect(protect: Float) =
        edit { it[Keys.PROTECT] = protect.coerceIn(0.0f, 0.5f) }

    suspend fun setFilterRadius(radius: Int) =
        edit { it[Keys.FILTER_RADIUS] = radius.coerceIn(1, 7) }

    suspend fun setF0Margin(margin: Int) =
        edit { it[Keys.F0_MARGIN] = margin.coerceIn(0, 1) }

    private suspend inline fun edit(crossinline block: (androidx.datastore.preferences.core.MutablePreferences) -> Unit) {
        context.settingsDataStore.edit { block(it) }
    }
}
