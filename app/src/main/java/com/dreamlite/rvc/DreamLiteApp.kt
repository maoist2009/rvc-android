package com.dreamlite.rvc

import android.app.Application
import android.system.Os
import com.dreamlite.rvc.data.PackManager
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.data.VoiceManager
import com.dreamlite.rvc.engine.RvcSession
import com.dreamlite.rvc.service.RvcLiveService

/**
 * Application entry point.
 *
 * The QNN HTP backend resolves libadsprpc.so / libQnnHtpV79Skel.so through
 * ADSP_LIBRARY_PATH, which must point at the app's nativeLibraryDir before
 * the first QNN call — this is the same pattern that makes geniex-chat's
 * LiteRT backend work on this device. nativeLibraryDir is not knowable until
 * the Application is attached, so the env var is set here rather than in the
 * manifest.
 *
 * The session, settings and voice manager are process-wide singletons held
 * here, following the ML-Sharp-QNN structure.
 */
class DreamLiteApp : Application() {

    override fun onCreate() {
        super.onCreate()
        instance = this

        // Must precede any QNN call; nativeInit() happens lazily on first use.
        try {
            Os.setenv("ADSP_LIBRARY_PATH", appNativeLibDir, true)
        } catch (e: Exception) {
            android.util.Log.w(TAG, "setenv ADSP_LIBRARY_PATH failed", e)
        }

        // Every JNI entry point in RvcJni is a native method; the library has
        // to be loaded once per process or the first call throws
        // UnsatisfiedLinkError.
        RvcJni.load()

        // Make the voices directory visible to the user so they can drop a
        // voice in with a file manager.
        voicesDir.mkdirs()
        modelsDir.mkdirs()
    }

    private val appNativeLibDir: String by lazy {
        applicationInfo.nativeLibraryDir.also {
            android.util.Log.i(TAG, "nativeLibraryDir=$it")
        }
    }

    val voicesDir: java.io.File by lazy { java.io.File(filesDir, "voices") }
    val modelsDir: java.io.File by lazy { java.io.File(filesDir, "models") }

    val settingsRepository: SettingsRepository by lazy { SettingsRepository(this) }

    val voiceManager: VoiceManager by lazy { VoiceManager(this) }

    val packManager: PackManager by lazy { PackManager(this, voiceManager) }

    val session: RvcSession by lazy {
        RvcSession(
            appContext = this,
            voiceManager = voiceManager,
            settings = settingsRepository,
            libDir = appNativeLibDir,
            filesDir = filesDir.absolutePath,
        )
    }

    companion object {
        private const val TAG = "rvc_app"

        @Volatile
        private var instance: DreamLiteApp? = null

        fun get(): DreamLiteApp = instance!!

        const val NOTIF_CHANNEL_ID = "rvc_live"
    }
}
