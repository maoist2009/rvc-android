package com.dreamlite.rvc

import android.app.Application
import android.system.Os
import android.util.Log

class RvcApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Must precede the first call into libQnnHtp.so: the HTP stub resolves
        // libQnnHtpV79Skel.so out of ADSP_LIBRARY_PATH, and the app's own lib
        // dir is the only place the dynamic linker namespace of this process
        // guarantees to search. The skel ships in jniLibs, so the same dir
        // serves as libDir and skelDir.
        val libDir = applicationInfo.nativeLibraryDir
        Os.setenv("ADSP_LIBRARY_PATH", libDir, true)
        Os.setenv("LD_LIBRARY_PATH", libDir, true)
        Log.i(TAG, "nativeLibraryDir=$libDir")

        RvcJni.load()
    }

    companion object {
        private const val TAG = "rvc_app"
    }
}
