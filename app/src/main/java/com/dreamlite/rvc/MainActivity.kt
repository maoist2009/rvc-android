package com.dreamlite.rvc

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dreamlite.rvc.engine.RvcState
import com.dreamlite.rvc.service.RvcLiveService
import com.dreamlite.rvc.ui.nav.DreamLiteNavHost
import com.dreamlite.rvc.ui.home.HomeViewModel
import com.dreamlite.rvc.ui.theme.DreamLiteTheme
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // The foreground service follows the session state: it must be up
        // while a conversion is running, or the OS starves the QNN threads.
        val app = application as DreamLiteApp
        lifecycleScope.launch {
            app.session.state.collectLatest { s ->
                when (s) {
                    is RvcState.Running, is RvcState.Loading ->
                        RvcLiveService.start(this@MainActivity)
                    else -> RvcLiveService.stop(this@MainActivity)
                }
            }
        }

        setContent {
            DreamLiteTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    val micLauncher = rememberLauncherForActivityResult(
                        ActivityResultContracts.RequestPermission()
                    ) { granted ->
                        if (!granted) { /* the UI shows its own denied state */ }
                    }

                    // SAF picker for offline conversion: any audio the system
                    // can decode is accepted; AudioDecoder turns it into the
                    // 16 kHz mono f32 the pipeline consumes.
                    val homeVm: HomeViewModel = viewModel()
                    val audioPicker = rememberLauncherForActivityResult(
                        ActivityResultContracts.OpenDocument()
                    ) { uri ->
                        uri?.let { homeVm.convertOffline(it) }
                    }

                    val notifLauncher = rememberLauncherForActivityResult(
                        ActivityResultContracts.RequestPermission()
                    ) { /* best-effort; foreground priority does not need it */ }

                    LaunchedEffect(Unit) {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            notifLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        }
                    }

                    DreamLiteNavHost(
                        onRequestMic = { micLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                        onPickAudio = { audioPicker.launch(arrayOf("audio/*")) },
                    )
                }
            }
        }
    }
}
