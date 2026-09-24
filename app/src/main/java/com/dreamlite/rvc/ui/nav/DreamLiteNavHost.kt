package com.dreamlite.rvc.ui.nav

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dreamlite.rvc.R
import com.dreamlite.rvc.ui.home.HomeScreen
import com.dreamlite.rvc.ui.home.HomeViewModel
import com.dreamlite.rvc.ui.settings.SettingsScreen
import com.dreamlite.rvc.ui.settings.SettingsViewModel

/**
 * Two-screen root: Home (record / playback / live RTF) and Settings.
 *
 * navigation-compose is deliberately not used: its AARs are absent from this
 * machine's Gradle cache, so an offline build cannot resolve them. A plain
 * state switch gives the same UX for two destinations without a dependency.
 */
enum class Screen { Home, Settings }

@Composable
fun DreamLiteNavHost(
    onRequestMic: () -> Unit,
) {
    var screen by remember { mutableStateOf(Screen.Home) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = screen == Screen.Home,
                    onClick = { screen = Screen.Home },
                    icon = { Icon(Icons.Filled.Home, contentDescription = null) },
                    label = { Text(stringResource(R.string.nav_home)) },
                )
                NavigationBarItem(
                    selected = screen == Screen.Settings,
                    onClick = { screen = Screen.Settings },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text(stringResource(R.string.nav_settings)) },
                )
            }
        },
    ) { inner ->
        when (screen) {
            Screen.Home -> {
                val vm: HomeViewModel = viewModel()
                HomeScreen(
                    viewModel = vm,
                    onRequestMic = onRequestMic,
                    modifier = Modifier.fillMaxSize().padding(inner),
                )
            }
            Screen.Settings -> {
                val vm: SettingsViewModel = viewModel()
                SettingsScreen(
                    viewModel = vm,
                    modifier = Modifier.fillMaxSize().padding(inner),
                )
            }
        }
    }
}
