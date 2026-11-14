package com.dreamlite.rvc.ui.nav

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dreamlite.rvc.R
import com.dreamlite.rvc.ui.home.HomeScreen
import com.dreamlite.rvc.ui.home.HomeViewModel
import com.dreamlite.rvc.ui.settings.SettingsScreen
import com.dreamlite.rvc.ui.settings.SettingsViewModel

/**
 * One workflow page plus an escape hatch to Settings.
 *
 * The old bottom navigation advertised two equal destinations, but Settings
 * is maintenance (voices, packs, diagnostics) while Home is the entire task.
 * A top-bar gear keeps it reachable without making it look like a peer.
 *
 * navigation-compose is deliberately not used: its AARs are absent from this
 * machine's Gradle cache, so an offline build cannot resolve them. A plain
 * state switch gives the same UX for two destinations without a dependency.
 */
private enum class Screen { Home, Settings }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DreamLiteNavHost(
    onRequestMic: () -> Unit,
    onPickAudio: () -> Unit,
) {
    var screen by remember { mutableStateOf(Screen.Home) }
    val scrollBehavior = TopAppBarDefaults.enterAlwaysScrollBehavior()
    val homeVm: HomeViewModel = viewModel()

    Scaffold(
        modifier = Modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = if (screen == Screen.Home) stringResource(R.string.top_title)
                        else stringResource(R.string.nav_settings),
                        fontWeight = FontWeight.SemiBold,
                    )
                },
                navigationIcon = {
                    if (screen == Screen.Settings) {
                        IconButton(onClick = { screen = Screen.Home }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = stringResource(R.string.home_back),
                            )
                        }
                    }
                },
                actions = {
                    if (screen == Screen.Home) {
                        IconButton(onClick = { screen = Screen.Settings }) {
                            Icon(
                                imageVector = Icons.Filled.Settings,
                                contentDescription = stringResource(R.string.home_open_settings),
                            )
                        }
                    }
                },
                scrollBehavior = scrollBehavior,
            )
        },
    ) { inner ->
        when (screen) {
            Screen.Home -> HomeScreen(
                viewModel = homeVm,
                onRequestMic = onRequestMic,
                onPickAudio = onPickAudio,
                onOpenSettings = { screen = Screen.Settings },
                modifier = Modifier.fillMaxSize().padding(inner),
            )
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
