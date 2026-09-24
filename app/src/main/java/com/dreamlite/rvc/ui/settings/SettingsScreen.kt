package com.dreamlite.rvc.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.dreamlite.rvc.R
import com.dreamlite.rvc.ui.theme.Spacing

@Composable
fun SettingsScreen(viewModel: SettingsViewModel, modifier: Modifier = Modifier) {
    val settings by viewModel.settings.collectAsState()
    val voices by viewModel.voices.collectAsState()
    val scroll = rememberScrollState()

    Column(
        modifier = modifier
            .verticalScroll(scroll)
            .padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        VoiceCard(
            voices = voices,
            selected = settings.voiceName,
            onSelect = viewModel::setVoice,
        )

        RetrievalCard(
            indexRate = settings.indexRate,
            nprobe = settings.nprobe,
            onIndexRate = viewModel::setIndexRate,
            onNprobe = viewModel::setNprobe,
        )

        AudioCard(
            monitor = settings.monitor,
            noiseGateDb = settings.noiseGateDb,
            onMonitor = viewModel::setMonitor,
            onNoiseGate = viewModel::setNoiseGateDb,
        )

        DiagnosticsCard(
            jobLog = settings.jobLog,
            onJobLog = viewModel::setJobLog,
            filesDir = viewModel.filesDir,
        )
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        ),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(Spacing.lg), verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.SemiBold,
            )
            content()
        }
    }
}

@Composable
private fun VoiceCard(
    voices: List<com.dreamlite.rvc.data.VoiceManager.Voice>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    SectionCard(title = stringResource(R.string.settings_voice)) {
        if (voices.isEmpty()) {
            Text(
                text = stringResource(R.string.settings_voice_none),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = stringResource(R.string.settings_voice_hint),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            voices.forEach { v ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
                ) {
                    RadioButton(
                        selected = v.name == selected,
                        onClick = { onSelect(v.name) },
                    )
                    Column {
                        Text(
                            text = v.name,
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        if (!v.hasIndex) {
                            Text(
                                text = "no .index — retrieval off",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun RetrievalCard(
    indexRate: Float,
    nprobe: Int,
    onIndexRate: (Float) -> Unit,
    onNprobe: (Int) -> Unit,
) {
    SectionCard(title = stringResource(R.string.settings_retrieval)) {
        Text(
            text = stringResource(R.string.settings_index_rate),
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Text(
            text = if (indexRate <= 0f) stringResource(R.string.settings_index_rate_off)
            else stringResource(R.string.settings_index_rate_summary, indexRate),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = indexRate,
            onValueChange = onIndexRate,
            valueRange = 0.0f..1.0f,
            steps = 19,
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            text = stringResource(R.string.settings_nprobe),
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Text(
            text = if (nprobe <= 0) stringResource(R.string.settings_nprobe_auto) else nprobe.toString(),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = nprobe.toFloat(),
            onValueChange = { onNprobe(it.toInt()) },
            valueRange = 0.0f..16.0f,
            steps = 15,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun AudioCard(
    monitor: Boolean,
    noiseGateDb: Float,
    onMonitor: (Boolean) -> Unit,
    onNoiseGate: (Float) -> Unit,
) {
    SectionCard(title = stringResource(R.string.settings_audio)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                Text(
                    text = stringResource(R.string.settings_monitor),
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = stringResource(R.string.settings_monitor_subtitle),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked = monitor, onCheckedChange = onMonitor)
        }
        Text(
            text = stringResource(R.string.settings_noise_gate),
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Text(
            text = "%.0f dB".format(noiseGateDb),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = noiseGateDb,
            onValueChange = onNoiseGate,
            valueRange = -80.0f..0.0f,
            steps = 79,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun DiagnosticsCard(
    jobLog: Boolean,
    onJobLog: (Boolean) -> Unit,
    filesDir: String,
) {
    SectionCard(title = stringResource(R.string.settings_diag)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                Text(
                    text = stringResource(R.string.settings_job_log),
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = stringResource(R.string.settings_job_log_subtitle),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked = jobLog, onCheckedChange = onJobLog)
        }
        PathRow(stringResource(R.string.settings_files_dir), filesDir)
        PathRow(stringResource(R.string.settings_voices_dir), "$filesDir/voices")
        PathRow(stringResource(R.string.settings_models_dir), "$filesDir/models")
    }
}

@Composable
private fun PathRow(label: String, path: String) {
    Column {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = path,
            style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}
