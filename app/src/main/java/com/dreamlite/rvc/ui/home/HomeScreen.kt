package com.dreamlite.rvc.ui.home

import android.Manifest
import android.content.pm.PackageManager
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.dreamlite.rvc.R
import com.dreamlite.rvc.data.SettingsRepository
import com.dreamlite.rvc.data.VoiceManager
import com.dreamlite.rvc.engine.RvcState
import com.dreamlite.rvc.ui.theme.Spacing

/**
 * The whole app in one page: pick a voice, speak (or import a clip), and the
 * converted result lands in the same place. Parameters live in a bottom sheet
 * so the primary path stays a single big target plus one secondary action.
 *
 * Everything the old layout split across separate cards — record, convert,
 * stats — now shares one state machine, which is what the user actually has:
 * one pipeline, one output buffer.
 */
@Composable
fun HomeScreen(
    viewModel: HomeViewModel,
    onRequestMic: () -> Unit,
    onPickAudio: () -> Unit,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val state by viewModel.state.collectAsState()
    val level by viewModel.level.collectAsState()
    val isPlaying by viewModel.isPlaying.collectAsState()
    val settings by viewModel.settings.collectAsState()
    val voices by viewModel.voices.collectAsState()
    val context = LocalContext.current

    var showParams by remember { mutableStateOf(false) }
    var showVoices by remember { mutableStateOf(false) }

    // Re-check on every ON_RESUME: the permission dialog resolves after the
    // activity is paused, so a remember{} captured at first composition would
    // keep showing the "grant access" affordance after the user already did.
    var micGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val lifecycleOwner = LocalLifecycleOwner.current
    androidx.compose.runtime.DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, e ->
            if (e == Lifecycle.Event.ON_RESUME) {
                micGranted = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.RECORD_AUDIO
                ) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val running = state is RvcState.Running
    val busy = state is RvcState.Loading || running
    val voiceSelected = settings.voiceName.isNotEmpty() || voices.isNotEmpty()

    Column(
        modifier = modifier
            .verticalScroll(rememberScrollState())
            .padding(horizontal = Spacing.lg, vertical = Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        VoiceBar(
            voice = settings.voiceName.ifEmpty {
                voices.firstOrNull()?.name ?: stringResource(R.string.home_voice_none)
            },
            hasVoice = voiceSelected,
            onChange = { showVoices = true },
            onOpenSettings = onOpenSettings,
        )

        LiveHero(
            state = state,
            level = level,
            running = running,
            busy = busy,
            micGranted = micGranted,
            hasVoice = voiceSelected,
            pitch = settings.pitchSemitones,
            monitor = settings.monitor,
            onStart = { viewModel.startLive() },
            onStop = { viewModel.stop() },
            onRequestMic = onRequestMic,
            onPickAudio = onPickAudio,
            onPitchChange = viewModel::setPitchSemitones,
            onMonitorChange = viewModel::setMonitor,
            onOpenParams = { showParams = true },
            onOpenSettings = onOpenSettings,
        )

        ResultCard(
            state = state,
            isPlaying = isPlaying,
            hasOutput = viewModel.hasOutput(),
            sampleRate = viewModel.sampleRate(),
            onPlay = { viewModel.playConverted() },
            onStopPlay = { viewModel.stopPlaying() },
            onSave = { viewModel.saveWav() },
        )

        RuntimeCard(state = state)
    }

    if (showParams) {
        ParamsSheet(
            settings = settings,
            onIndexRateChange = viewModel::setIndexRate,
            onPitchChange = viewModel::setPitchSemitones,
            onRmsMixRateChange = viewModel::setRmsMixRate,
            onProtectChange = viewModel::setProtect,
            onFilterRadiusChange = viewModel::setFilterRadius,
            onF0MarginChange = viewModel::setF0Margin,
            onReset = {
                viewModel.setIndexRate(SettingsRepository.DEFAULTS.indexRate)
                viewModel.setPitchSemitones(SettingsRepository.DEFAULTS.pitchSemitones)
                viewModel.setRmsMixRate(SettingsRepository.DEFAULTS.rmsMixRate)
                viewModel.setProtect(SettingsRepository.DEFAULTS.protect)
                viewModel.setFilterRadius(SettingsRepository.DEFAULTS.filterRadius)
                viewModel.setF0Margin(SettingsRepository.DEFAULTS.f0Margin)
            },
            onDismiss = { showParams = false },
        )
    }

    if (showVoices) {
        VoiceSheet(
            voices = voices,
            selected = settings.voiceName,
            onSelect = {
                viewModel.setVoice(it)
                showVoices = false
            },
            onManage = {
                showVoices = false
                onOpenSettings()
            },
            onDismiss = { showVoices = false },
        )
    }
}

/* ---------------------------------------------------------------- voice bar */

@Composable
private fun VoiceBar(
    voice: String,
    hasVoice: Boolean,
    onChange: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Surface(
            shape = RoundedCornerShape(50),
            color = if (hasVoice) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surfaceContainerHigh,
            modifier = Modifier
                .weight(1f)
                .clip(RoundedCornerShape(50))
                .clickable(onClick = if (hasVoice) onChange else onOpenSettings),
        ) {
            Row(
                modifier = Modifier.padding(horizontal = Spacing.md, vertical = Spacing.sm),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            ) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(
                            if (hasVoice) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.outline
                        )
                )
                Text(
                    text = voice,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Medium,
                    color = if (hasVoice) MaterialTheme.colorScheme.onPrimaryContainer
                    else MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f, fill = false),
                )
                if (hasVoice) {
                    Text(
                        text = stringResource(R.string.home_voice_change),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                    )
                }
            }
        }
        IconButton(onClick = onOpenSettings) {
            Icon(
                imageVector = Icons.Filled.Tune,
                contentDescription = stringResource(R.string.home_open_settings),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/* --------------------------------------------------------------- live hero */

@Composable
private fun LiveHero(
    state: RvcState,
    level: Float,
    running: Boolean,
    busy: Boolean,
    micGranted: Boolean,
    hasVoice: Boolean,
    pitch: Float,
    monitor: Boolean,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onRequestMic: () -> Unit,
    onPickAudio: () -> Unit,
    onPitchChange: (Float) -> Unit,
    onMonitorChange: (Boolean) -> Unit,
    onOpenParams: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        ),
        shape = RoundedCornerShape(28.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            StatusLine(state = state, running = running)

            // The one big target. Its ring breathes with the input level while
            // running, so the button doubles as the level meter.
            val breathe by animateFloatAsState(
                targetValue = if (running) 1f else 0.94f,
                animationSpec = spring(dampingRatio = 0.45f, stiffness = 220f),
                label = "breathe",
            )
            val ring = if (running) 1f + level.coerceIn(0f, 1f) * 0.18f else 1f
            Box(contentAlignment = Alignment.Center) {
                Box(
                    modifier = Modifier
                        .size(176.dp * ring)
                        .clip(CircleShape)
                        .background(
                            MaterialTheme.colorScheme.primary.copy(
                                alpha = if (running) 0.12f else 0.06f
                            )
                        )
                )
                Box(
                    modifier = Modifier
                        .size(148.dp * (if (running) breathe else 1f))
                        .clip(CircleShape)
                        .background(
                            when {
                                running -> MaterialTheme.colorScheme.errorContainer
                                hasVoice -> MaterialTheme.colorScheme.primary
                                else -> MaterialTheme.colorScheme.surfaceContainerHighest
                            }
                        )
                        .clickable {
                            when {
                                running -> onStop()
                                !micGranted -> onRequestMic()
                                !hasVoice -> onOpenSettings()
                                else -> onStart()
                            }
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = if (running) Icons.Filled.Stop else Icons.Filled.Mic,
                        contentDescription = null,
                        tint = when {
                            running -> MaterialTheme.colorScheme.onErrorContainer
                            hasVoice -> MaterialTheme.colorScheme.onPrimary
                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                        },
                        modifier = Modifier.size(52.dp),
                    )
                }
            }

            Text(
                text = when {
                    running -> stringResource(R.string.home_tap_stop)
                    !hasVoice -> stringResource(R.string.home_voice_none_hint)
                    !micGranted -> stringResource(R.string.home_mic_needed_hint)
                    else -> stringResource(R.string.home_live_hint)
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )

            if (running) {
                LinearProgressIndicator(
                    progress = { level.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                )
            }

            PitchStepper(
                value = pitch,
                enabled = !busy || running,
                onChange = onPitchChange,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = stringResource(R.string.home_monitor),
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        text = stringResource(R.string.home_monitor_hint),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = monitor,
                    onCheckedChange = onMonitorChange,
                    enabled = !busy,
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            ) {
                OutlinedButton(
                    onClick = onPickAudio,
                    enabled = !busy && hasVoice,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Filled.AudioFile,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.size(Spacing.sm))
                    Text(stringResource(R.string.home_import_audio))
                }
                FilledTonalButton(
                    onClick = onOpenParams,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Filled.Tune,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.size(Spacing.sm))
                    Text(stringResource(R.string.home_parameters))
                }
            }

            if (state is RvcState.Error) {
                Text(
                    text = state.message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
private fun StatusLine(state: RvcState, running: Boolean) {
    val text = when (state) {
        is RvcState.Loading -> state.label
        is RvcState.Running -> stringResource(R.string.home_status_running)
        is RvcState.Done -> stringResource(R.string.home_recorded)
        is RvcState.Error -> stringResource(R.string.home_status_idle)
        else -> stringResource(R.string.home_ready)
    }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            Box(
                modifier = Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .background(
                        when {
                            state is RvcState.Error -> MaterialTheme.colorScheme.error
                            running -> MaterialTheme.colorScheme.primary
                            state is RvcState.Loading -> MaterialTheme.colorScheme.tertiary
                            else -> MaterialTheme.colorScheme.outline
                        }
                    )
            )
            Text(
                text = text,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
        (state as? RvcState.Running)?.let {
            Surface(
                shape = RoundedCornerShape(50),
                color = MaterialTheme.colorScheme.primaryContainer,
            ) {
                Text(
                    text = "RTF %.2f".format(it.rtf),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    modifier = Modifier.padding(horizontal = Spacing.sm, vertical = 2.dp),
                )
            }
        }
    }
}

@Composable
private fun PitchStepper(
    value: Float,
    enabled: Boolean,
    onChange: (Float) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Text(
            text = stringResource(R.string.home_pitch),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        IconButton(
            onClick = { onChange((value - 1f).coerceIn(-12f, 12f)) },
            enabled = enabled && value > -12f,
        ) {
            Icon(
                imageVector = Icons.Filled.Remove,
                contentDescription = stringResource(R.string.home_pitch_decrease),
                modifier = Modifier.size(20.dp),
            )
        }
        Text(
            text = stringResource(R.string.home_pitch_value, value.toInt()),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.padding(horizontal = Spacing.sm),
        )
        IconButton(
            onClick = { onChange((value + 1f).coerceIn(-12f, 12f)) },
            enabled = enabled && value < 12f,
        ) {
            Icon(
                imageVector = Icons.Filled.Add,
                contentDescription = stringResource(R.string.home_pitch_increase),
                modifier = Modifier.size(20.dp),
            )
        }
    }
}

/* ------------------------------------------------------------------ result */

@Composable
private fun ResultCard(
    state: RvcState,
    isPlaying: Boolean,
    hasOutput: Boolean,
    sampleRate: Int,
    onPlay: () -> Unit,
    onStopPlay: () -> Unit,
    onSave: () -> Unit,
) {
    val done = state as? RvcState.Done
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer
        ),
        shape = RoundedCornerShape(24.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = stringResource(R.string.home_result),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                done?.let {
                    Text(
                        text = stringResource(
                            R.string.home_result_meta,
                            it.samples.size / sampleRate.toFloat(),
                            sampleRate,
                        ),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            if (!hasOutput) {
                Text(
                    text = stringResource(R.string.home_result_empty),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
                ) {
                    Button(
                        onClick = if (isPlaying) onStopPlay else onPlay,
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(
                            imageVector = if (isPlaying) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(Modifier.size(Spacing.sm))
                        Text(
                            text = if (isPlaying) stringResource(R.string.home_stop_playback)
                            else stringResource(R.string.home_play_converted),
                        )
                    }
                    FilledTonalButton(onClick = onSave, modifier = Modifier.weight(1f)) {
                        Icon(
                            imageVector = Icons.Filled.Save,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.convert_save))
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ runtime */

@Composable
private fun RuntimeCard(state: RvcState) {
    var expanded by remember { mutableStateOf(false) }
    val stats = (state as? RvcState.Running)?.let {
        Stats.Live(it.rtf, it.msPerBlock, it.msIndex, it.blocks, it.peak)
    } ?: (state as? RvcState.Done)?.let { Stats.Final(it.rtf) }

    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer
        ),
        shape = RoundedCornerShape(24.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .clickable { expanded = !expanded },
    ) {
        Column(
            modifier = Modifier.padding(horizontal = Spacing.lg, vertical = Spacing.md),
            verticalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = stringResource(R.string.home_details),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Icon(
                    imageVector = if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (stats == null) {
                Text(
                    text = stringResource(R.string.home_no_output),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Column {
                        Text(
                            text = stringResource(R.string.stats_rtf),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            text = "%.2f".format(stats.rtf),
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold,
                            color = if (stats.rtf < 0.8f) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.error,
                        )
                    }
                    Text(
                        text = stringResource(R.string.stats_rtf_target),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                AnimatedVisibility(
                    visible = expanded,
                    enter = fadeIn(tween(120)) + expandVertically(tween(160)),
                    exit = fadeOut(tween(90)) + shrinkVertically(tween(140)),
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
                        LinearProgressIndicator(
                            progress = { (stats.rtf / 1.2f).coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth().height(6.dp),
                        )
                        if (stats is Stats.Live) {
                            StatRow(
                                label = stringResource(R.string.stats_block),
                                value = "%.0f ms".format(stats.msPerBlock),
                            )
                            if (stats.msIndex > 0) {
                                StatRow(
                                    label = stringResource(R.string.stats_stage_index),
                                    value = "%.1f ms".format(stats.msIndex),
                                )
                            }
                            StatRow(
                                label = stringResource(R.string.stats_peak),
                                value = "%.3f".format(stats.peak),
                            )
                            StatRow(
                                label = stringResource(R.string.stats_blocks),
                                value = stats.blocks.toString(),
                            )
                        }
                    }
                }
            }
        }
    }
}

private sealed class Stats {
    abstract val rtf: Float

    data class Live(
        override val rtf: Float,
        val msPerBlock: Float,
        val msIndex: Float,
        val blocks: Long,
        val peak: Float,
    ) : Stats()

    data class Final(override val rtf: Float) : Stats()
}

@Composable
private fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}

/* ------------------------------------------------------------------ sheets */

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ParamsSheet(
    settings: SettingsRepository.Settings,
    onIndexRateChange: (Float) -> Unit,
    onPitchChange: (Float) -> Unit,
    onRmsMixRateChange: (Float) -> Unit,
    onProtectChange: (Float) -> Unit,
    onFilterRadiusChange: (Int) -> Unit,
    onF0MarginChange: (Int) -> Unit,
    onReset: () -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = Spacing.lg)
                .padding(bottom = Spacing.xxl),
            verticalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            Text(
                text = stringResource(R.string.params_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = stringResource(R.string.params_hint),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            ParamSlider(
                label = stringResource(R.string.home_index_rate),
                valueText = if (settings.indexRate <= 0f) stringResource(R.string.params_index_off)
                else stringResource(R.string.params_index_on, settings.indexRate),
                value = settings.indexRate,
                range = 0f..1f,
                onChange = onIndexRateChange,
            )
            ParamSlider(
                label = stringResource(R.string.home_pitch),
                valueText = stringResource(R.string.home_pitch_value, settings.pitchSemitones.toInt()),
                value = settings.pitchSemitones,
                range = -12f..12f,
                steps = 23,
                onChange = { onPitchChange(it) },
            )
            ParamSlider(
                label = stringResource(R.string.home_rms_mix_rate),
                valueText = stringResource(R.string.home_rms_mix_rate_value, settings.rmsMixRate),
                value = settings.rmsMixRate,
                range = 0f..1f,
                onChange = onRmsMixRateChange,
            )
            ParamSlider(
                label = stringResource(R.string.home_protect),
                valueText = stringResource(R.string.home_protect_value, settings.protect),
                value = settings.protect,
                range = 0f..0.5f,
                onChange = onProtectChange,
            )
            ParamSlider(
                label = stringResource(R.string.home_filter_radius),
                valueText = stringResource(R.string.home_filter_radius_value, settings.filterRadius),
                value = settings.filterRadius.toFloat(),
                range = 1f..7f,
                steps = 5,
                onChange = { onFilterRadiusChange(it.toInt()) },
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = stringResource(R.string.home_f0_margin),
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Text(
                        text = stringResource(R.string.home_f0_margin_hint),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = settings.f0Margin > 0,
                    onCheckedChange = { onF0MarginChange(if (it) 1 else 0) },
                )
            }

            TextButton(
                onClick = onReset,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(stringResource(R.string.params_reset))
            }
        }
    }
}

@Composable
private fun ParamSlider(
    label: String,
    valueText: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    onChange: (Float) -> Unit,
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = valueText,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range,
            steps = steps,
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VoiceSheet(
    voices: List<VoiceManager.Voice>,
    selected: String,
    onSelect: (String) -> Unit,
    onManage: () -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = Spacing.lg)
                .padding(bottom = Spacing.xxl),
            verticalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            Text(
                text = stringResource(R.string.voice_sheet_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            if (voices.isEmpty()) {
                Text(
                    text = stringResource(R.string.home_voice_none_hint),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            voices.forEach { v ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .clickable { onSelect(v.name) }
                        .padding(vertical = Spacing.sm),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
                ) {
                    RadioButton(selected = v.name == selected, onClick = { onSelect(v.name) })
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = v.name,
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            text = if (v.hasIndex) "index ready"
                            else stringResource(R.string.voice_no_index),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (v.name == selected) {
                        Icon(
                            imageVector = Icons.Filled.Check,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
            Spacer(Modifier.height(Spacing.sm))
            OutlinedButton(onClick = onManage, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.voice_sheet_manage))
            }
        }
    }
}
