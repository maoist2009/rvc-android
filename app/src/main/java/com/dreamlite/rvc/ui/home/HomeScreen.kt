package com.dreamlite.rvc.ui.home

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.dreamlite.rvc.R
import com.dreamlite.rvc.engine.RvcState
import com.dreamlite.rvc.ui.theme.Spacing

@Composable
fun HomeScreen(
    viewModel: HomeViewModel,
    onRequestMic: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val state by viewModel.state.collectAsState()
    val level by viewModel.level.collectAsState()
    val context = LocalContext.current

    val micGranted = remember {
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
    }

    Column(
        modifier = modifier.padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        Text(
            text = stringResource(R.string.home_subtitle),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        RecordCard(
            state = state,
            level = level,
            micGranted = micGranted,
            onRecord = { viewModel.startLive() },
            onStop = { viewModel.stop() },
            onPlayConverted = { viewModel.playConverted() },
            onRequestMic = onRequestMic,
        )

        StatsCard(state = state)
    }
}

@Composable
private fun RecordCard(
    state: RvcState,
    level: Float,
    micGranted: Boolean,
    onRecord: () -> Unit,
    onStop: () -> Unit,
    onPlayConverted: () -> Unit,
    onRequestMic: () -> Unit,
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        ),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            val running = state is RvcState.Running
            val pulse by animateFloatAsState(
                targetValue = if (running) 1f else 0.85f,
                animationSpec = spring(dampingRatio = 0.4f, stiffness = 300f),
                label = "pulse",
            )
            val scale = if (running) 1f + level * 0.12f else pulse
            Box(
                modifier = Modifier
                    .size(112.dp * scale)
                    .clip(CircleShape)
                    .background(
                        if (running) MaterialTheme.colorScheme.errorContainer
                        else MaterialTheme.colorScheme.primaryContainer
                    ),
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    imageVector = if (running) Icons.Filled.Stop else Icons.Filled.Mic,
                    contentDescription = null,
                    tint = if (running) MaterialTheme.colorScheme.onErrorContainer
                    else MaterialTheme.colorScheme.onPrimaryContainer,
                    modifier = Modifier.size(48.dp),
                )
            }

            Text(
                text = when (state) {
                    is RvcState.Running -> stringResource(R.string.home_status_running)
                    is RvcState.Loading -> state.label
                    is RvcState.Done -> stringResource(R.string.home_recorded)
                    else -> stringResource(R.string.home_status_idle)
                },
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )

            if (micGranted) {
                Button(
                    onClick = if (running) onStop else onRecord,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = if (running) Icons.Filled.Stop else Icons.Filled.Mic,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.size(Spacing.sm))
                    Text(
                        text = if (running) stringResource(R.string.home_stop)
                        else stringResource(R.string.home_record),
                        fontWeight = FontWeight.Medium,
                    )
                }
            } else {
                Button(onClick = onRequestMic, modifier = Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.perm_mic_rationale))
                }
            }

            AnimatedVisibility(
                visible = state is RvcState.Done,
                enter = fadeIn(tween(200)),
                exit = fadeOut(tween(200)),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(Spacing.md),
                ) {
                    FilledTonalButton(
                        onClick = onPlayConverted,
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Filled.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.size(Spacing.sm))
                        Text(stringResource(R.string.home_play_converted))
                    }
                }
            }

            if (state is RvcState.Error) {
                Text(
                    text = state.message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

@Composable
private fun StatsCard(state: RvcState) {
    val stats = state as? RvcState.Running
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer
        ),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            Text(
                text = stringResource(R.string.stats_title),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
            if (stats == null) {
                Text(
                    text = stringResource(R.string.home_no_output),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                val rtfOk = stats.rtf < 0.8f
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
                            style = MaterialTheme.typography.displaySmall,
                            fontWeight = FontWeight.Bold,
                            color = if (rtfOk) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.error,
                        )
                    }
                    Text(
                        text = stringResource(R.string.stats_rtf_target),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                LinearProgressIndicator(
                    progress = { (stats.rtf / 1.2f).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                )
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
                Text(
                    text = "${stats.blocks} ${stringResource(R.string.stats_blocks)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
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
