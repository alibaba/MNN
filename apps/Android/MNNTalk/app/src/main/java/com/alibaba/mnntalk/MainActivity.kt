package com.alibaba.mnntalk

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.alibaba.mnntalk.engine.GenerationMetrics
import com.alibaba.mnntalk.engine.LocalVoiceChatEngine
import com.alibaba.mnntalk.engine.Speaker
import com.alibaba.mnntalk.engine.TranscriptLine
import com.alibaba.mnntalk.engine.VoiceChatState
import com.alibaba.mnntalk.model.ModelBundleState
import com.alibaba.mnntalk.model.VoiceModelBundleManager
import com.alibaba.mnntalk.model.VoiceModelPaths
import com.alibaba.mnntalk.model.VoiceModelSource
import com.alibaba.mnntalk.ui.VoiceOrbView
import java.io.File
import java.util.Locale
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    companion object {
        private const val EXTRA_MODEL_DIRECTORY = "mnn_model_dir"
        private const val EXTRA_CLEAR_MODEL_DIRECTORY = "clear_mnn_model_dir"
        private const val PREFERENCES_NAME = "developer_models"
        private const val PREFERENCE_MODEL_DIRECTORY = "model_directory"
    }

    private lateinit var offlineBadge: TextView
    private lateinit var voiceOrb: VoiceOrbView
    private lateinit var statusText: TextView
    private lateinit var statusHint: TextView
    private lateinit var downloadPanel: View
    private lateinit var downloadLabel: TextView
    private lateinit var downloadProgress: ProgressBar
    private lateinit var transcriptText: TextView
    private lateinit var metricsText: TextView
    private lateinit var primaryButton: Button
    private lateinit var newChatButton: Button

    private lateinit var modelManager: VoiceModelBundleManager
    private lateinit var voiceEngine: LocalVoiceChatEngine
    private var modelState: ModelBundleState = ModelBundleState.Missing
    private var voiceState: VoiceChatState = VoiceChatState.Idle
    private var prepareJob: Job? = null
    private var hasTranscripts = false

    private val microphonePermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            voiceEngine.startConversation()
        } else {
            Toast.makeText(this, R.string.permission_microphone, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()

        modelManager = VoiceModelBundleManager(
            applicationContext,
            readConfiguredModelDirectory()
        )
        voiceEngine = LocalVoiceChatEngine(applicationContext)

        primaryButton.setOnClickListener { onPrimaryAction() }
        newChatButton.setOnClickListener { voiceEngine.newConversation() }
        collectState()
    }

    private fun bindViews() {
        offlineBadge = findViewById(R.id.offlineBadge)
        voiceOrb = findViewById(R.id.voiceOrb)
        statusText = findViewById(R.id.statusText)
        statusHint = findViewById(R.id.statusHint)
        downloadPanel = findViewById(R.id.downloadPanel)
        downloadLabel = findViewById(R.id.downloadLabel)
        downloadProgress = findViewById(R.id.downloadProgress)
        transcriptText = findViewById(R.id.transcriptText)
        metricsText = findViewById(R.id.metricsText)
        primaryButton = findViewById(R.id.primaryButton)
        newChatButton = findViewById(R.id.newChatButton)
    }

    private fun collectState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    modelManager.state.collectLatest {
                        modelState = it
                        renderModelState(it)
                        if (it is ModelBundleState.Ready &&
                            (voiceState is VoiceChatState.Idle || voiceState is VoiceChatState.Error)
                        ) {
                            prepareModels(it.paths)
                        }
                    }
                }
                launch {
                    voiceEngine.state.collectLatest {
                        voiceState = it
                        renderVoiceState(it)
                    }
                }
                launch {
                    voiceEngine.transcripts.collectLatest(::renderTranscripts)
                }
                launch {
                    voiceEngine.metrics.collectLatest(::renderMetrics)
                }
            }
        }
    }

    private fun onPrimaryAction() {
        when (val bundle = modelState) {
            ModelBundleState.Missing,
            is ModelBundleState.Error -> modelManager.download()

            is ModelBundleState.Downloading -> Unit
            is ModelBundleState.Ready -> when (voiceState) {
                VoiceChatState.Idle,
                is VoiceChatState.Preparing,
                is VoiceChatState.Error -> prepareModels(bundle.paths)

                VoiceChatState.Ready -> requestMicrophoneAndStart()
                is VoiceChatState.Listening,
                is VoiceChatState.Thinking,
                is VoiceChatState.Speaking -> voiceEngine.stopConversation()
            }
        }
    }

    private fun prepareModels(paths: VoiceModelPaths) {
        if (prepareJob?.isActive == true) return
        prepareJob = lifecycleScope.launch {
            runCatching { voiceEngine.prepare(paths) }
        }
    }

    private fun requestMicrophoneAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            voiceEngine.startConversation()
        } else {
            microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun renderModelState(state: ModelBundleState) {
        when (state) {
            ModelBundleState.Missing -> {
                offlineBadge.setText(R.string.model_badge)
                downloadPanel.visibility = View.VISIBLE
                downloadLabel.setText(R.string.download_waiting)
                downloadProgress.progress = 0
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_download)
            }

            is ModelBundleState.Downloading -> {
                offlineBadge.text = "正在下载"
                downloadPanel.visibility = View.VISIBLE
                downloadLabel.text = getString(
                    R.string.download_progress,
                    state.label,
                    formatBytes(state.downloadedBytes),
                    formatBytes(state.totalBytes)
                )
                downloadProgress.progress = (state.progress * downloadProgress.max).toInt()
                primaryButton.isEnabled = false
                primaryButton.text = getString(
                    R.string.button_downloading,
                    (state.progress * 100).toInt()
                )
                statusText.setText(R.string.status_preparing)
                statusHint.text = "模型只需下载一次，完成后可以断网使用"
                setVisualMode(VoiceOrbView.Mode.PREPARING)
            }

            is ModelBundleState.Ready -> {
                offlineBadge.setText(
                    if (state.paths.source == VoiceModelSource.DEVELOPER_DIRECTORY) {
                        R.string.developer_directory_badge
                    } else {
                        R.string.offline_badge
                    }
                )
                downloadPanel.visibility = View.GONE
                if (voiceState is VoiceChatState.Idle) {
                    primaryButton.isEnabled = false
                    primaryButton.setText(R.string.button_prepare)
                }
            }

            is ModelBundleState.Error -> {
                offlineBadge.text = "下载失败"
                downloadPanel.visibility = View.VISIBLE
                downloadLabel.text = state.message
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_retry)
                statusText.setText(R.string.status_error)
                statusHint.text = state.message
                setVisualMode(VoiceOrbView.Mode.ERROR)
            }
        }
    }

    private fun renderVoiceState(state: VoiceChatState) {
        when (state) {
            VoiceChatState.Idle -> {
                setVisualMode(VoiceOrbView.Mode.PREPARING)
                statusText.setText(R.string.status_preparing)
                statusHint.setText(R.string.hint_ready)
            }

            is VoiceChatState.Preparing -> {
                setVisualMode(VoiceOrbView.Mode.PREPARING)
                statusText.setText(R.string.status_preparing)
                statusHint.text = state.detail
                primaryButton.isEnabled = false
                primaryButton.setText(R.string.button_prepare)
            }

            VoiceChatState.Ready -> {
                setVisualMode(VoiceOrbView.Mode.IDLE)
                statusText.setText(R.string.status_ready)
                statusHint.setText(R.string.hint_ready)
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_start)
                newChatButton.visibility =
                    if (hasTranscripts) View.VISIBLE else View.GONE
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }

            is VoiceChatState.Listening -> {
                setVisualMode(VoiceOrbView.Mode.LISTENING)
                statusText.setText(R.string.status_listening)
                statusHint.text = state.partialText.ifBlank {
                    getString(R.string.hint_listening)
                }
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_stop)
                newChatButton.visibility = View.VISIBLE
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }

            is VoiceChatState.Thinking -> {
                setVisualMode(VoiceOrbView.Mode.THINKING)
                statusText.setText(R.string.status_thinking)
                statusHint.setText(R.string.hint_interrupt)
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_stop)
            }

            is VoiceChatState.Speaking -> {
                setVisualMode(VoiceOrbView.Mode.SPEAKING)
                statusText.setText(R.string.status_speaking)
                statusHint.setText(R.string.hint_interrupt)
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_stop)
            }

            is VoiceChatState.Error -> {
                setVisualMode(VoiceOrbView.Mode.ERROR)
                statusText.setText(R.string.status_error)
                statusHint.text = state.message
                primaryButton.isEnabled = true
                primaryButton.setText(R.string.button_retry)
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
    }

    private fun setVisualMode(mode: VoiceOrbView.Mode) {
        voiceOrb.setMode(mode)
    }

    private fun renderTranscripts(lines: List<TranscriptLine>) {
        if (lines.isEmpty()) {
            hasTranscripts = false
            transcriptText.setText(R.string.transcript_empty)
            transcriptText.setTextColor(ContextCompat.getColor(this, R.color.ink_secondary))
            newChatButton.visibility = View.GONE
            return
        }

        hasTranscripts = true
        val builder = SpannableStringBuilder()
        lines.filter { it.text.isNotBlank() }.forEachIndexed { index, line ->
            if (index > 0) builder.append("\n\n")
            val label = if (line.speaker == Speaker.USER) {
                getString(R.string.speaker_you)
            } else {
                getString(R.string.speaker_mnn)
            }
            val labelStart = builder.length
            builder.append(label)
            builder.setSpan(
                StyleSpan(Typeface.BOLD),
                labelStart,
                builder.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            builder.setSpan(
                ForegroundColorSpan(
                    ContextCompat.getColor(
                        this,
                        if (line.speaker == Speaker.USER) R.color.mint else R.color.brand
                    )
                ),
                labelStart,
                builder.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            builder.append("\n").append(line.text.trim())
        }
        transcriptText.text = builder
        transcriptText.setTextColor(ContextCompat.getColor(this, R.color.ink))
        newChatButton.visibility = View.VISIBLE
    }

    private fun renderMetrics(metrics: GenerationMetrics?) {
        if (metrics == null) {
            metricsText.visibility = View.GONE
            return
        }
        metricsText.visibility = View.VISIBLE
        metricsText.text = String.format(
            Locale.US,
            "纯端侧 · Prefill %.1f tok/s · Decode %.1f tok/s",
            metrics.firstStageTokensPerSecond,
            metrics.decodeTokensPerSecond
        )
    }

    private fun formatBytes(bytes: Long): String {
        val gibibytes = bytes / (1024f * 1024f * 1024f)
        return if (gibibytes >= 1f) {
            String.format(Locale.US, "%.1f GB", gibibytes)
        } else {
            String.format(Locale.US, "%.0f MB", bytes / (1024f * 1024f))
        }
    }

    private fun readConfiguredModelDirectory(): File? {
        val preferences = getSharedPreferences(PREFERENCES_NAME, MODE_PRIVATE)
        if (intent.getBooleanExtra(EXTRA_CLEAR_MODEL_DIRECTORY, false)) {
            preferences.edit().remove(PREFERENCE_MODEL_DIRECTORY).apply()
        }
        if (intent.hasExtra(EXTRA_MODEL_DIRECTORY)) {
            val requestedDirectory = intent.getStringExtra(EXTRA_MODEL_DIRECTORY)?.trim()
            if (requestedDirectory.isNullOrEmpty()) {
                preferences.edit().remove(PREFERENCE_MODEL_DIRECTORY).apply()
            } else {
                preferences.edit()
                    .putString(PREFERENCE_MODEL_DIRECTORY, requestedDirectory)
                    .apply()
            }
        }
        return preferences.getString(PREFERENCE_MODEL_DIRECTORY, null)
            ?.takeIf { it.isNotBlank() }
            ?.let(::File)
    }

    override fun onDestroy() {
        prepareJob?.cancel()
        voiceEngine.close()
        modelManager.close()
        super.onDestroy()
    }
}
