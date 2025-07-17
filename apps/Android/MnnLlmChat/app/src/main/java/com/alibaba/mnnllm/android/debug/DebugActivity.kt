// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.debug

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.asr.AsrService
import com.alibaba.mnnllm.android.audio.AudioChunksPlayer
import com.alibaba.mnnllm.android.utils.VoiceModelPathUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.taobao.meta.avatar.tts.TtsService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class DebugActivity : AppCompatActivity() {

    companion object {
        const val TAG = "DebugActivity"
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 1001
        private const val KEY_SHOW_MODEL_INFO_ENABLED = "debug_show_model_info_enabled"
        private const val KEY_ALLOW_NETWORK_MARKET_DATA = "debug_allow_network_market_data"
        private const val KEY_ENABLE_NETWORK_DELAY = "debug_enable_network_delay"
        
        @JvmStatic
        fun isShowModelInfoEnabled(context: android.content.Context): Boolean {
            return PreferenceUtils.getBoolean(context, KEY_SHOW_MODEL_INFO_ENABLED, false)
        }
        
        @JvmStatic
        fun isNetworkDelayEnabled(context: android.content.Context): Boolean {
            return PreferenceUtils.getBoolean(context, KEY_ENABLE_NETWORK_DELAY, false)
        }
        
        @JvmStatic
        fun checkRepoUpdates(context: android.content.Context, callback: (Boolean, String?) -> Unit) {
            CoroutineScope(Dispatchers.IO).launch {
//                delay(300)
            }
        }
    }

    private lateinit var scrollView: ScrollView
    private lateinit var logTextView: TextView
    private lateinit var asrTestButton: Button
    private lateinit var clearLogButton: Button
    private lateinit var closeDebugModeButton: Button
    private lateinit var ttsTestButton: Button
    private lateinit var ttsInputText: EditText
    private lateinit var ttsProcessButton: Button
    private lateinit var showModelInfoSwitch: Switch
    private lateinit var allowNetworkSwitch: Switch
    private lateinit var networkDelaySwitch: Switch
    
    private var recognizeService: AsrService? = null
    private var isRecording = false
    private var ttsService: TtsService? = null
    private var audioPlayer: AudioChunksPlayer? = null
    private var isTtsInitialized = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug)
        
        initViews()
        setupClickListeners()
        loadDebugSettings()
        log("Debug Activity started")
    }

    private fun initViews() {
        scrollView = findViewById(R.id.scrollView)
        logTextView = findViewById(R.id.logTextView)
        asrTestButton = findViewById(R.id.asrTestButton)
        clearLogButton = findViewById(R.id.clearLogButton)
        closeDebugModeButton = findViewById(R.id.closeDebugModeButton)
        ttsTestButton = findViewById(R.id.ttsTestButton)
        ttsInputText = findViewById(R.id.ttsInputText)
        ttsProcessButton = findViewById(R.id.ttsProcessButton)
        showModelInfoSwitch = findViewById(R.id.showModelInfoSwitch)
        allowNetworkSwitch = findViewById(R.id.allowNetworkSwitch)
        networkDelaySwitch = findViewById(R.id.networkDelaySwitch)
    }

    private fun setupClickListeners() {
        asrTestButton.setOnClickListener {
            if (isRecording) {
                stopAsrTest()
            } else {
                startAsrTest()
            }
        }

        clearLogButton.setOnClickListener {
            clearLog()
        }

        closeDebugModeButton.setOnClickListener {
            closeDebugMode()
        }

        ttsTestButton.setOnClickListener {
            if (isTtsInitialized) {
                stopTtsTest()
            } else {
                startTtsTest()
            }
        }

        ttsProcessButton.setOnClickListener {
            processTtsText()
        }

        showModelInfoSwitch.setOnCheckedChangeListener { _, isChecked ->
            PreferenceUtils.setBoolean(this, KEY_SHOW_MODEL_INFO_ENABLED, isChecked)
            log("Model info menu visibility: ${if (isChecked) "enabled" else "disabled"}")
        }

        allowNetworkSwitch.setOnCheckedChangeListener { _, isChecked ->
            PreferenceUtils.setBoolean(this, KEY_ALLOW_NETWORK_MARKET_DATA, isChecked)
            log("Allow network to fetch model market data: ${if (isChecked) "enabled" else "disabled"}")
        }

        networkDelaySwitch.setOnCheckedChangeListener { _, isChecked ->
            PreferenceUtils.setBoolean(this, KEY_ENABLE_NETWORK_DELAY, isChecked)
            log("Network delay simulation: ${if (isChecked) "enabled" else "disabled"}")
        }
    }

    private fun loadDebugSettings() {
        val isModelInfoEnabled = PreferenceUtils.getBoolean(this, KEY_SHOW_MODEL_INFO_ENABLED, false)
        showModelInfoSwitch.isChecked = isModelInfoEnabled
        log("Loaded debug settings - Model info menu: ${if (isModelInfoEnabled) "enabled" else "disabled"}")

        val isAllowNetwork = PreferenceUtils.getBoolean(this, KEY_ALLOW_NETWORK_MARKET_DATA, true)
        allowNetworkSwitch.isChecked = isAllowNetwork
        log("Loaded debug settings - Allow network: ${if (isAllowNetwork) "enabled" else "disabled"}")

        val isNetworkDelayEnabled = PreferenceUtils.getBoolean(this, KEY_ENABLE_NETWORK_DELAY, false)
        networkDelaySwitch.isChecked = isNetworkDelayEnabled
        log("Loaded debug settings - Network delay: ${if (isNetworkDelayEnabled) "enabled" else "disabled"}")
    }



    private fun startAsrTest() {
        if (checkRecordAudioPermission()) {
            CoroutineScope(Dispatchers.Main).launch {
                try {
                log("Starting ASR test...")
                    val modelDir = "/data/local/tmp/asr_models/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20"
                    log("Using ASR model path: $modelDir")
                    recognizeService = AsrService(this@DebugActivity, modelDir)
                    
                    withContext(Dispatchers.IO) {
                        recognizeService?.initRecognizer()
                    }
                    recognizeService?.onRecognizeText = { text ->
                        runOnUiThread {
                            log("ASR Result: $text")
                        }
                    }

                    recognizeService?.startRecord()
                    isRecording = true
                    asrTestButton.text = getString(R.string.stop_asr_test)
                    log("ASR test started successfully")
                    
                } catch (e: Exception) {
                    log("ASR test failed: ${e.message}")
                    Log.e(TAG, "ASR test failed", e)
                }
            }
        }
    }

    private fun stopAsrTest() {
        try {
            recognizeService?.stopRecord()
            recognizeService = null
            isRecording = false
            asrTestButton.text = getString(R.string.start_asr_test)
            log("ASR test stopped")
        } catch (e: Exception) {
            log("Error stopping ASR test: ${e.message}")
            Log.e(TAG, "Error stopping ASR test", e)
        }
    }

    private fun checkRecordAudioPermission(): Boolean {
        return when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                true
            }
            shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) -> {
                Toast.makeText(this, R.string.recording_permission_denied, Toast.LENGTH_LONG).show()
                false
            }
            else -> {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    REQUEST_RECORD_AUDIO_PERMISSION
                )
                false
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        when (requestCode) {
            REQUEST_RECORD_AUDIO_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    log("Record audio permission granted")
                    startAsrTest()
                } else {
                    log("Record audio permission denied")
                    Toast.makeText(this, R.string.recording_permission_denied, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun log(message: String) {
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())
        val logMessage = "[$timestamp] $message\n"
        
        runOnUiThread {
            logTextView.append(logMessage)
            scrollView.post {
                scrollView.fullScroll(View.FOCUS_DOWN)
            }
        }
        
        Log.d(TAG, message)
    }

    private fun clearLog() {
        logTextView.text = ""
        log("Log cleared")
    }

    private fun closeDebugMode() {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle(R.string.close_debug_mode_title)
            .setMessage(R.string.close_debug_mode_message)
            .setPositiveButton(R.string.close_debug_mode) { _, _ ->
                // Save debug mode deactivation state to SharedPreferences
                val sharedPreferences = getSharedPreferences("com.alibaba.mnnllm.android_preferences", MODE_PRIVATE)
                sharedPreferences.edit().putBoolean("debug_mode_activated", false).apply()
                
                log("Debug mode deactivated")
                Toast.makeText(this, getString(R.string.debug_mode_closed_message), Toast.LENGTH_LONG).show()
                
                // Close the activity
                finish()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun startTtsTest() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                log("Starting TTS test...")
                ttsService = TtsService()
                audioPlayer = AudioChunksPlayer()
                
                withContext(Dispatchers.IO) {
                    try {
                        val modelDir = VoiceModelPathUtils.getTtsModelPath(this@DebugActivity)
                        log("Using TTS model path: $modelDir")
                        log("Initializing TTS with model directory: $modelDir")
                        val initResult = ttsService?.init(modelDir)
                        if (initResult == true) {
                            log("TTS Service initialized successfully")
                            isTtsInitialized = true
                            withContext(Dispatchers.Main) {
                                ttsTestButton.text = getString(R.string.stop_tts_test)
                                ttsInputText.isEnabled = true
                                ttsProcessButton.isEnabled = true
                                ttsInputText.setText("Hello, this is a test of the TTS system.")
                            }
                            log("TTS test started successfully")
                        } else {
                            log("TTS Service initialization failed")
                            Log.e(TAG, "TTS Service initialization failed")
                        }
                    } catch (e: Exception) {
                        log("TTS initialization error: ${e.message}")
                        Log.e(TAG, "TTS initialization error", e)
                    }
                }
                
            } catch (e: Exception) {
                log("TTS test failed: ${e.message}")
                Log.e(TAG, "TTS test failed", e)
            }
        }
    }

    private fun stopTtsTest() {
        try {
            audioPlayer?.destroy()
            ttsService?.destroy()
            ttsService = null
            audioPlayer = null
            isTtsInitialized = false
            ttsTestButton.text = getString(R.string.start_tts_test)
            ttsInputText.isEnabled = false
            ttsProcessButton.isEnabled = false
            log("TTS test stopped")
        } catch (e: Exception) {
            log("Error stopping TTS test: ${e.message}")
            Log.e(TAG, "Error stopping TTS test", e)
        }
    }

    private fun processTtsText() {
        val text = ttsInputText.text.toString().trim()
        if (text.isEmpty()) {
            log("Please enter some text")
            return
        }

        if (!isTtsInitialized || ttsService == null) {
            log("TTS Service not initialized. Please start TTS test first.")
            return
        }

        lifecycleScope.launch {
            try {
                log("Processing TTS text: $text")
                
                // Wait for TTS service to be ready
                val isReady = ttsService?.waitForInitComplete()
                if (isReady != true) {
                    log("TTS Service not ready")
                    return@launch
                }

                log("TTS Service is ready, processing text...")

                // Process text with TTS
                val audioData = ttsService?.process(text, 0)
                if (audioData != null && audioData.isNotEmpty()) {
                    log("Generated audio data with ${audioData.size} samples")
                    
                    // Update UI with results
                    log("Generated ${audioData.size} audio samples. Playing...")

                    // Initialize audio player if needed
                    if (audioPlayer == null) {
                        audioPlayer = AudioChunksPlayer()
                        audioPlayer!!.sampleRate = 44100
                    }
                    audioPlayer?.start()

                    // Play the audio
                    audioPlayer?.playChunk(audioData)
                    
                    // Wait for playback to complete
                    audioPlayer?.waitStop()
                    
                    log("Playback completed. Generated ${audioData.size} samples.")
                } else {
                    log("Failed to generate audio data or audio data is empty")
                }

            } catch (e: Exception) {
                log("Error processing TTS: ${e.message}")
                Log.e(TAG, "Error processing TTS", e)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAsrTest()
        stopTtsTest()
    }
}