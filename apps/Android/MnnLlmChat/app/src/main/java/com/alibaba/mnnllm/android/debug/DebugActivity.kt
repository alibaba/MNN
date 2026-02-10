// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.debug

import android.Manifest
import android.content.Intent
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
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
import android.widget.Spinner
import android.widget.ArrayAdapter
import android.widget.FrameLayout
import android.view.LayoutInflater
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.asr.AsrService
import com.alibaba.mnnllm.android.audio.AudioChunksPlayer
import com.alibaba.mnnllm.android.utils.VoiceModelPathUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.BuildConfig
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.taobao.meta.avatar.tts.TtsService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class TestCase(
    val id: String,
    val title: String,
    val layoutResId: Int
)

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
    private lateinit var testCaseSpinner: Spinner
    private lateinit var testCaseContainer: FrameLayout
    private lateinit var clearLogButton: Button
    private lateinit var copyLogButton: Button

    // Test case views - will be initialized when layouts are loaded
    private var asrTestButton: Button? = null
    private var ttsTestButton: Button? = null
    private var ttsInputText: EditText? = null
    private var ttsProcessButton: Button? = null
    private var showModelInfoSwitch: Switch? = null
    private var allowNetworkSwitch: Switch? = null
    private var networkDelaySwitch: Switch? = null
    private var videoDecoderTestButton: Button? = null
    private var videoDecoderProcessButton: Button? = null
    private var closeDebugModeButton: Button? = null

    private var recognizeService: AsrService? = null
    private var isRecording = false
    private var ttsService: TtsService? = null
    private var audioPlayer: AudioChunksPlayer? = null
    private var isTtsInitialized = false

    private val testCases = listOf(
        TestCase("asr", "ASR Test", R.layout.debug_test_asr),
        TestCase("tts", "TTS Test", R.layout.debug_test_tts),
        TestCase("video", "Video Decoder Test", R.layout.debug_test_video),
        TestCase("scan", "Model Scan Test", R.layout.debug_test_scan),
        TestCase("settings", "Debug Settings", R.layout.debug_test_settings)
    )

    private var scanModelButton: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug)

        initViews()
        setupSpinner()
        loadDebugSettings()
        log("Debug Activity started")
    }

    private fun initViews() {
        scrollView = findViewById(R.id.scrollView)
        logTextView = findViewById(R.id.logTextView)
        testCaseSpinner = findViewById(R.id.testCaseSpinner)
        testCaseContainer = findViewById(R.id.testCaseContainer)
        clearLogButton = findViewById(R.id.clearLogButton)
        copyLogButton = findViewById(R.id.copyLogButton)
        
        val titleTextView = findViewById<TextView>(R.id.titleTextView)
        val baseTitle = getString(R.string.debug_activity_title)
        val buildType = if (BuildConfig.DEBUG) "Debug" else "Release"
        titleTextView.text = "$baseTitle ($buildType)"

        setupClickListeners()
    }

    private fun setupSpinner() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            testCases.map { it.title }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        testCaseSpinner.adapter = adapter

        testCaseSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                loadTestCaseLayout(testCases[position])
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {
                // Do nothing
            }
        }

        // Load first test case by default
        if (testCases.isNotEmpty()) {
            loadTestCaseLayout(testCases[0])
        }
    }

    private fun loadTestCaseLayout(testCase: TestCase) {
        log("Loading test case: ${testCase.title}")
        
        // Clear previous layout
        testCaseContainer.removeAllViews()
        
        // Inflate new layout
        val inflater = LayoutInflater.from(this)
        val view = inflater.inflate(testCase.layoutResId, testCaseContainer, false)
        testCaseContainer.addView(view)
        
        // Initialize views for this test case
        when (testCase.id) {
            "asr" -> initAsrViews(view)
            "tts" -> initTtsViews(view)
            "video" -> initVideoViews(view)
            "scan" -> initScanViews(view)
            "settings" -> initSettingsViews(view)
        }
    }

    private fun initScanViews(parentView: View) {
        scanModelButton = parentView.findViewById(R.id.scanModelButton)
        scanModelButton?.setOnClickListener {
            startModelScanTest()
        }
    }

    private fun startModelScanTest() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                log("=== Calling ModelListManager.debugScanModels ===")
                val result = ModelListManager.debugScanModels(this@DebugActivity)
                
                withContext(Dispatchers.Main) {
                   logTextView.append(result)
                   scrollView.post {
                        scrollView.fullScroll(View.FOCUS_DOWN)
                   }
                }
                
                log("=== Scan Completed ===")
                
            } catch (e: Exception) {
                log("Scan failed: ${e.message}")
                Log.e(TAG, "Scan failed", e)
            }
        }
    }

    private fun initAsrViews(parentView: View) {
        asrTestButton = parentView.findViewById(R.id.asrTestButton)
        asrTestButton?.setOnClickListener {
            if (isRecording) {
                stopAsrTest()
            } else {
                startAsrTest()
            }
        }
    }

    private fun initTtsViews(parentView: View) {
        ttsTestButton = parentView.findViewById(R.id.ttsTestButton)
        ttsInputText = parentView.findViewById(R.id.ttsInputText)
        ttsProcessButton = parentView.findViewById(R.id.ttsProcessButton)

        ttsTestButton?.setOnClickListener {
            if (isTtsInitialized) {
                stopTtsTest()
            } else {
                startTtsTest()
            }
        }

        ttsProcessButton?.setOnClickListener {
            processTtsText()
        }
    }

    private fun initVideoViews(parentView: View) {
        videoDecoderTestButton = parentView.findViewById(R.id.videoDecoderTestButton)
        videoDecoderProcessButton = parentView.findViewById(R.id.videoDecoderProcessButton)

        videoDecoderTestButton?.setOnClickListener {
            startVideoDecoderTest()
        }

        videoDecoderProcessButton?.setOnClickListener {
            processVideoFile()
        }
    }

    private fun initSettingsViews(parentView: View) {
        showModelInfoSwitch = parentView.findViewById(R.id.showModelInfoSwitch)
        allowNetworkSwitch = parentView.findViewById(R.id.allowNetworkSwitch)
        networkDelaySwitch = parentView.findViewById(R.id.networkDelaySwitch)
        closeDebugModeButton = parentView.findViewById(R.id.closeDebugModeButton)

        // Load current settings
        val isModelInfoEnabled = PreferenceUtils.getBoolean(this, KEY_SHOW_MODEL_INFO_ENABLED, false)
        showModelInfoSwitch?.isChecked = isModelInfoEnabled

        val isAllowNetwork = PreferenceUtils.getBoolean(this, KEY_ALLOW_NETWORK_MARKET_DATA, true)
        allowNetworkSwitch?.isChecked = isAllowNetwork

        val isNetworkDelayEnabled = PreferenceUtils.getBoolean(this, KEY_ENABLE_NETWORK_DELAY, false)
        networkDelaySwitch?.isChecked = isNetworkDelayEnabled

        // Setup listeners
        showModelInfoSwitch?.setOnCheckedChangeListener { _, isChecked ->
            PreferenceUtils.setBoolean(this, KEY_SHOW_MODEL_INFO_ENABLED, isChecked)
            log("Model info menu visibility: ${if (isChecked) "enabled" else "disabled"}")
        }

        allowNetworkSwitch?.setOnCheckedChangeListener { _, isChecked ->
            PreferenceUtils.setBoolean(this, KEY_ALLOW_NETWORK_MARKET_DATA, isChecked)
            log("Allow network to fetch model market data: ${if (isChecked) "enabled" else "disabled"}")
        }

        networkDelaySwitch?.setOnCheckedChangeListener { _, isChecked ->
            PreferenceUtils.setBoolean(this, KEY_ENABLE_NETWORK_DELAY, isChecked)
            log("Network delay simulation: ${if (isChecked) "enabled" else "disabled"}")
        }

        closeDebugModeButton?.setOnClickListener {
            closeDebugMode()
        }
    }

    private fun setupClickListeners() {
        clearLogButton.setOnClickListener {
            clearLog()
        }
        copyLogButton.setOnClickListener {
            copyLog()
        }
    }

    private fun loadDebugSettings() {
        log("Debug settings loaded")
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
                    asrTestButton?.text = getString(R.string.stop_asr_test)
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
            asrTestButton?.text = getString(R.string.start_asr_test)
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

    private fun copyLog() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Debug Log", logTextView.text)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(this, R.string.log_copied_to_clipboard, Toast.LENGTH_SHORT).show()
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
                                ttsTestButton?.text = getString(R.string.stop_tts_test)
                                ttsInputText?.isEnabled = true
                                ttsProcessButton?.isEnabled = true
                                ttsInputText?.setText("Hello, this is a test of the TTS system.")
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
            ttsTestButton?.text = getString(R.string.start_tts_test)
            ttsInputText?.isEnabled = false
            ttsProcessButton?.isEnabled = false
            log("TTS test stopped")
        } catch (e: Exception) {
            log("Error stopping TTS test: ${e.message}")
            Log.e(TAG, "Error stopping TTS test", e)
        }
    }

    private fun processTtsText() {
        val text = ttsInputText?.text.toString().trim()
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

    private fun startVideoDecoderTest() {
        log("Starting Video Decoder Test Activity...")
        val intent = Intent(this, VideoDecoderTestActivity::class.java)
        startActivity(intent)
    }

    private fun processVideoFile() {
        log("Video file processing not implemented yet")
        Toast.makeText(this, "Video file processing not implemented yet", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAsrTest()
        stopTtsTest()
    }
}