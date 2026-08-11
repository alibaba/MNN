package com.alibaba.mnn.tts.demo

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import org.json.JSONObject
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.security.MessageDigest
import java.util.Locale

import com.taobao.meta.avatar.tts.TtsService
import com.alibaba.mnn.tts.demo.audio.AudioChunksPlayer

class MainActivity : AppCompatActivity() {
    private lateinit var resultText: TextView
    private lateinit var inputText: EditText
    private lateinit var processButton: Button
    private lateinit var modelRecyclerView: RecyclerView
    private lateinit var languageSpinner: Spinner
    private lateinit var ttsService: TtsService
    private lateinit var audioPlayer: AudioChunksPlayer
    private lateinit var modelAdapter: ModelAdapter
    private val roundTripEvaluator = TtsRoundTripEvaluator()
    
    // State
    private var allModels: List<Pair<String, ModelConfig>> = emptyList()
    private var selectedModelPath: String? = null
    private var currentSpeakerId: String = ""
    private var currentLanguage: String = "en"
    private var initializedLanguage: String = ""
    private var isTtsInitialized = false
    private var autoRunTriggered = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initModelList()
        initAudioPlayer()
        setupTtsTest()
    }

    private fun initViews() {
        resultText = findViewById(R.id.resultText)
        inputText = findViewById(R.id.inputText)
        processButton = findViewById(R.id.processButton)
        modelRecyclerView = findViewById(R.id.modelRecyclerView)
        languageSpinner = findViewById(R.id.languageSpinner)
    }

    private fun initModelList() {
        modelAdapter = ModelAdapter(
            onModelSelected = { modelPath, config ->
                if (selectedModelPath != modelPath) {
                    selectedModelPath = modelPath
                    currentSpeakerId = "" // Reset or pick first?
                    loadTtsModel(modelPath)
                }
            },
            onSpeakerSelected = { speakerId ->
                currentSpeakerId = speakerId
                Log.d("TTS_TEST", "Speaker selected: $speakerId")
            },
            onPlayClicked = { modelPath, speakerId ->
                val text = inputText.text.toString().trim()
                if (text.isEmpty()) {
                    resultText.text = "Please enter some text"
                    return@ModelAdapter
                }
                
                currentSpeakerId = speakerId
                if (selectedModelPath == modelPath && isTtsInitialized) {
                    processTtsText(text)
                } else {
                    // Load and then play
                    selectedModelPath = modelPath
                    loadTtsModelAndPlay(modelPath, text)
                }
            }
        )
        modelRecyclerView.layoutManager = LinearLayoutManager(this)
        modelRecyclerView.adapter = modelAdapter

        lifecycleScope.launch {
            allModels = scanTtsModels()
            setupLanguageFilter()
            maybeHandleAutoRunIntent()
        }
    }

    private fun loadTtsModelAndPlay(modelPath: String, text: String) {
        lifecycleScope.launch {
            try {
                resultText.text = "Loading model: ${File(modelPath).name}..."
                val resolvedLanguage = TtsLanguageResolver.resolve(text, currentLanguage)
                val initResult = initTtsService(modelPath, resolvedLanguage)
                if (initResult) {
                    processTtsText(text)
                } else {
                    resultText.text = "Failed to load model: ${File(modelPath).name}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing TTS service", e)
                resultText.text = "Error loading model: ${e.message}"
            }
        }
    }

    private suspend fun scanTtsModels(): List<Pair<String, ModelConfig>> = withContext(Dispatchers.IO) {
        val modelsDir = File("/data/local/tmp/tts_models")
        val modelList = mutableListOf<Pair<String, ModelConfig>>()

        try {
            if (modelsDir.exists() && modelsDir.isDirectory) {
                modelsDir.listFiles()?.forEach { file ->
                    if (file.isDirectory) {
                        // Check if it contains config.json
                        val configFile = File(file, "config.json")
                        if (configFile.exists()) {
                            val config = readModelConfig(file.absolutePath)
                            modelList.add(file.absolutePath to config)
                            Log.d(TAG, "Found model: ${file.absolutePath}")
                        }
                    }
                }
            } else {
                Log.w(TAG, "Models directory does not exist: ${modelsDir.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error scanning models directory", e)
        }

        modelList.sortedBy { it.first }
    }
    
    private fun setupLanguageFilter() {
        // Collect all unique languages
        val languages = mutableSetOf<String>()
        allModels.forEach { (_, config) ->
            if (config.languages.isNotEmpty()) {
                languages.addAll(config.languages)
            }
        }
        
        // Convert to list and sort
        val langList = languages.toList().sorted().toMutableList()
        if (langList.isEmpty()) langList.add("en") // Default fallback

        val langAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, langList)
        langAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        languageSpinner.adapter = langAdapter
        
        languageSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                currentLanguage = langList[position]
                // Apply filter
                filterModels(currentLanguage)
                
                // Update TTS service language if initialized
                if (isTtsInitialized) {
                     ttsService.setLanguage(currentLanguage)
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // Initial filter
        if (langList.isNotEmpty()) {
            currentLanguage = langList[0]
            filterModels(currentLanguage)
        }
    }
    
    private fun filterModels(language: String) {
        val filtered = allModels.filter { (_, config) ->
            config.languages.isEmpty() || config.languages.contains(language)
        }.map { it.first }
        
        modelAdapter.updateModels(filtered)
        
        if (filtered.isEmpty()) {
            resultText.text = "No models found for language: $language"
        } else {
             resultText.text = "Found ${filtered.size} models"
        }
    }

    private fun loadTtsModel(modelPath: String) {
        lifecycleScope.launch {
            try {
                resultText.text = "Loading model: ${File(modelPath).name}..."
                configureAudioPlayer(modelPath)
                Log.d(TAG, "Initializing TTS Service with model: $modelPath")
                val initResult = initTtsService(modelPath, currentLanguage)
                if (initResult) {
                    Log.d(TAG, "TTS Service initialized successfully")
                    resultText.text = "Model loaded: ${File(modelPath).name}\nTTS Service ready"
                } else {
                    Log.e(TAG, "TTS Service initialization failed")
                    resultText.text = "Failed to load model: ${File(modelPath).name}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing TTS service", e)
                resultText.text = "Error loading model: ${e.message}"
            }
        }
    }

    private fun readModelConfig(modelPath: String): ModelConfig {
        try {
            val configFile = File(modelPath, "config.json")
            if (configFile.exists()) {
                val content = configFile.readText()
                val json = JSONObject(content)
                
                val speakers = mutableListOf<String>()
                if (json.has("speakers")) {
                    val speakersJson = json.getJSONArray("speakers")
                    for (i in 0 until speakersJson.length()) {
                        speakers.add(speakersJson.getString(i))
                    }
                }

                val languages = mutableListOf<String>()
                if (json.has("languages")) {
                    val languagesJson = json.getJSONArray("languages")
                    for (i in 0 until languagesJson.length()) {
                        languages.add(languagesJson.getString(i))
                    }
                }

                val sampleRate = when (val value = json.opt("sample_rate")) {
                    is Number -> value.toInt()
                    is String -> value.toIntOrNull()
                    else -> null
                } ?: DEFAULT_SAMPLE_RATE

                return ModelConfig(speakers, languages, sampleRate)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading config.json", e)
        }
        return ModelConfig()
    }

    private fun initAudioPlayer() {
        audioPlayer = AudioChunksPlayer()
        audioPlayer.sampleRate = DEFAULT_SAMPLE_RATE
        audioPlayer.start()
    }

    private fun setupTtsTest() {
        processButton.setOnClickListener {
            if (!isTtsInitialized || selectedModelPath == null) {
                resultText.text = "Please select a model first"
                return@setOnClickListener
            }

            val text = inputText.text.toString().trim()
            if (text.isNotEmpty()) {
                processTtsText(text)
            } else {
                resultText.text = "Please enter some text"
            }
        }
    }

    private fun processTtsText(text: String) {
        lifecycleScope.launch {
            try {
                val modelPath = selectedModelPath
                if (modelPath == null) {
                    resultText.text = "Please select a model first"
                    return@launch
                }
                val modelConfig = configureAudioPlayer(modelPath)

                val resolvedLanguage = TtsLanguageResolver.resolve(text, currentLanguage)
                if (!isTtsInitialized || initializedLanguage != resolvedLanguage) {
                    resultText.text = "Loading model: ${File(modelPath).name}..."
                    if (!initTtsService(modelPath, resolvedLanguage)) {
                        resultText.text = "Failed to load model: ${File(modelPath).name}"
                        return@launch
                    }
                }

                // Wait for TTS service to be ready
                val isReady = ttsService.waitForInitComplete()
                if (!isReady) {
                    resultText.text = "TTS Service not ready"
                    return@launch
                }

                Log.d(TAG, "Processing text: $text")
                resultText.text = "Processing: $text"

                // Process text with TTS
                Log.d(TAG, "Resolved language: $resolvedLanguage for text: $text")
                if (currentSpeakerId.isNotEmpty()) {
                    ttsService.setSpeakerId(currentSpeakerId)
                }
                val audioData = ttsService.process(text, 0)
                Log.d(TAG, "Generated audio data with ${audioData.size} samples")
                val audioSha256 = computeAudioSha256(audioData)
                val roundTripResult = withContext(Dispatchers.Default) {
                    roundTripEvaluator.evaluate(audioData, modelConfig.sampleRate, text)
                }
                logRoundTripResult(
                    modelPath = modelPath,
                    language = resolvedLanguage,
                    sampleRate = modelConfig.sampleRate,
                    originalText = text,
                    audioSampleCount = audioData.size,
                    audioSha256 = audioSha256,
                    result = roundTripResult
                )
                
                // Update UI with results
                resultText.text = buildRoundTripSummary(audioData.size, roundTripResult)

                // Play the audio
                audioPlayer.playChunk(audioData)
                
                // Wait for playback to complete
                audioPlayer.waitStop()
                
                resultText.text = buildRoundTripSummary(audioData.size, roundTripResult, completed = true)
                Log.d(TAG, "Audio playback completed")

            } catch (e: Exception) {
                Log.e(TAG, "Error processing TTS", e)
                resultText.text = "Error: ${e.message}"
            }
        }
    }

    private suspend fun initTtsService(modelPath: String, language: String): Boolean {
        if (isTtsInitialized) {
            try {
                ttsService.destroy()
            } catch (e: Exception) {
                Log.e(TAG, "Error destroying previous TTS service", e)
            }
            isTtsInitialized = false
            initializedLanguage = ""
        }

        ttsService = TtsService()
        ttsService.setLanguage(language)
        val initResult = ttsService.init(modelPath)
        if (initResult) {
            isTtsInitialized = true
            initializedLanguage = language
        }
        return initResult
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            audioPlayer.destroy()
            ttsService.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }

    private fun maybeHandleAutoRunIntent() {
        if (autoRunTriggered || !intent.getBooleanExtra(EXTRA_AUTO_RUN, false)) {
            return
        }

        val text = intent.getStringExtra(EXTRA_INPUT_TEXT)?.trim().orEmpty()
        if (text.isEmpty()) {
            resultText.text = "Auto-run text is empty"
            return
        }

        val requestedModelPath = intent.getStringExtra(EXTRA_MODEL_PATH)?.trim().orEmpty()
        val targetModel = when {
            requestedModelPath.isNotEmpty() -> allModels.firstOrNull { it.first == requestedModelPath }
            else -> allModels.firstOrNull()
        }

        if (targetModel == null) {
            resultText.text = if (requestedModelPath.isNotEmpty()) {
                "Auto-run model not found: $requestedModelPath"
            } else {
                "Auto-run found no model"
            }
            Log.e(TAG, "Auto-run failed to resolve model path: $requestedModelPath")
            return
        }

        autoRunTriggered = true
        selectedModelPath = targetModel.first
        currentSpeakerId = targetModel.second.speakers.firstOrNull().orEmpty()
        inputText.setText(text)
        Log.i(TAG, "TTS_AUTORUN_MODEL=${targetModel.first}")
        Log.i(TAG, "TTS_AUTORUN_TEXT=$text")
        loadTtsModelAndPlay(targetModel.first, text)
    }

    private fun configureAudioPlayer(modelPath: String): ModelConfig {
        val modelConfig = readModelConfig(modelPath)
        audioPlayer.sampleRate = modelConfig.sampleRate
        Log.i(TAG, "Configured sample rate ${modelConfig.sampleRate} for $modelPath")
        return modelConfig
    }

    private fun buildRoundTripSummary(
        sampleCount: Int,
        roundTripResult: TtsRoundTripResult,
        completed: Boolean = false
    ): String {
        val prefix = if (completed) "Playback completed." else "Generated audio. Playing..."
        val recognizedText = roundTripResult.recognizedText.ifBlank { "<empty>" }
        val errorSuffix = roundTripResult.error?.let { "\nError: $it" }.orEmpty()
        return "$prefix Generated $sampleCount samples." +
            "\nRoundtrip: ${roundTripResult.status}" +
            "\nASR: $recognizedText" +
            "\nSimilarity: ${String.format(Locale.US, "%.3f", roundTripResult.similarity)}" +
            errorSuffix
    }

    private fun logRoundTripResult(
        modelPath: String,
        language: String,
        sampleRate: Int,
        originalText: String,
        audioSampleCount: Int,
        audioSha256: String,
        result: TtsRoundTripResult
    ) {
        Log.i(TAG, "TTS_MODEL_PATH=$modelPath")
        Log.i(TAG, "TTS_LANGUAGE=$language")
        Log.i(TAG, "TTS_SAMPLE_RATE=$sampleRate")
        Log.i(TAG, "TTS_INPUT_TEXT=$originalText")
        Log.i(TAG, "TTS_AUDIO_SAMPLE_COUNT=$audioSampleCount")
        Log.i(TAG, "TTS_AUDIO_SHA256=$audioSha256")
        Log.i(TAG, "TTS_ROUNDTRIP_TEXT=${result.recognizedText}")
        Log.i(TAG, "TTS_ROUNDTRIP_HAS_CHINESE=${result.hasChinese}")
        Log.i(TAG, "TTS_ROUNDTRIP_SIMILARITY=${String.format(Locale.US, "%.3f", result.similarity)}")
        Log.i(TAG, "TTS_ROUNDTRIP_STATUS=${result.status}")
        if (result.error != null) {
            Log.i(TAG, "TTS_ROUNDTRIP_ERROR=${result.error}")
        }
    }

    private fun computeAudioSha256(audioData: ShortArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        for (sample in audioData) {
            digest.update((sample.toInt() and 0xFF).toByte())
            digest.update(((sample.toInt() ushr 8) and 0xFF).toByte())
        }
        return digest.digest().joinToString(separator = "") { "%02x".format(it) }
    }

    companion object {
        private const val TAG = "TTS_TEST"
        private const val DEFAULT_SAMPLE_RATE = 44100
        const val EXTRA_AUTO_RUN = "auto_run"
        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_INPUT_TEXT = "input_text"
    }
}
