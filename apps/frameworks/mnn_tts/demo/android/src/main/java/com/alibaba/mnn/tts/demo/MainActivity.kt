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
    
    // State
    private var allModels: List<Pair<String, ModelConfig>> = emptyList()
    private var selectedModelPath: String? = null
    private var currentSpeakerId: String = ""
    private var currentLanguage: String = "en"
    private var isTtsInitialized = false

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
        }
    }

    private fun loadTtsModelAndPlay(modelPath: String, text: String) {
        if (isTtsInitialized) {
            try {
                ttsService.destroy()
                isTtsInitialized = false
            } catch (e: Exception) {
                Log.e("TTS_TEST", "Error destroying previous TTS service", e)
            }
        }

        ttsService = TtsService()
        
        lifecycleScope.launch {
            try {
                resultText.text = "Loading model: ${File(modelPath).name}..."
                val initResult = ttsService.init(modelPath)
                if (initResult) {
                    isTtsInitialized = true
                    if (currentLanguage.isNotEmpty()) {
                        ttsService.setLanguage(currentLanguage)
                    }
                    processTtsText(text)
                } else {
                    resultText.text = "Failed to load model: ${File(modelPath).name}"
                }
            } catch (e: Exception) {
                Log.e("TTS_TEST", "Error initializing TTS service", e)
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
                            Log.d("TTS_TEST", "Found model: ${file.absolutePath}")
                        }
                    }
                }
            } else {
                Log.w("TTS_TEST", "Models directory does not exist: ${modelsDir.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e("TTS_TEST", "Error scanning models directory", e)
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
        if (isTtsInitialized) {
            try {
                ttsService.destroy()
                isTtsInitialized = false
            } catch (e: Exception) {
                Log.e("TTS_TEST", "Error destroying previous TTS service", e)
            }
        }

        ttsService = TtsService()
        
        lifecycleScope.launch {
            try {
                resultText.text = "Loading model: ${File(modelPath).name}..."
                Log.d("TTS_TEST", "Initializing TTS Service with model: $modelPath")
                val initResult = ttsService.init(modelPath)
                if (initResult) {
                    isTtsInitialized = true
                    Log.d("TTS_TEST", "TTS Service initialized successfully")
                    resultText.text = "Model loaded: ${File(modelPath).name}\nTTS Service ready"
                    
                    // Set initial language if selected
                    if (currentLanguage.isNotEmpty()) {
                        ttsService.setLanguage(currentLanguage)
                    }
                } else {
                    Log.e("TTS_TEST", "TTS Service initialization failed")
                    resultText.text = "Failed to load model: ${File(modelPath).name}"
                }
            } catch (e: Exception) {
                Log.e("TTS_TEST", "Error initializing TTS service", e)
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
                
                return ModelConfig(speakers, languages)
            }
        } catch (e: Exception) {
            Log.e("TTS_TEST", "Error reading config.json", e)
        }
        return ModelConfig()
    }

    private fun initAudioPlayer() {
        audioPlayer = AudioChunksPlayer()
        audioPlayer.sampleRate = 44100 // Common TTS sample rate
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
                // Wait for TTS service to be ready
                val isReady = ttsService.waitForInitComplete()
                if (!isReady) {
                    resultText.text = "TTS Service not ready"
                    return@launch
                }

                Log.d("TTS_TEST", "Processing text: $text")
                resultText.text = "Processing: $text"

                // Process text with TTS
                if (currentLanguage.isNotEmpty()) {
                    ttsService.setLanguage(currentLanguage) 
                }
                if (currentSpeakerId.isNotEmpty()) {
                    ttsService.setSpeakerId(currentSpeakerId)
                }
                val audioData = ttsService.process(text, 0)
                Log.d("TTS_TEST", "Generated audio data with ${audioData.size} samples")
                
                // Update UI with results
                resultText.text = "Generated ${audioData.size} audio samples. Playing..."

                // Play the audio
                audioPlayer.playChunk(audioData)
                
                // Wait for playback to complete
                audioPlayer.waitStop()
                
                resultText.text = "Playback completed. Generated ${audioData.size} samples."
                Log.d("TTS_TEST", "Audio playback completed")

            } catch (e: Exception) {
                Log.e("TTS_TEST", "Error processing TTS", e)
                resultText.text = "Error: ${e.message}"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            audioPlayer.destroy()
            ttsService.destroy()
        } catch (e: Exception) {
            Log.e("TTS_TEST", "Error during cleanup", e)
        }
    }
} 