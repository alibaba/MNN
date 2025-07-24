package com.alibaba.mnn.tts.demo

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

import com.taobao.meta.avatar.tts.TtsService
import com.alibaba.mnn.tts.demo.audio.AudioChunksPlayer

class MainActivity : AppCompatActivity() {
    private lateinit var resultText: TextView
    private lateinit var inputText: EditText
    private lateinit var processButton: Button
    private lateinit var ttsService: TtsService
    private lateinit var audioPlayer: AudioChunksPlayer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initTtsService()
        initAudioPlayer()
        setupTtsTest()
    }

    private fun initViews() {
        resultText = findViewById(R.id.resultText)
        inputText = findViewById(R.id.inputText)
        processButton = findViewById(R.id.processButton)
    }

    private fun initTtsService() {
        ttsService = TtsService()
        lifecycleScope.launch {
            try {
                val modelDir = "/data/local/tmp/test_new_tts/bert-vits/"
                val initResult = ttsService.init(modelDir)
                if (initResult) {
                    Log.d("TTS_TEST", "TTS Service initialized successfully")
                    resultText.text = "TTS Service ready"
                } else {
                    Log.e("TTS_TEST", "TTS Service initialization failed")
                    resultText.text = "TTS Service initialization failed"
                }
            } catch (e: Exception) {
                Log.e("TTS_TEST", "Error initializing TTS service", e)
                resultText.text = "Error: ${e.message}"
            }
        }
    }

    private fun initAudioPlayer() {
        audioPlayer = AudioChunksPlayer()
        audioPlayer.sampleRate = 44100 // Common TTS sample rate
        audioPlayer.start()
    }

    private fun setupTtsTest() {
        processButton.setOnClickListener {
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