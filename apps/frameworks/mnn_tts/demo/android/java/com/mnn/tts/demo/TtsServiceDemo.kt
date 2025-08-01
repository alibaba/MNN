package com.mnn.tts.demo

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import com.taobao.meta.avatar.tts.TtsService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class TtsServiceDemo(private val context: Context) {
    private val ttsService = TtsService()
    private var audioTrack: AudioTrack? = null
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_OUT_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    init {
        setupAudioTrack()
    }

    private fun setupAudioTrack() {
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            channelConfig,
            audioFormat
        )

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(audioFormat)
                    .setChannelMask(channelConfig)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    suspend fun initialize(modelDir: String) {
        val result = ttsService.init(modelDir)
        if (!result) {
            Log.e(TAG, "Failed to initialize TTS service")
        }
    }

    fun speak(text: String, language: String = "en") {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                ttsService.setLanguage(language)
                if (!ttsService.waitForInitComplete()) {
                    Log.e(TAG, "TTS not initialized")
                    return@launch
                }

                val audio = ttsService.process(text, 0)
                audioTrack?.let { track ->
                    track.play()
                    track.write(audio, 0, audio.size)
                    track.stop()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in speak", e)
            }
        }
    }

    fun destroy() {
        audioTrack?.release()
        audioTrack = null
        ttsService.destroy()
    }

    companion object {
        private const val TAG = "TtsServiceDemo"
    }
} 