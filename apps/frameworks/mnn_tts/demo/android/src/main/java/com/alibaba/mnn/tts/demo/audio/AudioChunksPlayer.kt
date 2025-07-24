package com.alibaba.mnn.tts.demo.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AudioChunksPlayer {
    private var audioTrack: AudioTrack? = null
    var sampleRate: Int = 22050 // 默认采样率
        set(value) {
            field = value
            // 当采样率改变时，需要重新初始化 AudioTrack
            if (isPlaying) {
                stop()
                start()
            }
        }

    private var isPlaying = false
    private val TAG = "AudioChunksPlayer"

    fun start() {
        if (isPlaying) return

        try {
            val bufferSize = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
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
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            audioTrack?.play()
            isPlaying = true
            Log.d(TAG, "AudioTrack started with sample rate: $sampleRate")
        } catch (e: Exception) {
            Log.e(TAG, "Error starting AudioTrack", e)
            isPlaying = false
        }
    }

    suspend fun playChunk(audioData: ShortArray) = withContext(Dispatchers.IO) {
        try {
            if (!isPlaying) {
                start()
            }
            audioTrack?.write(audioData, 0, audioData.size)
            Log.d(TAG, "Played audio chunk of size: ${audioData.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Error playing audio chunk", e)
        }
    }

    suspend fun waitStop() = withContext(Dispatchers.IO) {
        try {
            audioTrack?.let {
                it.stop()
                it.flush()
                it.play()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error waiting for audio to stop", e)
        }
    }

    fun stop() {
        try {
            audioTrack?.stop()
            isPlaying = false
            Log.d(TAG, "AudioTrack stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping AudioTrack", e)
        }
    }

    fun destroy() {
        try {
            stop()
            audioTrack?.release()
            audioTrack = null
            Log.d(TAG, "AudioTrack destroyed")
        } catch (e: Exception) {
            Log.e(TAG, "Error destroying AudioTrack", e)
        }
    }
} 