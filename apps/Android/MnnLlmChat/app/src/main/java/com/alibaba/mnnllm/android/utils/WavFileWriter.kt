// Created by ruoyi.sjd Assistant
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils

import android.util.Log
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Utility class for writing PCM audio data to WAV files
 */
class WavFileWriter(
    private val filePath: String,
    private val sampleRate: Int = 24000,
    private val channels: Int = 1,
    private val bitsPerSample: Int = 16
) {
    companion object {
        private const val TAG = "WavFileWriter"
    }

    private val audioData = mutableListOf<FloatArray>()
    private var totalSamples = 0

    /**
     * Add audio chunk to be written to WAV file
     */
    fun addAudioChunk(data: FloatArray) {
        audioData.add(data.copyOf())
        totalSamples += data.size
        Log.d(TAG, "Added audio chunk with ${data.size} samples, total: $totalSamples")
    }

    /**
     * Write all collected audio data to WAV file
     */
    fun writeToFile(): Boolean {
        return try {
            FileOutputStream(filePath).use { fos ->
                // Calculate data size
                val dataSize = totalSamples * channels * (bitsPerSample / 8)
                val fileSize = 36 + dataSize

                // Write WAV header
                writeWavHeader(fos, fileSize, dataSize)

                // Convert and write audio data
                writeAudioData(fos)

                Log.d(TAG, "Successfully wrote WAV file: $filePath, samples: $totalSamples")
                true
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to write WAV file: $filePath", e)
            false
        }
    }

    private fun writeWavHeader(fos: FileOutputStream, fileSize: Int, dataSize: Int) {
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        header.put("RIFF".toByteArray())
        header.putInt(fileSize)
        header.put("WAVE".toByteArray())

        // fmt chunk
        header.put("fmt ".toByteArray())
        header.putInt(16) // PCM format chunk size
        header.putShort(1) // Audio format (1 = PCM)
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * channels * (bitsPerSample / 8)) // Byte rate
        header.putShort((channels * (bitsPerSample / 8)).toShort()) // Block align
        header.putShort(bitsPerSample.toShort())

        // data chunk
        header.put("data".toByteArray())
        header.putInt(dataSize)

        fos.write(header.array())
    }

    private fun writeAudioData(fos: FileOutputStream) {
        for (chunk in audioData) {
            val buffer = ByteBuffer.allocate(chunk.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in chunk) {
                // Convert float (-1.0 to 1.0) to 16-bit PCM
                val limitedSample = sample.coerceIn(-1.0f, 1.0f)
                val shortSample = (limitedSample * 32767.0f).toInt().toShort()
                buffer.putShort(shortSample)
            }
            fos.write(buffer.array())
        }
    }

    /**
     * Clear all collected audio data
     */
    fun clear() {
        audioData.clear()
        totalSamples = 0
        Log.d(TAG, "Cleared audio data")
    }

    /**
     * Get total number of samples collected
     */
    fun getTotalSamples(): Int = totalSamples
} 