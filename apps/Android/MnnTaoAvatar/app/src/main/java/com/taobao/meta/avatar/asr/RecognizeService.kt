// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.asr

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.mnn.OnlineCtcFstDecoderConfig
import com.k2fsa.sherpa.mnn.OnlineRecognizer
import com.k2fsa.sherpa.mnn.OnlineRecognizerConfig
import com.k2fsa.sherpa.mnn.getEndpointConfig
import com.k2fsa.sherpa.mnn.getFeatureConfig
import com.k2fsa.sherpa.mnn.getModelConfig
import com.k2fsa.sherpa.mnn.getOnlineLMConfig
import com.taobao.meta.avatar.MainActivity
import com.taobao.meta.avatar.record.RecordPermission.REQUEST_RECORD_AUDIO_PERMISSION
import com.taobao.meta.avatar.utils.DeviceUtils
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class RecognizeService(private val activity: MainActivity) {

    private val permissions = arrayOf(Manifest.permission.RECORD_AUDIO)

    companion object {
        const val TAG = "OnlineRecorder"
    }

    private val initComplete = CompletableDeferred<Boolean>()
    private var recognizer: OnlineRecognizer? = null
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private val audioSource = MediaRecorder.AudioSource.MIC
    private val sampleRateInHz = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val isRecording = AtomicBoolean(false)
    @Volatile
    private var isLoaded = false
    var onRecognizeText: ((String) -> Unit)? = null

    private val acceptTimeNs = AtomicLong(0)
    private val decodeTimeNs = AtomicLong(0)
    private val chunkCount = AtomicLong(0)
    private val decodeCount = AtomicLong(0)

    private var utteranceProcTimeNs: Long = 0
    private var utteranceAudioTimeSec: Double = 0.0
    private var totalRtf: Double = 0.0
    private var utteranceCount: Int = 0

    suspend fun initRecognizer() {
        val type = if (DeviceUtils.isChinese) 0 else 1
        Log.i(TAG, "Select model type $type")
        val config = getModelConfig(type)?.let {
            OnlineRecognizerConfig(
                getFeatureConfig(sampleRateInHz, 80),
                it,
                getOnlineLMConfig(type),
                OnlineCtcFstDecoderConfig("", 3000),
                getEndpointConfig(),
                true,
                "greedy_search",
                4,
                "",
                1.5f,
                "", ""
            )
        }!!
        CoroutineScope(Dispatchers.IO).async {
            recognizer = OnlineRecognizer(null, config)
        }.await()
        isLoaded = true
        initComplete.complete(true)
    }

    private fun initMicrophone(): Boolean {
        if (ActivityCompat.checkSelfPermission(
                activity,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(activity, permissions, REQUEST_RECORD_AUDIO_PERMISSION)
            return false
        }
        val numBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)
        Log.i(TAG, "buffer size in milliseconds: ${numBytes * 1000.0f / sampleRateInHz}")
        audioRecord = AudioRecord(audioSource, sampleRateInHz, channelConfig, audioFormat, numBytes * 2)
        return true
    }

    private fun processSamples() {
        Log.i(TAG, "processing samples")
        val stream = recognizer!!.createStream("")
        val interval = 0.1 // i.e., 100 ms
        val bufferSize = (interval * sampleRateInHz).toInt() // in samples
        val buffer = ShortArray(bufferSize)

        while (isRecording.get() && audioRecord != null) {
            val ret = audioRecord!!.read(buffer, 0, buffer.size)
            if (ret > 0) {
                chunkCount.incrementAndGet()
                val samples = FloatArray(ret) { i -> buffer[i] / 32768.0f }
                utteranceAudioTimeSec += ret.toDouble() / sampleRateInHz
                val tStartProc = System.nanoTime()
                val tAcceptStart = tStartProc
                stream.acceptWaveform(samples, sampleRateInHz)
                val tAcceptEnd = System.nanoTime()
                acceptTimeNs.addAndGet(tAcceptEnd - tAcceptStart)
                while (recognizer!!.isReady(stream)) {
                    val tDecodeStart = System.nanoTime()
                    recognizer!!.decode(stream)
                    val tDecodeEnd = System.nanoTime()
                    decodeTimeNs.addAndGet(tDecodeEnd - tDecodeStart)
                    decodeCount.incrementAndGet()
                }
                val tEndProc = System.nanoTime()
                utteranceProcTimeNs += (tEndProc - tStartProc)

                val isEndpoint = recognizer!!.isEndpoint(stream)
                var text = recognizer!!.getResult(stream).text

                if (isEndpoint && recognizer!!.config.modelConfig.paraformer.encoder.isNotEmpty()) {
                    val tailPaddings = FloatArray((0.8 * sampleRateInHz).toInt())
                    stream.acceptWaveform(tailPaddings, sampleRateInHz)
                    while (recognizer!!.isReady(stream)) {
                        recognizer!!.decode(stream)
                    }
                    text = recognizer!!.getResult(stream).text
                }

                if (isEndpoint) {
                    val T_proc = utteranceProcTimeNs / 1_000_000_000.0
                    val T_audio = utteranceAudioTimeSec
                    val rtf = if (T_audio > 0) T_proc / T_audio else 0.0
                    totalRtf += rtf
                    utteranceCount += 1
//                    Log.i(TAG, "Utterance RTF = ${"%.3f".format(rtf)} over ${"%.2f".format(T_audio)}s audio")
                    recognizer!!.reset(stream)
                    if (text.isNotEmpty()) {
                        onRecognizeText?.invoke(text)
                        Log.d(TAG, "recognize text: $text")
                    }
                    utteranceProcTimeNs = 0
                    utteranceAudioTimeSec = 0.0
                }
            }
        }
        stream.release()
        val totalChunks = chunkCount.get().takeIf { it > 0 } ?: 1
        val totalDecodes = decodeCount.get().takeIf { it > 0 } ?: 1
        val avgAcceptMs = acceptTimeNs.get() / totalChunks / 1_000_000.0
        val avgDecodeMs = decodeTimeNs.get() / totalDecodes / 1_000_000.0
        Log.i(TAG, "Average acceptWaveform: ${"%.2f".format(avgAcceptMs)} ms over $totalChunks chunks")
        Log.i(TAG, "Average decode: ${"%.2f".format(avgDecodeMs)} ms over $totalDecodes calls")

        if (utteranceCount > 0) {
            val avgRtf = totalRtf / utteranceCount
            Log.i(TAG, "Average RTF over $utteranceCount utterances = ${"%.3f".format(avgRtf)}")
        }
        acceptTimeNs.set(0)
        decodeTimeNs.set(0)
        chunkCount.set(0)
        decodeCount.set(0)
        totalRtf = 0.0
        utteranceCount = 0
    }

    fun stopRecord() {
        Log.i(TAG, "stopRecord isRecording: ${isRecording.get()}")
        if (!isRecording.get()) {
            return
        }
        isRecording.set(false)
        if (audioRecord != null) {
            audioRecord!!.stop()
            audioRecord!!.release()
            audioRecord = null
            Log.i(TAG, "Stopped recording")
        }
    }

    fun startRecord() {
        if (isRecording.get()) {
            return
        }
        val ret = initMicrophone()
        if (!ret) {
            Log.e(TAG, "Failed to initialize microphone")
            return
        }
        audioRecord?.let {
            Log.i(TAG, "state: ${it.state}")
            it.startRecording()
            isRecording.set(true)
            recordingThread = Thread { this.processSamples() }
            recordingThread!!.start()
            Log.i(TAG, "Started recording")
        }
    }
}