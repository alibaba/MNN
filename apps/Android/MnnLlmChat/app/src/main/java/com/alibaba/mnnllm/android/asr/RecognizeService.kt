// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.asr

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import com.alibaba.mnnllm.android.utils.Permissions.REQUEST_RECORD_AUDIO_PERMISSION
import com.k2fsa.sherpa.mnn.OnlineCtcFstDecoderConfig
import com.k2fsa.sherpa.mnn.OnlineRecognizer
import com.k2fsa.sherpa.mnn.OnlineRecognizerConfig
import com.k2fsa.sherpa.mnn.getEndpointConfig
import com.k2fsa.sherpa.mnn.getFeatureConfig
import com.k2fsa.sherpa.mnn.getModelConfig
import com.k2fsa.sherpa.mnn.getOnlineLMConfig
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import java.util.concurrent.atomic.AtomicBoolean

class RecognizeService(private val activity: Activity) {

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
    @Volatile
    private var initStarted = false

    var onRecognizeText: ((String) -> Unit)? = null

    suspend fun initRecognizer() {
        if (initStarted) {
            initComplete.await()
        }
        initStarted = true
        val type = 0
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
        Log.i(TAG, "buffer size in milliseconds: " + (numBytes * 1000.0f / sampleRateInHz))
        audioRecord =
            AudioRecord(audioSource, sampleRateInHz, channelConfig, audioFormat, numBytes * 2)
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
                val samples = FloatArray(ret)
                for (i in 0 until ret) {
                    samples[i] = buffer[i] / 32768.0f
                }
                stream.acceptWaveform(samples, sampleRateInHz)
                while (recognizer!!.isReady(stream)) {
                    recognizer!!.decode(stream)
                }
                val isEndpoint = recognizer!!.isEndpoint(stream)
                var text = recognizer!!.getResult(stream).text

                if (isEndpoint && !recognizer!!.config.modelConfig.paraformer.encoder.isEmpty()) {
                    val tailPaddings = FloatArray((0.8 * sampleRateInHz).toInt())
                    stream.acceptWaveform(tailPaddings, sampleRateInHz)
                    while (recognizer!!.isReady(stream)) {
                        recognizer!!.decode(stream)
                    }
                    text = recognizer!!.getResult(stream).text
                }
                if (isEndpoint) {
                    recognizer!!.reset(stream)
                    if (text.isNotEmpty()) {
                        onRecognizeText?.invoke(text)
                        Log.d(TAG, "recorgnize text  :${text}")
                    }
                }
            }
        }
        stream.release()
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
        if (audioRecord != null) {
            Log.i(TAG, "state: " + audioRecord!!.state)
            audioRecord?.startRecording()
            isRecording.set(true)
            recordingThread = Thread { this.processSamples() }
            recordingThread!!.start()
            Log.i(TAG, "Started recording")
        }
    }
}