package com.alibaba.mnntalk.engine

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import androidx.core.content.ContextCompat
import com.k2fsa.sherpa.mnn.OnlineCtcFstDecoderConfig
import com.k2fsa.sherpa.mnn.OnlineRecognizer
import com.k2fsa.sherpa.mnn.OnlineRecognizerConfig
import com.k2fsa.sherpa.mnn.getEndpointConfig
import com.k2fsa.sherpa.mnn.getFeatureConfig
import com.k2fsa.sherpa.mnn.getModelConfigFromDirectory
import com.k2fsa.sherpa.mnn.getOnlineLMConfigFromDirectory
import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class StreamingAsr(
    private val context: Context,
    private val modelDirectory: File
) : Closeable {
    interface Listener {
        fun onPartial(text: String)
        fun onFinal(text: String)
        fun onError(error: Throwable)
    }

    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val CHUNK_MILLIS = 100
    }

    private val recording = AtomicBoolean(false)
    private var recognizer: OnlineRecognizer? = null
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private var echoCanceler: AcousticEchoCanceler? = null
    private var noiseSuppressor: NoiseSuppressor? = null

    suspend fun initialize() = withContext(Dispatchers.IO) {
        check(modelDirectory.isDirectory) {
            "ASR model directory not found: ${modelDirectory.absolutePath}"
        }
        val modelConfig = checkNotNull(getModelConfigFromDirectory(modelDirectory.absolutePath)) {
            "Unsupported ASR model: ${modelDirectory.absolutePath}"
        }
        val config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(SAMPLE_RATE, 80),
            modelConfig = modelConfig,
            lmConfig = getOnlineLMConfigFromDirectory(modelDirectory.absolutePath),
            ctcFstDecoderConfig = OnlineCtcFstDecoderConfig("", 3000),
            endpointConfig = getEndpointConfig(),
            enableEndpoint = true,
            decodingMethod = "greedy_search",
            maxActivePaths = 4,
            hotwordsFile = "",
            hotwordsScore = 1.5f,
            ruleFsts = "",
            ruleFars = ""
        )
        recognizer = OnlineRecognizer(null, config)
    }

    fun start(listener: Listener) {
        if (recording.get()) return
        checkNotNull(recognizer) { "ASR is not initialized" }
        check(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        ) { "Microphone permission is not granted" }

        val channel = AudioFormat.CHANNEL_IN_MONO
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, channel, encoding)
        check(minimum > 0) { "Unable to determine microphone buffer size" }

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            SAMPLE_RATE,
            channel,
            encoding,
            minimum * 2
        )
        check(recorder.state == AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            "Unable to initialize microphone"
        }
        audioRecord = recorder
        enableAudioEffects(recorder.audioSessionId)

        recording.set(true)
        recorder.startRecording()
        recordingThread = Thread(
            { processMicrophone(recorder, listener) },
            "mnn-voice-asr"
        ).apply { start() }
    }

    fun stop() {
        if (!recording.compareAndSet(true, false)) return
        runCatching { audioRecord?.stop() }
        recordingThread?.interrupt()
        runCatching { recordingThread?.join(500) }
        recordingThread = null
        releaseRecorder()
    }

    private fun processMicrophone(recorder: AudioRecord, listener: Listener) {
        val activeRecognizer = recognizer ?: return
        val stream = activeRecognizer.createStream("")
        val chunkSize = SAMPLE_RATE * CHUNK_MILLIS / 1000
        val pcm = ShortArray(chunkSize)
        var lastPartial = ""

        try {
            while (recording.get()) {
                val count = recorder.read(pcm, 0, pcm.size)
                if (count <= 0) continue
                val samples = FloatArray(count) { index -> pcm[index] / 32768.0f }
                stream.acceptWaveform(samples, SAMPLE_RATE)
                while (activeRecognizer.isReady(stream)) {
                    activeRecognizer.decode(stream)
                }

                val text = activeRecognizer.getResult(stream).text.trim()
                if (text.isNotEmpty() && text != lastPartial) {
                    lastPartial = text
                    listener.onPartial(text)
                }

                if (activeRecognizer.isEndpoint(stream)) {
                    if (text.isNotEmpty()) {
                        listener.onFinal(text)
                    }
                    activeRecognizer.reset(stream)
                    lastPartial = ""
                    listener.onPartial("")
                }
            }
        } catch (error: Throwable) {
            if (recording.get()) {
                listener.onError(error)
            }
        } finally {
            stream.release()
        }
    }

    private fun enableAudioEffects(audioSessionId: Int) {
        if (AcousticEchoCanceler.isAvailable()) {
            echoCanceler = AcousticEchoCanceler.create(audioSessionId)?.apply {
                enabled = true
            }
        }
        if (NoiseSuppressor.isAvailable()) {
            noiseSuppressor = NoiseSuppressor.create(audioSessionId)?.apply {
                enabled = true
            }
        }
    }

    private fun releaseRecorder() {
        echoCanceler?.release()
        echoCanceler = null
        noiseSuppressor?.release()
        noiseSuppressor = null
        audioRecord?.release()
        audioRecord = null
    }

    override fun close() {
        stop()
        recognizer?.release()
        recognizer = null
    }
}
