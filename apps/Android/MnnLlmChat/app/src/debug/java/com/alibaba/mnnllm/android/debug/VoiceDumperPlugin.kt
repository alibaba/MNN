// Created by smoke-test automation on 2026/03/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.debug

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.utils.VoiceModelPathUtils
import com.alibaba.mnnllm.android.utils.WavFileWriter
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import com.k2fsa.sherpa.mnn.OnlineCtcFstDecoderConfig
import com.k2fsa.sherpa.mnn.OnlineRecognizer
import com.k2fsa.sherpa.mnn.OnlineRecognizerConfig
import com.k2fsa.sherpa.mnn.getEndpointConfig
import com.k2fsa.sherpa.mnn.getFeatureConfig
import com.k2fsa.sherpa.mnn.getModelConfigFromDirectory
import com.k2fsa.sherpa.mnn.getOnlineLMConfigFromDirectory
import com.taobao.meta.avatar.tts.TtsService
import kotlinx.coroutines.runBlocking
import java.io.File
import java.io.PrintStream
import kotlin.math.max

/**
 * Stetho DumperPlugin for testing Voice (TTS/ASR) capabilities via adb command line.
 *
 * Usage:
 *   dumpapp voice status                      - Check voice models status
 *   dumpapp voice tts init [modelPath]        - Initialize TTS service
 *   dumpapp voice tts test <text>             - Test TTS with given text
 *   dumpapp voice tts destroy                 - Destroy TTS service
 *   dumpapp voice asr status                  - Check ASR service status
 *
 * Examples:
 *   dumpapp voice status
 *   dumpapp voice tts init
 *   dumpapp voice tts init /data/local/tmp/tts_models/default
 *   dumpapp voice tts test "Hello world"
 */
internal class VoiceDumperPlugin(
    private val contextProvider: () -> Context = { MnnLlmApplication.getInstance() },
    private val ttsClientFactory: () -> TtsClient = { RealTtsClient(TtsService()) },
    private val roundTripEvaluatorFactory: () -> TtsRoundTripEvaluator = { RealTtsRoundTripEvaluator() },
    private val audioArtifactWriter: TtsAudioArtifactWriter = WavTtsAudioArtifactWriter()
) : DumperPlugin {
    companion object {
        private const val TAG = "VoiceDumperPlugin"
    }

    // Shared TTS service instance for the duration of testing
    private var ttsService: TtsClient? = null
    private var isTtsInitialized = false
    private var initializedTtsModelPath: String? = null

    override fun getName(): String = "voice"

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "status" -> handleStatus(writer)
            "tts" -> handleTts(writer, args.drop(1))
            "asr" -> handleAsr(writer, args.drop(1))
            else -> doUsage(writer)
        }
    }

    private fun getContext(): Context {
        return contextProvider()
    }

    private fun handleStatus(writer: PrintStream) {
        val context = getContext()
        val (isReady, message) = VoiceModelPathUtils.checkVoiceModelsStatus(context)

        writer.println("=== Voice Models Status ===")
        writer.println("READY=$isReady")
        writer.println("MESSAGE=$message")
        writer.println()

        // Show detailed paths
        val ttsModelPath = VoiceModelPathUtils.getTtsModelPath(context)
        writer.println("TTS_MODEL_PATH=$ttsModelPath")
        writer.println("TTS_LANGUAGE=${VoiceModelPathUtils.getTtsLanguage(context)}")
        writer.println("TTS_SAMPLE_RATE=${VoiceModelPathUtils.getTtsSampleRate(ttsModelPath)}")
        writer.println("ASR_MODEL_PATH=${VoiceModelPathUtils.getAsrModelPath(context)}")
        writer.println()

        // Show internal state
        writer.println("TTS_SERVICE_INITIALIZED=$isTtsInitialized")
    }

    private fun handleTts(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            doTtsUsage(writer)
            return
        }

        when (args[0]) {
            "init" -> handleTtsInit(writer, args.drop(1))
            "test" -> handleTtsTest(writer, args.drop(1))
            "destroy" -> handleTtsDestroy(writer)
            "status" -> handleTtsStatus(writer)
            else -> doTtsUsage(writer)
        }
    }

    private fun handleTtsInit(writer: PrintStream, args: List<String>) {
        val context = getContext()

        if (isTtsInitialized && ttsService != null) {
            writer.println("TTS_INIT=ALREADY_INITIALIZED")
            return
        }

        try {
            val modelPath = args.firstOrNull().takeUnless { it.isNullOrBlank() }
                ?: VoiceModelPathUtils.getTtsModelPath(context)
            val language = VoiceModelPathUtils.getTtsLanguage(context)
            writer.println("TTS_MODEL_PATH=$modelPath")
            writer.println("TTS_LANGUAGE=$language")

            val startTime = System.currentTimeMillis()
            ttsService = ttsClientFactory()
            ttsService?.setLanguage(language)
            val initResult = runBlocking { ttsService?.init(modelPath) ?: false }
            val initTime = System.currentTimeMillis() - startTime

            if (initResult) {
                isTtsInitialized = true
                initializedTtsModelPath = modelPath
                writer.println("TTS_INIT=SUCCESS")
                writer.println("TTS_INIT_TIME_MS=$initTime")
                Log.i(TAG, "TTS initialized successfully in ${initTime}ms")
            } else {
                ttsService?.destroy()
                ttsService = null
                initializedTtsModelPath = null
                writer.println("TTS_INIT=FAIL")
                writer.println("TTS_INIT_ERROR=init() returned false")
                Log.e(TAG, "TTS initialization failed")
            }
        } catch (e: Exception) {
            ttsService?.destroy()
            ttsService = null
            isTtsInitialized = false
            initializedTtsModelPath = null
            writer.println("TTS_INIT=FAIL")
            writer.println("TTS_INIT_ERROR=${e.message}")
            Log.e(TAG, "TTS initialization exception", e)
        }
    }

    private fun handleTtsTest(writer: PrintStream, args: List<String>) {
        if (!isTtsInitialized || ttsService == null) {
            writer.println("TTS_TEST=FAIL")
            writer.println("TTS_TEST_ERROR=TTS not initialized. Run 'dumpapp voice tts init' first.")
            return
        }

        val text = if (args.isNotEmpty()) {
            args.joinToString(" ")
        } else {
            "Hello, this is a TTS smoke test."
        }

        writer.println("TTS_TEST_INPUT=$text")

        try {
            // Wait for init to complete
            val isReady = runBlocking { ttsService?.waitForInitComplete() ?: false }
            if (!isReady) {
                writer.println("TTS_TEST=FAIL")
                writer.println("TTS_TEST_ERROR=TTS service not ready")
                return
            }

            val startTime = System.currentTimeMillis()
            val audioData = ttsService?.process(text, 0)
            val processTime = System.currentTimeMillis() - startTime

            if (audioData != null && audioData.isNotEmpty()) {
                val sampleCount = audioData.size
                val modelPath = initializedTtsModelPath ?: VoiceModelPathUtils.getTtsModelPath(getContext())
                val sampleRate = VoiceModelPathUtils.getTtsSampleRate(modelPath)
                val durationSec = sampleCount.toFloat() / sampleRate.toFloat()
                val textLength = text.length
                // RTF = processing time / audio duration
                val rtf = if (durationSec > 0) (processTime / 1000f) / durationSec else 0f
                val wavPath = runCatching {
                    audioArtifactWriter.write(getContext(), audioData, sampleRate)
                }

                writer.println("TTS_TEST=SUCCESS")
                writer.println("TTS_MODEL_PATH=$modelPath")
                writer.println("TTS_SAMPLE_RATE=$sampleRate")
                writer.println("TTS_AUDIO_SAMPLES=$sampleCount")
                writer.println("TTS_AUDIO_DURATION_SEC=${"%.2f".format(durationSec)}")
                writer.println("TTS_PROCESS_TIME_MS=$processTime")
                writer.println("TTS_RTF=${"%.3f".format(rtf)}")
                writer.println("TTS_TEXT_LENGTH=$textLength")
                wavPath.onSuccess { writer.println("TTS_AUDIO_WAV_PATH=$it") }
                wavPath.onFailure { writer.println("TTS_AUDIO_WAV_ERROR=${it.message}") }

                val roundTrip = roundTripEvaluatorFactory().evaluate(getContext(), audioData, sampleRate, text)
                writer.println("TTS_ROUNDTRIP_TEXT=${roundTrip.recognizedText}")
                writer.println("TTS_ROUNDTRIP_HAS_CHINESE=${roundTrip.hasChinese}")
                writer.println("TTS_ROUNDTRIP_SIMILARITY=${"%.3f".format(roundTrip.similarity)}")
                writer.println("TTS_ROUNDTRIP_STATUS=${roundTrip.status}")
                roundTrip.error?.let { writer.println("TTS_ROUNDTRIP_ERROR=$it") }

                Log.i(TAG, "TTS test success: $sampleCount samples, ${durationSec}s, RTF=${rtf}")
            } else {
                writer.println("TTS_TEST=FAIL")
                writer.println("TTS_TEST_ERROR=No audio data generated")
                Log.e(TAG, "TTS test failed: no audio data")
            }
        } catch (e: Exception) {
            writer.println("TTS_TEST=FAIL")
            writer.println("TTS_TEST_ERROR=${e.message}")
            Log.e(TAG, "TTS test exception", e)
        }
    }

    private fun handleTtsDestroy(writer: PrintStream) {
        try {
            ttsService?.destroy()
            ttsService = null
            isTtsInitialized = false
            initializedTtsModelPath = null
            writer.println("TTS_DESTROY=SUCCESS")
            Log.i(TAG, "TTS service destroyed")
        } catch (e: Exception) {
            writer.println("TTS_DESTROY=FAIL")
            writer.println("TTS_DESTROY_ERROR=${e.message}")
            Log.e(TAG, "TTS destroy exception", e)
        }
    }

    private fun handleTtsStatus(writer: PrintStream) {
        writer.println("=== TTS Service Status ===")
        writer.println("TTS_INITIALIZED=$isTtsInitialized")
        writer.println("TTS_SERVICE_EXISTS=${ttsService != null}")
        writer.println("TTS_MODEL_PATH=${initializedTtsModelPath ?: ""}")
    }

    private fun handleAsr(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            doAsrUsage(writer)
            return
        }

        when (args[0]) {
            "status" -> handleAsrStatus(writer)
            else -> doAsrUsage(writer)
        }
    }

    private fun handleAsrStatus(writer: PrintStream) {
        val context = getContext()
        writer.println("=== ASR Service Status ===")
        writer.println("ASR_MODEL_PATH=${VoiceModelPathUtils.getAsrModelPath(context)}")
        // Note: ASR requires microphone and real audio input, so we only check model path
        // Full ASR testing would require audio file input or simulated audio
        writer.println("ASR_TEST_NOTE=ASR requires audio input. Use UI smoke test for full ASR validation.")
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp voice <command> [args]")
        writer.println()
        writer.println("Commands:")
        writer.println("  status                        Check voice models status")
        writer.println("  tts <subcommand>              TTS operations")
        writer.println("  asr <subcommand>              ASR operations")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp voice status")
        writer.println("  dumpapp voice tts init [modelPath]")
        writer.println("  dumpapp voice tts test \"Hello world\"")
    }

    private fun doTtsUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp voice tts <subcommand>")
        writer.println()
        writer.println("Subcommands:")
        writer.println("  init                          Initialize TTS service")
        writer.println("  test [text]                   Test TTS with given text")
        writer.println("  destroy                       Destroy TTS service")
        writer.println("  status                        Show TTS service status")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp voice tts init [modelPath]")
        writer.println("  dumpapp voice tts test \"Hello world\"")
        writer.println("  dumpapp voice tts destroy")
    }

    private fun doAsrUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp voice asr <subcommand>")
        writer.println()
        writer.println("Subcommands:")
        writer.println("  status                        Show ASR model path and status")
        writer.println()
        writer.println("Note: Full ASR testing requires audio input. Use UI smoke test for ASR validation.")
    }
}

internal interface TtsAudioArtifactWriter {
    fun write(context: Context, audioData: ShortArray, sampleRate: Int): String
}

internal class WavTtsAudioArtifactWriter : TtsAudioArtifactWriter {
    override fun write(context: Context, audioData: ShortArray, sampleRate: Int): String {
        val artifactDir = File(context.cacheDir, "tts-dumpapp-artifacts")
        if (!artifactDir.exists() && !artifactDir.mkdirs()) {
            throw IllegalStateException("Failed to create artifact directory: ${artifactDir.absolutePath}")
        }
        val artifactFile = File(artifactDir, "tts_${System.currentTimeMillis()}_${sampleRate}hz.wav")
        val wavWriter = WavFileWriter(
            filePath = artifactFile.absolutePath,
            sampleRate = sampleRate,
            channels = 1,
            bitsPerSample = 16
        )
        val floatAudio = FloatArray(audioData.size) { index -> audioData[index] / 32768.0f }
        wavWriter.addAudioChunk(floatAudio)
        if (!wavWriter.writeToFile()) {
            throw IllegalStateException("Failed to write wav artifact: ${artifactFile.absolutePath}")
        }
        return artifactFile.absolutePath
    }
}

internal interface TtsClient {
    suspend fun init(modelDir: String): Boolean
    suspend fun waitForInitComplete(): Boolean
    fun setLanguage(language: String)
    fun process(text: String, id: Int): ShortArray
    fun destroy()
}

internal class RealTtsClient(
    private val service: TtsService
) : TtsClient {
    override suspend fun init(modelDir: String): Boolean = service.init(modelDir)

    override suspend fun waitForInitComplete(): Boolean = service.waitForInitComplete()

    override fun setLanguage(language: String) {
        service.setLanguage(language)
    }

    override fun process(text: String, id: Int): ShortArray = service.process(text, id)

    override fun destroy() {
        service.destroy()
    }
}

internal data class TtsRoundTripResult(
    val recognizedText: String,
    val hasChinese: Boolean,
    val similarity: Double,
    val status: String,
    val error: String? = null
)

internal interface TtsRoundTripEvaluator {
    fun evaluate(context: Context, audioData: ShortArray, sampleRate: Int, originalText: String): TtsRoundTripResult
}

internal class RealTtsRoundTripEvaluator : TtsRoundTripEvaluator {
    override fun evaluate(
        context: Context,
        audioData: ShortArray,
        sampleRate: Int,
        originalText: String
    ): TtsRoundTripResult {
        return try {
            val modelDir = VoiceModelPathUtils.getAsrModelPath(context)
            val modelConfig = getModelConfigFromDirectory(modelDir)
                ?: return TtsRoundTripResult("", false, 0.0, "FAIL", "ASR model config missing")
            val recognizer = OnlineRecognizer(
                config = OnlineRecognizerConfig(
                    featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
                    modelConfig = modelConfig,
                    lmConfig = getOnlineLMConfigFromDirectory(modelDir),
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
            )

            val recognizedText = recognizer.createStream().let { stream ->
                try {
                    val samples = FloatArray(audioData.size) { index -> audioData[index] / 32768.0f }
                    stream.acceptWaveform(samples, sampleRate)
                    while (recognizer.isReady(stream)) {
                        recognizer.decode(stream)
                    }
                    val tailPaddings = FloatArray((sampleRate * 0.5f).toInt())
                    stream.acceptWaveform(tailPaddings, sampleRate)
                    stream.inputFinished()
                    while (recognizer.isReady(stream)) {
                        recognizer.decode(stream)
                    }
                    recognizer.getResult(stream).text
                } finally {
                    stream.release()
                    recognizer.release()
                }
            }

            val hasChinese = recognizedText.any { it.code in 0x4E00..0x9FFF }
            val similarity = normalizedSimilarity(originalText, recognizedText)
            val status = if (hasChinese && similarity >= 0.5) "PASS" else "FAIL"
            TtsRoundTripResult(recognizedText, hasChinese, similarity, status)
        } catch (e: Exception) {
            TtsRoundTripResult("", false, 0.0, "FAIL", e.message ?: e.javaClass.simpleName)
        }
    }

    private fun normalizedSimilarity(expected: String, actual: String): Double {
        val normalizedExpected = normalizeText(expected)
        val normalizedActual = normalizeText(actual)
        if (normalizedExpected.isEmpty() || normalizedActual.isEmpty()) {
            return 0.0
        }
        val distance = levenshtein(normalizedExpected, normalizedActual)
        val maxLength = max(normalizedExpected.length, normalizedActual.length)
        return 1.0 - (distance.toDouble() / maxLength.toDouble())
    }

    private fun normalizeText(text: String): String {
        return text.replace(Regex("[\\p{Punct}\\p{IsPunctuation}\\s]"), "")
    }

    private fun levenshtein(left: String, right: String): Int {
        if (left == right) return 0
        if (left.isEmpty()) return right.length
        if (right.isEmpty()) return left.length

        val prev = IntArray(right.length + 1) { it }
        val curr = IntArray(right.length + 1)

        for (i in left.indices) {
            curr[0] = i + 1
            for (j in right.indices) {
                val cost = if (left[i] == right[j]) 0 else 1
                curr[j + 1] = minOf(
                    curr[j] + 1,
                    prev[j + 1] + 1,
                    prev[j] + cost
                )
            }
            for (j in prev.indices) {
                prev[j] = curr[j]
            }
        }

        return prev[right.length]
    }
}
