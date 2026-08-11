package com.alibaba.mnn.tts.demo

import com.k2fsa.sherpa.mnn.OnlineCtcFstDecoderConfig
import com.k2fsa.sherpa.mnn.OnlineRecognizer
import com.k2fsa.sherpa.mnn.OnlineRecognizerConfig
import com.k2fsa.sherpa.mnn.getEndpointConfig
import com.k2fsa.sherpa.mnn.getFeatureConfig
import com.k2fsa.sherpa.mnn.getModelConfigFromDirectory
import com.k2fsa.sherpa.mnn.getOnlineLMConfigFromDirectory
import kotlin.math.max

data class TtsRoundTripResult(
    val recognizedText: String,
    val hasChinese: Boolean,
    val similarity: Double,
    val status: String,
    val error: String? = null
)

object TtsRoundTripScorer {
    private const val PASS_THRESHOLD = 0.5

    fun score(expected: String, actual: String): TtsRoundTripResult {
        val hasChinese = actual.any { it.code in 0x4E00..0x9FFF }
        val similarity = normalizedSimilarity(expected, actual)
        val status = if (hasChinese && similarity >= PASS_THRESHOLD) "PASS" else "FAIL"
        return TtsRoundTripResult(
            recognizedText = actual,
            hasChinese = hasChinese,
            similarity = similarity,
            status = status
        )
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

        val previous = IntArray(right.length + 1) { it }
        val current = IntArray(right.length + 1)

        for (i in left.indices) {
            current[0] = i + 1
            for (j in right.indices) {
                val cost = if (left[i] == right[j]) 0 else 1
                current[j + 1] = minOf(
                    current[j] + 1,
                    previous[j + 1] + 1,
                    previous[j] + cost
                )
            }
            for (j in previous.indices) {
                previous[j] = current[j]
            }
        }

        return previous[right.length]
    }
}

class TtsRoundTripEvaluator(
    private val asrModelDir: String = DEFAULT_ASR_MODEL_DIR
) {
    fun evaluate(audioData: ShortArray, sampleRate: Int, originalText: String): TtsRoundTripResult {
        return try {
            val modelConfig = getModelConfigFromDirectory(asrModelDir)
                ?: return TtsRoundTripResult("", false, 0.0, "FAIL", "ASR model config missing")
            val recognizer = OnlineRecognizer(
                config = OnlineRecognizerConfig(
                    featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
                    modelConfig = modelConfig,
                    lmConfig = getOnlineLMConfigFromDirectory(asrModelDir),
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

            TtsRoundTripScorer.score(expected = originalText, actual = recognizedText)
        } catch (t: Throwable) {
            TtsRoundTripResult("", false, 0.0, "FAIL", t.message ?: t.javaClass.simpleName)
        }
    }

    companion object {
        const val DEFAULT_ASR_MODEL_DIR =
            "/data/local/tmp/asr_models/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20"
    }
}
