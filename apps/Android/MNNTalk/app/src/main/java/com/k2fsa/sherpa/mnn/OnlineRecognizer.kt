package com.k2fsa.sherpa.mnn

import android.content.res.AssetManager

data class EndpointRule(
    var mustContainNonSilence: Boolean,
    var minTrailingSilence: Float,
    var minUtteranceLength: Float
)

data class EndpointConfig(
    var rule1: EndpointRule = EndpointRule(false, 2.4f, 0.0f),
    var rule2: EndpointRule = EndpointRule(true, 1.4f, 0.0f),
    var rule3: EndpointRule = EndpointRule(false, 0.0f, 20.0f)
)

data class OnlineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = ""
)

data class OnlineParaformerModelConfig(
    var encoder: String = "",
    var decoder: String = ""
)

data class OnlineZipformer2CtcModelConfig(var model: String = "")

data class OnlineNeMoCtcModelConfig(var model: String = "")

data class OnlineModelConfig(
    var transducer: OnlineTransducerModelConfig = OnlineTransducerModelConfig(),
    var paraformer: OnlineParaformerModelConfig = OnlineParaformerModelConfig(),
    var zipformer2Ctc: OnlineZipformer2CtcModelConfig = OnlineZipformer2CtcModelConfig(),
    var neMoCtc: OnlineNeMoCtcModelConfig = OnlineNeMoCtcModelConfig(),
    var tokens: String = "",
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
    var modelType: String = "",
    var modelingUnit: String = "",
    var bpeVocab: String = ""
)

data class OnlineLMConfig(
    var model: String = "",
    var scale: Float = 0.5f
)

data class OnlineCtcFstDecoderConfig(
    var graph: String = "",
    var maxActive: Int = 3000
)

data class OnlineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OnlineModelConfig = OnlineModelConfig(),
    var lmConfig: OnlineLMConfig = OnlineLMConfig(),
    var ctcFstDecoderConfig: OnlineCtcFstDecoderConfig = OnlineCtcFstDecoderConfig(),
    var endpointConfig: EndpointConfig = EndpointConfig(),
    var enableEndpoint: Boolean = true,
    var decodingMethod: String = "greedy_search",
    var maxActivePaths: Int = 4,
    var hotwordsFile: String = "",
    var hotwordsScore: Float = 1.5f,
    var ruleFsts: String = "",
    var ruleFars: String = "",
    var blankPenalty: Float = 0.0f
)

data class OnlineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray
)

class OnlineRecognizer(
    assetManager: AssetManager? = null,
    val config: OnlineRecognizerConfig
) {
    private var ptr: Long = if (assetManager == null) {
        newFromFile(config)
    } else {
        newFromAsset(assetManager, config)
    }

    fun createStream(hotwords: String = ""): OnlineStream {
        return OnlineStream(createStream(ptr, hotwords))
    }

    fun reset(stream: OnlineStream) = reset(ptr, stream.ptr)

    fun decode(stream: OnlineStream) = decode(ptr, stream.ptr)

    fun isEndpoint(stream: OnlineStream) = isEndpoint(ptr, stream.ptr)

    fun isReady(stream: OnlineStream) = isReady(ptr, stream.ptr)

    fun getResult(stream: OnlineStream): OnlineRecognizerResult {
        val result = getResult(ptr, stream.ptr)
        @Suppress("UNCHECKED_CAST")
        return OnlineRecognizerResult(
            text = result[0] as String,
            tokens = result[1] as Array<String>,
            timestamps = result[2] as FloatArray
        )
    }

    fun release() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    @Suppress("deprecation")
    protected fun finalize() = release()

    private external fun delete(ptr: Long)
    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OnlineRecognizerConfig
    ): Long
    private external fun newFromFile(config: OnlineRecognizerConfig): Long
    private external fun createStream(ptr: Long, hotwords: String): Long
    private external fun reset(ptr: Long, streamPtr: Long)
    private external fun decode(ptr: Long, streamPtr: Long)
    private external fun isEndpoint(ptr: Long, streamPtr: Long): Boolean
    private external fun isReady(ptr: Long, streamPtr: Long): Boolean
    private external fun getResult(ptr: Long, streamPtr: Long): Array<Any>

    companion object {
        init {
            System.loadLibrary("sherpa-mnn-jni")
        }
    }
}

fun getEndpointConfig(): EndpointConfig = EndpointConfig()
