package com.k2fsa.sherpa.mnn

import android.content.res.AssetManager
import android.util.Log

data class EndpointRule(
    var mustContainNonSilence: Boolean,
    var minTrailingSilence: Float,
    var minUtteranceLength: Float,
)

data class EndpointConfig(
    var rule1: EndpointRule = EndpointRule(false, 2.4f, 0.0f),
    var rule2: EndpointRule = EndpointRule(true, 1.4f, 0.0f),
    var rule3: EndpointRule = EndpointRule(false, 0.0f, 20.0f)
)

data class OnlineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = "",
)

data class OnlineParaformerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
)

data class OnlineZipformer2CtcModelConfig(
    var model: String = "",
)

data class OnlineNeMoCtcModelConfig(
    var model: String = "",
)

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
    var bpeVocab: String = "",
)

data class OnlineLMConfig(
    var model: String = "",
    var scale: Float = 0.5f,
)

data class OnlineCtcFstDecoderConfig(
    var graph: String = "",
    var maxActive: Int = 3000,
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
    var blankPenalty: Float = 0.0f,
)

data class OnlineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray,
    // TODO(fangjun): Add more fields
)

class OnlineRecognizer(
    assetManager: AssetManager? = null,
    val config: OnlineRecognizerConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    fun createStream(hotwords: String = ""): OnlineStream {
        val p = createStream(ptr, hotwords)
        return OnlineStream(p)
    }

    fun reset(stream: OnlineStream) = reset(ptr, stream.ptr)
    fun decode(stream: OnlineStream) = decode(ptr, stream.ptr)
    fun isEndpoint(stream: OnlineStream) = isEndpoint(ptr, stream.ptr)
    fun isReady(stream: OnlineStream) = isReady(ptr, stream.ptr)
    fun getResult(stream: OnlineStream): OnlineRecognizerResult {
        val objArray = getResult(ptr, stream.ptr)

        val text = objArray[0] as String
        val tokens = objArray[1] as Array<String>
        val timestamps = objArray[2] as FloatArray

        return OnlineRecognizerResult(text = text, tokens = tokens, timestamps = timestamps)
    }

    private external fun delete(ptr: Long)

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OnlineRecognizerConfig,
    ): Long

    private external fun newFromFile(
        config: OnlineRecognizerConfig,
    ): Long

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

private var ASR_MODEL_DIR = "/data/local/tmp/asr_models"

/**
 * 设置ASR模型目录
 * @param modelDir 模型目录路径
 */
fun setAsrModelDir(modelDir: String) {
    Log.d("OnlineRecognizer", "Setting ASR model directory to: $modelDir")
    ASR_MODEL_DIR = modelDir
}

/**
 * 获取当前ASR模型目录
 * @return 当前模型目录路径
 */
fun getAsrModelDir(): String {
    return ASR_MODEL_DIR
}

/**
 * 根据配置类型获取模型配置 (已弃用，建议使用 getModelConfigFromDirectory)
 * @param type 配置类型 (0=中英双语, 1=英语)
 * @return OnlineModelConfig 或 null
 */
@Deprecated("Use getModelConfigFromDirectory() instead for better flexibility")
fun getModelConfig(type: Int): OnlineModelConfig? {
    Log.w("OnlineRecognizer", "getModelConfig(type) is deprecated, consider using getModelConfigFromDirectory()")
    
    // 使用传统的目录结构作为回退
    val modelDir = when (type) {
        0 -> "${ASR_MODEL_DIR}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20"
        1 -> "${ASR_MODEL_DIR}/sherpa-mnn-streaming-zipformer-en-2023-02-21"
        else -> ASR_MODEL_DIR
    }
    
    return AsrConfigManager.getModelConfigFromDirectory(modelDir)
}

/**
 * 从指定模型目录获取模型配置（推荐使用）
 * @param modelDir 完整的模型目录路径 (如: /path/to/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20)
 * @return OnlineModelConfig 或 null
 */
fun getModelConfigFromDirectory(modelDir: String): OnlineModelConfig? {
    Log.d("OnlineRecognizer", "Getting model config from directory: $modelDir")
    return AsrConfigManager.getModelConfigFromDirectory(modelDir)
}

/*
Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own LM model. (It should be straightforward to train a new NN LM model
by following the code, https://github.com/k2-fsa/icefall/blob/master/icefall/rnn_lm/train.py)

@param type
0 - sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
 */
/**
 * 根据配置类型获取语言模型配置 (已弃用，建议使用 getOnlineLMConfigFromDirectory)
 * @param type 配置类型 (0=使用LM, 其他=不使用LM)
 * @return OnlineLMConfig
 */
@Deprecated("Use getOnlineLMConfigFromDirectory() instead for better flexibility")
fun getOnlineLMConfig(type: Int): OnlineLMConfig {
    Log.w("OnlineRecognizer", "getOnlineLMConfig(type) is deprecated, consider using getOnlineLMConfigFromDirectory()")
    
    val modelDir = when (type) {
        0 -> "${ASR_MODEL_DIR}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20"
        1 -> "${ASR_MODEL_DIR}/sherpa-mnn-streaming-zipformer-en-2023-02-21"
        else -> ASR_MODEL_DIR
    }
    
    return AsrConfigManager.getLmConfigFromDirectory(modelDir)
}

/**
 * 从指定模型目录获取语言模型配置（推荐使用）
 * @param modelDir 完整的模型目录路径 (如: /path/to/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20)
 * @return OnlineLMConfig
 */
fun getOnlineLMConfigFromDirectory(modelDir: String): OnlineLMConfig {
    Log.d("OnlineRecognizer", "Getting LM config from directory: $modelDir")
    return AsrConfigManager.getLmConfigFromDirectory(modelDir)
}

fun getEndpointConfig(): EndpointConfig {
    return EndpointConfig(
        rule1 = EndpointRule(false, 2.4f, 0.0f),
        rule2 = EndpointRule(true, 1.4f, 0.0f),
        rule3 = EndpointRule(false, 0.0f, 20.0f)
    )
}

/*
使用示例：

// 1. 设置自定义模型目录
setAsrModelDir("/path/to/your/models")

// 2. 获取当前模型目录
val currentDir = getAsrModelDir()

// 3. 在 AsrService 中使用
val asrService = AsrService(activity, "/path/to/your/models")

// 4. 或者使用默认目录
val asrService = AsrService(activity) // 使用默认的 /data/local/tmp/asr_models
*/

