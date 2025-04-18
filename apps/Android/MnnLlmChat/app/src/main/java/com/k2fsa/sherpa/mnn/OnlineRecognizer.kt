package com.k2fsa.sherpa.mnn

import android.content.res.AssetManager

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


/*
Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own. (It should be straightforward to add a new model
by following the code)

@param type
0 - sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

1 - csukuangfj/sherpa-onnx-lstm-zh-2023-02-20 (Chinese)

    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/lstm-transducer-models.html#csukuangfj-sherpa-onnx-lstm-zh-2023-02-20-chinese

2 - csukuangfj/sherpa-onnx-lstm-en-2023-02-17 (English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/lstm-transducer-models.html#csukuangfj-sherpa-onnx-lstm-en-2023-02-17-english

3,4 - pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
    https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
    3 - int8 encoder
    4 - float32 encoder

5 - csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en
    https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en

6 - sherpa-onnx-streaming-zipformer-en-2023-06-26
    https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26

7 - shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14 (French)
    https://huggingface.co/shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14

8 - csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    encoder int8, decoder/joiner float32

 */
fun getModelConfig(type: Int): OnlineModelConfig? {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        1 -> {
            val modelDir = "sherpa-onnx-lstm-zh-2023-02-20"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-11-avg-1.onnx",
                    decoder = "$modelDir/decoder-epoch-11-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-11-avg-1.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "lstm",
            )
        }

        2 -> {
            val modelDir = "sherpa-onnx-lstm-en-2023-02-17"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "lstm",
            )
        }

        3 -> {
            val modelDir = "icefall-asr-zipformer-streaming-wenetspeech-20230615"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/exp/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx",
                    decoder = "$modelDir/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx",
                    joiner = "$modelDir/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx",
                ),
                tokens = "$modelDir/data/lang_char/tokens.txt",
                modelType = "zipformer2",
            )
        }

        4 -> {
            val modelDir = "icefall-asr-zipformer-streaming-wenetspeech-20230615"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx",
                    decoder = "$modelDir/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx",
                    joiner = "$modelDir/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx",
                ),
                tokens = "$modelDir/data/lang_char/tokens.txt",
                modelType = "zipformer2",
            )
        }

        5 -> {
            val modelDir = "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
            return OnlineModelConfig(
                paraformer = OnlineParaformerModelConfig(
                    encoder = "$modelDir/encoder.int8.onnx",
                    decoder = "$modelDir/decoder.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "paraformer",
            )
        }

        6 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer2",
            )
        }

        7 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-fr-2023-04-14"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-29-avg-9-with-averaged-model.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-29-avg-9-with-averaged-model.onnx",
                    joiner = "$modelDir/joiner-epoch-29-avg-9-with-averaged-model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        8 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        9 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        10 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        11 -> {
            val modelDir = "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms"
            return OnlineModelConfig(
                neMoCtc = OnlineNeMoCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        12 -> {
            val modelDir = "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms"
            return OnlineModelConfig(
                neMoCtc = OnlineNeMoCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        13 -> {
            val modelDir = "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-1040ms"
            return OnlineModelConfig(
                neMoCtc = OnlineNeMoCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        14 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-korean-2024-06-16"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }
    }
    return null
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
fun getOnlineLMConfig(type: Int): OnlineLMConfig {
    when (type) {
        0 -> {
            val modelDir = "ASR_MODEL_DIR"
            return OnlineLMConfig(
                model = "$modelDir/with-state-epoch-99-avg-1.int8.onnx",
                scale = 0.5f,
            )
        }
    }
    return OnlineLMConfig()
}

fun getEndpointConfig(): EndpointConfig {
    return EndpointConfig(
        rule1 = EndpointRule(false, 2.4f, 0.0f),
        rule2 = EndpointRule(true, 1.4f, 0.0f),
        rule3 = EndpointRule(false, 0.0f, 20.0f)
    )
}

