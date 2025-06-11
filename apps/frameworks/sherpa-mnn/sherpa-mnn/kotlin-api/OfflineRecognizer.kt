package com.k2fsa.sherpa.mnn

import android.content.res.AssetManager

data class OfflineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray,
    val lang: String,
    val emotion: String,
    val event: String,
)

data class OfflineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = "",
)

data class OfflineParaformerModelConfig(
    var model: String = "",
)

data class OfflineNemoEncDecCtcModelConfig(
    var model: String = "",
)

data class OfflineWhisperModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var language: String = "en", // Used with multilingual model
    var task: String = "transcribe", // transcribe or translate
    var tailPaddings: Int = 1000, // Padding added at the end of the samples
)

data class OfflineFireRedAsrModelConfig(
    var encoder: String = "",
    var decoder: String = "",
)

data class OfflineMoonshineModelConfig(
    var preprocessor: String = "",
    var encoder: String = "",
    var uncachedDecoder: String = "",
    var cachedDecoder: String = "",
)

data class OfflineSenseVoiceModelConfig(
    var model: String = "",
    var language: String = "",
    var useInverseTextNormalization: Boolean = true,
)

data class OfflineModelConfig(
    var transducer: OfflineTransducerModelConfig = OfflineTransducerModelConfig(),
    var paraformer: OfflineParaformerModelConfig = OfflineParaformerModelConfig(),
    var whisper: OfflineWhisperModelConfig = OfflineWhisperModelConfig(),
    var fireRedAsr: OfflineFireRedAsrModelConfig = OfflineFireRedAsrModelConfig(),
    var moonshine: OfflineMoonshineModelConfig = OfflineMoonshineModelConfig(),
    var nemo: OfflineNemoEncDecCtcModelConfig = OfflineNemoEncDecCtcModelConfig(),
    var senseVoice: OfflineSenseVoiceModelConfig = OfflineSenseVoiceModelConfig(),
    var teleSpeech: String = "",
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
    var modelType: String = "",
    var tokens: String = "",
    var modelingUnit: String = "",
    var bpeVocab: String = "",
)

data class OfflineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OfflineModelConfig = OfflineModelConfig(),
    // var lmConfig: OfflineLMConfig(), // TODO(fangjun): enable it
    var decodingMethod: String = "greedy_search",
    var maxActivePaths: Int = 4,
    var hotwordsFile: String = "",
    var hotwordsScore: Float = 1.5f,
    var ruleFsts: String = "",
    var ruleFars: String = "",
    var blankPenalty: Float = 0.0f,
)

class OfflineRecognizer(
    assetManager: AssetManager? = null,
    config: OfflineRecognizerConfig,
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

    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        return OfflineStream(p)
    }

    fun getResult(stream: OfflineStream): OfflineRecognizerResult {
        val objArray = getResult(stream.ptr)

        val text = objArray[0] as String
        val tokens = objArray[1] as Array<String>
        val timestamps = objArray[2] as FloatArray
        val lang = objArray[3] as String
        val emotion = objArray[4] as String
        val event = objArray[5] as String
        return OfflineRecognizerResult(
            text = text,
            tokens = tokens,
            timestamps = timestamps,
            lang = lang,
            emotion = emotion,
            event = event
        )
    }

    fun decode(stream: OfflineStream) = decode(ptr, stream.ptr)

    private external fun delete(ptr: Long)

    private external fun createStream(ptr: Long): Long

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OfflineRecognizerConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineRecognizerConfig,
    ): Long

    private external fun decode(ptr: Long, streamPtr: Long)

    private external fun getResult(streamPtr: Long): Array<Any>

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

0 - csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14 (Chinese)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-09-14-chinese
    int8

1 - icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04 (English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#icefall-asr-multidataset-pruned-transducer-stateless7-2023-05-04-english
    encoder int8, decoder/joiner float32

2 - sherpa-onnx-whisper-tiny.en
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html#tiny-en
    encoder int8, decoder int8

3 - sherpa-onnx-whisper-base.en
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html#tiny-en
    encoder int8, decoder int8

4 - pkufool/icefall-asr-zipformer-wenetspeech-20230615 (Chinese)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#pkufool-icefall-asr-zipformer-wenetspeech-20230615-chinese
    encoder/joiner int8, decoder fp32

 */
fun getOfflineModelConfig(type: Int): OfflineModelConfig? {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-paraformer-zh-2023-09-14"
            return OfflineModelConfig(
                paraformer = OfflineParaformerModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "paraformer",
            )
        }

        1 -> {
            val modelDir = "icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-30-avg-4.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-30-avg-4.onnx",
                    joiner = "$modelDir/joiner-epoch-30-avg-4.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        2 -> {
            val modelDir = "sherpa-onnx-whisper-tiny.en"
            return OfflineModelConfig(
                whisper = OfflineWhisperModelConfig(
                    encoder = "$modelDir/tiny.en-encoder.int8.onnx",
                    decoder = "$modelDir/tiny.en-decoder.int8.onnx",
                ),
                tokens = "$modelDir/tiny.en-tokens.txt",
                modelType = "whisper",
            )
        }

        3 -> {
            val modelDir = "sherpa-onnx-whisper-base.en"
            return OfflineModelConfig(
                whisper = OfflineWhisperModelConfig(
                    encoder = "$modelDir/base.en-encoder.int8.onnx",
                    decoder = "$modelDir/base.en-decoder.int8.onnx",
                ),
                tokens = "$modelDir/base.en-tokens.txt",
                modelType = "whisper",
            )
        }


        4 -> {
            val modelDir = "icefall-asr-zipformer-wenetspeech-20230615"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-12-avg-4.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-12-avg-4.onnx",
                    joiner = "$modelDir/joiner-epoch-12-avg-4.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        5 -> {
            val modelDir = "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-20-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-20-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-20-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        6 -> {
            val modelDir = "sherpa-onnx-nemo-ctc-en-citrinet-512"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        7 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        8 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-en-24500"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        9 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        10 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-es-1424"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        11 -> {
            val modelDir = "sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04"
            return OfflineModelConfig(
                teleSpeech = "$modelDir/model.int8.onnx",
                tokens = "$modelDir/tokens.txt",
                modelType = "telespeech_ctc",
            )
        }

        12 -> {
            val modelDir = "sherpa-onnx-zipformer-thai-2024-06-20"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-12-avg-5.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-12-avg-5.onnx",
                    joiner = "$modelDir/joiner-epoch-12-avg-5.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        13 -> {
            val modelDir = "sherpa-onnx-zipformer-korean-2024-06-24"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        14 -> {
            val modelDir = "sherpa-onnx-paraformer-zh-small-2024-03-09"
            return OfflineModelConfig(
                paraformer = OfflineParaformerModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "paraformer",
            )
        }

        15 -> {
            val modelDir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
            return OfflineModelConfig(
                senseVoice = OfflineSenseVoiceModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        16 -> {
            val modelDir = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        17 -> {
            val modelDir = "sherpa-onnx-zipformer-ru-2024-09-18"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder.int8.onnx",
                    decoder = "$modelDir/decoder.onnx",
                    joiner = "$modelDir/joiner.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        18 -> {
            val modelDir = "sherpa-onnx-small-zipformer-ru-2024-09-18"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder.int8.onnx",
                    decoder = "$modelDir/decoder.onnx",
                    joiner = "$modelDir/joiner.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        19 -> {
            val modelDir = "sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        20 -> {
            val modelDir = "sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder.int8.onnx",
                    decoder = "$modelDir/decoder.onnx",
                    joiner = "$modelDir/joiner.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "nemo_transducer",
            )
        }

        21 -> {
            val modelDir = "sherpa-onnx-moonshine-tiny-en-int8"
            return OfflineModelConfig(
                moonshine = OfflineMoonshineModelConfig(
                    preprocessor = "$modelDir/preprocess.onnx",
                    encoder = "$modelDir/encode.int8.onnx",
                    uncachedDecoder = "$modelDir/uncached_decode.int8.onnx",
                    cachedDecoder = "$modelDir/cached_decode.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        22 -> {
            val modelDir = "sherpa-onnx-moonshine-base-en-int8"
            return OfflineModelConfig(
                moonshine = OfflineMoonshineModelConfig(
                    preprocessor = "$modelDir/preprocess.onnx",
                    encoder = "$modelDir/encode.int8.onnx",
                    uncachedDecoder = "$modelDir/uncached_decode.int8.onnx",
                    cachedDecoder = "$modelDir/cached_decode.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        23 -> {
            val modelDir = "sherpa-onnx-zipformer-zh-en-2023-11-22"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-34-avg-19.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-34-avg-19.onnx",
                    joiner = "$modelDir/joiner-epoch-34-avg-19.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        24 -> {
            val modelDir = "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16"
            return OfflineModelConfig(
                fireRedAsr = OfflineFireRedAsrModelConfig(
                    encoder = "$modelDir/encoder.int8.onnx",
                    decoder = "$modelDir/decoder.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }
    }
    return null
}
