package com.k2fsa.sherpa.mnn

fun main() {
  testOnlineAsr("transducer")
  testOnlineAsr("zipformer2-ctc")
  testOnlineAsr("ctc-hlg")
  testOnlineAsr("nemo-ctc")
}

fun testOnlineAsr(type: String) {
    val featConfig = FeatureConfig(
        sampleRate = 16000,
        featureDim = 80,
    )

    var ctcFstDecoderConfig  = OnlineCtcFstDecoderConfig()
    val waveFilename: String
    val modelConfig: OnlineModelConfig = when (type) {
      "transducer" -> {
        waveFilename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/0.wav"
        // please refer to
        // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
        // to dowload pre-trained models
        OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx",
                decoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx",
                joiner = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx",
            ),
            tokens = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt",
            numThreads = 1,
            debug = false,
        )
      }
      "zipformer2-ctc" -> {
        waveFilename = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav"
        OnlineModelConfig(
            zipformer2Ctc = OnlineZipformer2CtcModelConfig(
                model = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx",
            ),
            tokens = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt",
            numThreads = 1,
            debug = false,
        )
      }
      "nemo-ctc" -> {
        waveFilename = "./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/test_wavs/0.wav"
        OnlineModelConfig(
            neMoCtc = OnlineNeMoCtcModelConfig(
                model = "./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/model.onnx",
            ),
            tokens = "./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/tokens.txt",
            numThreads = 1,
            debug = false,
        )
      }
      "ctc-hlg" -> {
        waveFilename = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/1.wav"
        ctcFstDecoderConfig.graph = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst"
        OnlineModelConfig(
            zipformer2Ctc = OnlineZipformer2CtcModelConfig(
                model = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx",
            ),
            tokens = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt",
            numThreads = 1,
            debug = false,
        )
      }
      else -> throw IllegalArgumentException(type)
    }

    val endpointConfig = EndpointConfig()

    val lmConfig = OnlineLMConfig()

    val config = OnlineRecognizerConfig(
        modelConfig = modelConfig,
        lmConfig = lmConfig,
        featConfig = featConfig,
        ctcFstDecoderConfig=ctcFstDecoderConfig,
        endpointConfig = endpointConfig,
        enableEndpoint = true,
        decodingMethod = "greedy_search",
        maxActivePaths = 4,
    )

    val recognizer = OnlineRecognizer(
        config = config,
    )

    val objArray = WaveReader.readWaveFromFile(
        filename = waveFilename,
    )
    val samples: FloatArray = objArray[0] as FloatArray
    val sampleRate: Int = objArray[1] as Int

    val stream = recognizer.createStream()
    stream.acceptWaveform(samples, sampleRate = sampleRate)
    while (recognizer.isReady(stream)) {
        recognizer.decode(stream)
    }

    val tailPaddings = FloatArray((sampleRate * 0.5).toInt()) // 0.5 seconds
    stream.acceptWaveform(tailPaddings, sampleRate = sampleRate)
    stream.inputFinished()
    while (recognizer.isReady(stream)) {
        recognizer.decode(stream)
    }

    println("results: ${recognizer.getResult(stream).text}")

    stream.release()
    recognizer.release()
}
