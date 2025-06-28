package com.k2fsa.sherpa.mnn

fun main() {
  testAudioTagging()
}

fun testAudioTagging() {
  val config = AudioTaggingConfig(
      model=AudioTaggingModelConfig(
        zipformer=OfflineZipformerAudioTaggingModelConfig(
          model="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.int8.onnx",
        ),
        numThreads=1,
        debug=true,
        provider="cpu",
      ),
      labels="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv",
      topK=5,
   )
  val tagger = AudioTagging(config=config)

  val testFiles = arrayOf(
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/1.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/2.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/3.wav",
    "./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/4.wav",
  )
  println("----------")
  for (waveFilename in testFiles) {
    val stream = tagger.createStream()

    val objArray = WaveReader.readWaveFromFile(
        filename = waveFilename,
    )
    val samples: FloatArray = objArray[0] as FloatArray
    val sampleRate: Int = objArray[1] as Int

    stream.acceptWaveform(samples, sampleRate = sampleRate)
    val events = tagger.compute(stream)
    stream.release()

    println(waveFilename)
    println(events)
    println("----------")
  }

  tagger.release()
}

