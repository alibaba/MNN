package com.k2fsa.sherpa.mnn

fun main() {
  test()
}

fun test() {
  val recognizer = createOnlineRecognizer()
  val waveFilename = "./itn-zh-number.wav";

  val objArray = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )
  val samples: FloatArray = objArray[0] as FloatArray
  val sampleRate: Int = objArray[1] as Int

  val stream = recognizer.createStream()
  stream.acceptWaveform(samples, sampleRate=sampleRate)
  while (recognizer.isReady(stream)) {
    recognizer.decode(stream)
  }

  val result = recognizer.getResult(stream).text
  println(result)

  stream.release()
  recognizer.release()
}

fun createOnlineRecognizer(): OnlineRecognizer {
  val config = OnlineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getModelConfig(8)!!,
  )

  config.ruleFsts = "./itn_zh_number.fst"
  println(config)

  return OnlineRecognizer(config = config)
}

