package com.k2fsa.sherpa.mnn

fun main() {
  testSpokenLanguageIdentifcation()
}

fun testSpokenLanguageIdentifcation() {
  val config = SpokenLanguageIdentificationConfig(
    whisper = SpokenLanguageIdentificationWhisperConfig(
      encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx",
      decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx",
      tailPaddings = 33,
    ),
    numThreads=1,
    debug=true,
    provider="cpu",
  )
  val slid = SpokenLanguageIdentification(config=config)

  val testFiles = arrayOf(
    "./spoken-language-identification-test-wavs/ar-arabic.wav",
    "./spoken-language-identification-test-wavs/bg-bulgarian.wav",
    "./spoken-language-identification-test-wavs/de-german.wav",
  )

  for (waveFilename in testFiles) {
    val objArray = WaveReader.readWaveFromFile(
        filename = waveFilename,
    )
    val samples: FloatArray = objArray[0] as FloatArray
    val sampleRate: Int = objArray[1] as Int

    val stream = slid.createStream()
    stream.acceptWaveform(samples, sampleRate = sampleRate)
    val lang = slid.compute(stream)
    stream.release()
    println(waveFilename)
    println(lang)
  }

  slid.release()
}

