package com.k2fsa.sherpa.mnn

fun main() {
  testOfflineSpeakerDiarization()
}

fun callback(numProcessedChunks: Int, numTotalChunks: Int, arg: Long): Int {
  val progress = numProcessedChunks.toFloat() / numTotalChunks * 100
  val s = "%.2f".format(progress)
  println("Progress: ${s}%");

  return 0
}

fun testOfflineSpeakerDiarization() {
  var config = OfflineSpeakerDiarizationConfig(
    segmentation=OfflineSpeakerSegmentationModelConfig(
      pyannote=OfflineSpeakerSegmentationPyannoteModelConfig("./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    ),
    embedding=SpeakerEmbeddingExtractorConfig(
      model="./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    ),

    // The test wave file ./0-four-speakers-zh.wav contains four speakers, so
    // we use numClusters=4 here. If you don't know the number of speakers
    // in the test wave file, please set the threshold like below.
    //
    // clustering=FastClusteringConfig(threshold=0.5),
    //
    // WARNING: You need to tune threshold by yourself.
    // A larger threshold leads to fewer clusters, i.e., few speakers.
    // A smaller threshold leads to more clusters, i.e., more speakers.
    //
    clustering=FastClusteringConfig(numClusters=4),
  )

  val sd = OfflineSpeakerDiarization(config=config)

  val waveData = WaveReader.readWave(
      filename = "./0-four-speakers-zh.wav",
  )

  if (sd.sampleRate() != waveData.sampleRate) {
    println("Expected sample rate: ${sd.sampleRate()}, given: ${waveData.sampleRate}")
    return
  }

  // val segments = sd.process(waveData.samples) // this one is also ok
  val segments = sd.processWithCallback(waveData.samples, callback=::callback)
  for (segment in segments) {
    println("${segment.start} -- ${segment.end} speaker_${segment.speaker}")
  }
}
