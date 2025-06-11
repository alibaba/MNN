package com.k2fsa.sherpa.mnn

fun main() {
  testSpeakerRecognition()
}

fun testSpeakerRecognition() {
    val config = SpeakerEmbeddingExtractorConfig(
        model="./3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
        )
    val extractor = SpeakerEmbeddingExtractor(config = config)

    val embedding1a = computeEmbedding(extractor, "./speaker1_a_cn_16k.wav")
    val embedding2a = computeEmbedding(extractor, "./speaker2_a_cn_16k.wav")
    val embedding1b = computeEmbedding(extractor, "./speaker1_b_cn_16k.wav")

    var manager = SpeakerEmbeddingManager(extractor.dim())
    var ok = manager.add(name = "speaker1", embedding=embedding1a)
    check(ok)

    manager.add(name = "speaker2", embedding=embedding2a)
    check(ok)

    var name = manager.search(embedding=embedding1b, threshold=0.5f)
    check(name == "speaker1")

    manager.release()

    manager = SpeakerEmbeddingManager(extractor.dim())
    val embeddingList = mutableListOf(embedding1a, embedding1b)
    ok = manager.add(name = "s1", embedding=embeddingList.toTypedArray())
    check(ok)

    name = manager.search(embedding=embedding1b, threshold=0.5f)
    check(name == "s1")

    name = manager.search(embedding=embedding2a, threshold=0.5f)
    check(name.length == 0)

    manager.release()
    extractor.release()
    println("Speaker ID test done!")
}

fun computeEmbedding(extractor: SpeakerEmbeddingExtractor, filename: String): FloatArray {
    var objArray = WaveReader.readWaveFromFile(
        filename = filename,
    )
    var samples: FloatArray = objArray[0] as FloatArray
    var sampleRate: Int = objArray[1] as Int

    val stream = extractor.createStream()
    stream.acceptWaveform(sampleRate = sampleRate, samples=samples)
    stream.inputFinished()
    check(extractor.isReady(stream))

    val embedding = extractor.compute(stream)

    stream.release()

    return embedding
}
