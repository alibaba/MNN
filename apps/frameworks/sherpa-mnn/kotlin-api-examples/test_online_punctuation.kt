package com.k2fsa.sherpa.mnn

fun main() {
  testPunctuation()
}

// https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
fun testPunctuation() {
  val config = OnlinePunctuationConfig(
      model=OnlinePunctuationModelConfig(
          cnnBilstm="./sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx",
          bpeVocab="./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab",
          numThreads=1,
          debug=true,
          provider="cpu",
      )
  )
  val punct = OnlinePunctuation(config = config)
  val sentences = arrayOf(
        "how are you doing fantastic thank you what is about you",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
  )
  println("---")
  for (text in sentences) {
    val out = punct.addPunctuation(text)
    println("Input: $text")
    println("Output: $out")
    println("---")
  }
}
