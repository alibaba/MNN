package com.k2fsa.sherpa.mnn

fun main() {
  testPunctuation()
}

fun testPunctuation() {
  val config = OfflinePunctuationConfig(
      model=OfflinePunctuationModelConfig(
          ctTransformer="./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx",
          numThreads=1,
          debug=true,
          provider="cpu",
      )
  )
  val punct = OfflinePunctuation(config = config)
  val sentences = arrayOf(
        "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
        "我们都是木头人不会说话不会动",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
  )
  println("---")
  for (text in sentences) {
    val out = punct.addPunctuation(text)
    println("Input: $text")
    println("Output: $out")
    println("---")
  }
  println(sentences)

}
