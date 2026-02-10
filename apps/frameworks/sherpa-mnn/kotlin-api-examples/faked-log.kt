package android.util

class Log {
  companion object {
    fun i(tag: String, msg: String) {
      println("$tag, $msg")
    }
  }
}

