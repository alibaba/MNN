package com.alibaba.mnn.tts

object MNNTTS {
    @JvmStatic
    external fun getHelloWorldFromJNI(): String
    init {
        System.loadLibrary("mnn_tts")
    }
}

class Native {
    external fun platformFunction()
} 