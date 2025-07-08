package com.taobao.meta.avatar.tts

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async

class TtsService {
    private var ttsServiceNative: Long = 0
    @Volatile
    private var isLoaded = false
    private var initDeferred: Deferred<Boolean>? = null
    private var currentLanguage: String = DEFAULT_LANGUAGE

    init {
        ttsServiceNative = nativeCreateTTS(currentLanguage)
    }

    fun destroy() {
        nativeDestroy(ttsServiceNative)
        ttsServiceNative = 0
    }

    suspend fun init(modelDir: String): Boolean {
        if (isLoaded) return true
        if (initDeferred == null) {
            initDeferred = CoroutineScope(Dispatchers.IO).async {
                nativeLoadResourcesFromFile(ttsServiceNative,
                    modelDir,
                    "",
                    "")
                true
            }
        }
        val result = initDeferred!!.await()
        if (result) {
            isLoaded = true
        }
        return result
    }

    suspend fun waitForInitComplete(): Boolean {
        if (isLoaded) return true
        initDeferred?.let {
            return it.await()
        }
        return isLoaded
    }

    fun setCurrentIndex(index: Int) {
        nativeSetCurrentIndex(ttsServiceNative, index)
    }

    fun process(text: String, id: Int): ShortArray {
        return nativeProcess(ttsServiceNative, text, id)
    }

    fun setLanguage(language: String) {
        if (currentLanguage != language) {
            currentLanguage = language
            if (ttsServiceNative != 0L) {
                destroy()
            }
            ttsServiceNative = nativeCreateTTS(language)
        }
    }

    // Native methods
    private external fun nativeSetCurrentIndex(ttsServiceNative: Long, index: Int)
    private external fun nativeCreateTTS(language:String): Long
    private external fun nativeDestroy(nativePtr: Long)
    private external fun nativeLoadResourcesFromFile(nativePtr: Long,
                                                   resourceDir: String,
                                                   modelName:String,
                                                   mmapDir:String): Boolean
    private external fun nativeProcess(nativePtr: Long, text: String, id: Int): ShortArray

    companion object {
        private const val TAG = "TtsService"
        private const val LIBRARY_NAME = "mnn_tts"
        private const val DEFAULT_LANGUAGE = "en"
        
        init {
            System.loadLibrary(LIBRARY_NAME)
            Log.d(TAG, "mnn_tts native library loaded successfully")
        }
    }
}