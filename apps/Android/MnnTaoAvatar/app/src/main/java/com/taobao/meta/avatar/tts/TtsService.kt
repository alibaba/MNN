package com.taobao.meta.avatar.tts

import android.util.Log
import com.k2fsa.sherpa.mnn.GeneratedAudio
import com.taobao.meta.avatar.debug.DebugModule
import com.taobao.meta.avatar.utils.AppUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async

class TtsService {

    val useSherpaTts
        get() = DebugModule.TTS_USE_SHERPA

    private var sherpaTts: SherpaTts? = null
    private var ttsServiceNative: Long = 0
    @Volatile
    private var isLoaded = false
    private var initDeferred: Deferred<Boolean>? = null

    init {
        ttsServiceNative = nativeCreateTTS(if (AppUtils.isChinese())  "zh" else "en")
    }

    fun destroy() {
        nativeDestroy(ttsServiceNative)
        ttsServiceNative = 0
    }

    suspend fun init(modelDir: String?): Boolean {
        if (isLoaded) return true
        if (initDeferred == null) {
            initDeferred = CoroutineScope(Dispatchers.IO).async {
                if (useSherpaTts) {
                    sherpaTts = SherpaTts()
                    sherpaTts?.init(null)
                    return@async true
                }
                nativeLoadResourcesFromFile(ttsServiceNative,
                    modelDir!!,
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

    fun processSherpa(text: String, id: Int): GeneratedAudio? {
        Log.d(TAG, "processSherpa: $text $id")
        synchronized(this) {
            return sherpaTts?.process(text)
        }
    }

    // Native methods
    private external fun nativeSetCurrentIndex(ttsServiceNative: Long, index: Int);
    private external fun nativeCreateTTS(language:String): Long
    private external fun nativeDestroy(nativePtr: Long)
    private external fun nativeLoadResourcesFromFile(nativePtr: Long,
                                                     resourceDir: String,
                                                     modelName:String,
                                                     mmapDir:String): Boolean
    private external fun nativeProcess(nativePtr: Long, text: String, id: Int): ShortArray

    companion object {
        private const val TAG = "TtsService"
    }

}
