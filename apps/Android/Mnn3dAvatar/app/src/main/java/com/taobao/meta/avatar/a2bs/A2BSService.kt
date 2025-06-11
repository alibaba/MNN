package com.taobao.meta.avatar.a2bs

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class A2BSService {

    private var a2bsServiceNative: Long = 0

    @Volatile
    private var isLoaded = false
    private var initDeferred: CompletableDeferred<Boolean>? = null
    private val mutex = Mutex()

    private fun destroy() {
        nativeDestroy(a2bsServiceNative)
        a2bsServiceNative = 0
        isLoaded = false
    }

    suspend fun waitForInitComplete(): Boolean {
        if (isLoaded) return true
        initDeferred?.let {
            return it.await()
        }
        return isLoaded
    }

    private fun loadA2bsResources(resourceDir: String?, tempDir: String): Boolean {
        val loadTime = System.currentTimeMillis()
        Log.d(TAG, "LoadA2bsResourcesFromFile begin ")
        val result =  nativeLoadA2bsResources(a2bsServiceNative, resourceDir, tempDir)
        Log.d(TAG, "LoadA2bsResourcesFromFile end, cost: ${System.currentTimeMillis() - loadTime}")
        return result
    }

    suspend fun init(modelDir: String?, context: Context): Boolean = mutex.withLock {
        if (isLoaded) return true
        val result = withContext(Dispatchers.IO) {
            val tempDir = context.cacheDir.absolutePath + "/a2bs_tmp"
            loadA2bsResources(modelDir, tempDir)
        }
        if (result) {
            isLoaded = true
            if (initDeferred != null) {
                initDeferred?.complete(true)
            }
        }
        return result
    }

    fun process(index:Int, audioData: ShortArray, sampleRate: Int): AudioToBlendShapeData {
        return nativeProcessBuffer(a2bsServiceNative, index, audioData, sampleRate)
    }

    private external fun nativeCreateA2BS(): Long
    private external fun nativeDestroy(nativePtr: Long)
    private external fun nativeLoadA2bsResources(
        nativePtr: Long,
        resourceDir: String?,
        tempDir: String
    ): Boolean

    private external fun nativeProcessBuffer(
        nativePtr: Long,
        index: Int,
        audioData: ShortArray,
        sampleRate: Int
    ): AudioToBlendShapeData

    init {
        a2bsServiceNative = nativeCreateA2BS()
    }

    companion object {
        private const val TAG = "A2BSService"

        init {
            System.loadLibrary("mnn_a2bs")
        }
    }
}
