package com.taobao.meta.avatar.tts

import android.content.Context
import android.content.SharedPreferences
import android.preference.PreferenceManager
import android.util.Log
import com.k2fsa.sherpa.mnn.GeneratedAudio
import com.taobao.meta.avatar.MHConfig
import com.taobao.meta.avatar.debug.DebugModule
import com.taobao.meta.avatar.settings.MainSettings
import com.taobao.meta.avatar.utils.AppUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import org.json.JSONObject

class TtsService {

    val useSherpaTts
        get() = DebugModule.TTS_USE_SHERPA

    private var sherpaTts: SherpaTts? = null
    private var ttsServiceNative: Long = 0
    @Volatile
    private var isLoaded = false
    private var initDeferred: Deferred<Boolean>? = null
    private var sharedPreferences: SharedPreferences? = null
    private var applicationContext: Context? = null
    private var currentSpeakerId: String? = null
    private val preferenceChangeListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key == "tts_speaker_id") {
            handleSpeakerIdChange()
        }
    }

    init {
        ttsServiceNative = nativeCreateTTS(if (AppUtils.isChinese())  "zh" else "en")
    }

    fun destroy() {
        // 取消注册 SharedPreferences 监听器
        sharedPreferences?.unregisterOnSharedPreferenceChangeListener(preferenceChangeListener)
        sharedPreferences = null
        applicationContext = null
        nativeDestroy(ttsServiceNative)
        ttsServiceNative = 0
    }

    suspend fun init(modelDir: String?, context: Context? = null): Boolean {
        if (isLoaded) return true
        if (initDeferred == null) {
            initDeferred = CoroutineScope(Dispatchers.IO).async {
                if (useSherpaTts) {
                    sherpaTts = SherpaTts()
                    sherpaTts?.init(null)
                    return@async true
                }
                
                // 1. 根据语言选择模型目录
                val isChinese = AppUtils.isChinese()
                val actualModelDir = if (isChinese) {
                    MHConfig.TTS_MODEL_DIR
                } else {
                    MHConfig.TTS_MODEL_DIR_EN
                }
                
                // 2. 构建参数覆盖 Map
                val overrideParams = mutableMapOf<String, String>()
                context?.let {
                    // 只有英文模式支持 speaker_id
                    if (!isChinese) {
                        val speakerId = MainSettings.getTtsSpeakerId(it)
                        if (speakerId.isNotEmpty()) {
                            overrideParams["speaker_id"] = speakerId
                        }
                    }
                    
                    // speed 设置暂时不支持，使用 config.json 中的默认值
                    // val speed = MainSettings.getTtsSpeed(it)
                    // overrideParams["speed"] = speed.toString()
                }
                
                // 3. 序列化为 JSON
                val paramsJson = if (overrideParams.isEmpty()) {
                    "{}"
                } else {
                    JSONObject(overrideParams as Map<*, *>).toString()
                }
                
                Log.d(TAG, "Loading TTS from: $actualModelDir with params: $paramsJson")
                
                nativeLoadResourcesFromFile(
                    ttsServiceNative,
                    actualModelDir,
                    "",
                    "",
                    paramsJson
                )
                true
            }
        }
        val result = initDeferred!!.await()
        if (result) {
            isLoaded = true
            // 注册 SharedPreferences 监听器（当有 Context 时）
            context?.let {
                registerPreferenceListener(it)
                // 初始化当前 speaker ID
                if (!AppUtils.isChinese()) {
                    currentSpeakerId = MainSettings.getTtsSpeakerId(it)
                }
            }
        }
        return result
    }
    
    private fun registerPreferenceListener(context: Context) {
        if (sharedPreferences == null) {
            // 保存 applicationContext 避免内存泄漏
            applicationContext = context.applicationContext
            sharedPreferences = PreferenceManager.getDefaultSharedPreferences(applicationContext)
            sharedPreferences?.registerOnSharedPreferenceChangeListener(preferenceChangeListener)
            Log.d(TAG, "Registered SharedPreferences listener for speaker ID changes")
        }
    }
    
    private fun handleSpeakerIdChange() {
        if (!isLoaded) {
            Log.d(TAG, "TtsService not loaded yet, speaker ID change will be applied after initialization")
            return
        }
        
        // 只在英文模式下处理
        if (AppUtils.isChinese()) {
            return
        }
        
        applicationContext?.let { ctx ->
            val newSpeakerId = MainSettings.getTtsSpeakerId(ctx)
            if (newSpeakerId != currentSpeakerId && newSpeakerId.isNotEmpty()) {
                try {
                    setSpeakerId(newSpeakerId)
                    currentSpeakerId = newSpeakerId
                    Log.d(TAG, "Speaker ID changed to: $newSpeakerId")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to set speaker ID: $newSpeakerId", e)
                }
            }
        }
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

    fun setSpeakerId(speakerId: String) {
        nativeSetSpeakerId(ttsServiceNative, speakerId)
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
                                                     modelName: String,
                                                     mmapDir: String,
                                                     paramsJson: String): Boolean  // 新增：JSON 格式的参数覆盖
    private external fun nativeSetSpeakerId(nativePtr: Long, speakerId: String)  // 动态设置音色
    private external fun nativeProcess(nativePtr: Long, text: String, id: Int): ShortArray

    companion object {
        private const val TAG = "TtsService"
    }

}
