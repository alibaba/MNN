package com.alibaba.mnn.tts.demo

import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.util.Log
import com.taobao.meta.avatar.tts.TtsService
import java.io.File

class MnnTtsService : TextToSpeechService() {
    
    private var ttsService: TtsService? = null
    private var isInitialized = false
    private val defaultModelPath = "/data/local/tmp/tts_models/default"
    
    companion object {
        private const val TAG = "MnnTtsService"
        private val SUPPORTED_LANGUAGES = setOf("zh-CN", "zh_CN", "cmn-Hans-CN")
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "MnnTtsService created")
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        Log.d(TAG, "onIsLanguageAvailable: lang=$lang, country=$country, variant=$variant")
        
        // 检查是否支持中文
        val locale = buildLocaleString(lang, country)
        return when {
            SUPPORTED_LANGUAGES.contains(locale) -> TextToSpeech.LANG_COUNTRY_AVAILABLE
            lang == "zh" -> TextToSpeech.LANG_AVAILABLE
            else -> TextToSpeech.LANG_NOT_SUPPORTED
        }
    }

    override fun onGetLanguage(): Array<String> {
        Log.d(TAG, "onGetLanguage")
        // 返回默认语言：中文（中国）
        return arrayOf("zh", "CHN", "")
    }

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        Log.d(TAG, "onLoadLanguage: lang=$lang, country=$country, variant=$variant")
        
        val locale = buildLocaleString(lang, country)
        if (!SUPPORTED_LANGUAGES.contains(locale) && lang != "zh") {
            return TextToSpeech.LANG_NOT_SUPPORTED
        }

        // 初始化 TTS 引擎
        if (!isInitialized) {
            initializeTtsEngine()
        }

        return if (isInitialized) {
            TextToSpeech.LANG_COUNTRY_AVAILABLE
        } else {
            TextToSpeech.ERROR
        }
    }

    override fun onStop() {
        Log.d(TAG, "onStop")
        // 停止当前的合成任务
    }

    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) {
            Log.e(TAG, "Invalid synthesis request or callback")
            return
        }

        val text = request.charSequenceText?.toString() ?: request.text
        if (text.isNullOrEmpty()) {
            callback.error()
            return
        }

        Log.d(TAG, "onSynthesizeText: text=$text, language=${request.language}, country=${request.country}")

        try {
            // 确保 TTS 引擎已初始化
            if (!isInitialized) {
                initializeTtsEngine()
            }

            if (!isInitialized || ttsService == null) {
                Log.e(TAG, "TTS engine not initialized")
                callback.error()
                return
            }

            // TTS 引擎已初始化，直接使用

            // 开始合成
            val sampleRate = 44100
            callback.start(sampleRate, android.media.AudioFormat.ENCODING_PCM_16BIT, 1)

            // 使用 TTS 服务处理文本
            val audioData = ttsService?.process(text, 0)
            
            if (audioData != null && audioData.isNotEmpty()) {
                Log.d(TAG, "Generated ${audioData.size} audio samples")
                
                // 将 FloatArray 转换为 ByteArray (PCM 16-bit)
                val maxBufferSize = callback.maxBufferSize
                val byteBuffer = ByteArray(maxBufferSize)
                var offset = 0
                
                for (sample in audioData) {
                    // 转换 float 到 16-bit PCM
                    val pcmValue = (sample * 32767f).toInt().coerceIn(-32768, 32767).toShort()
                    
                    // 写入字节（小端序）
                    byteBuffer[offset++] = (pcmValue.toInt() and 0xFF).toByte()
                    byteBuffer[offset++] = ((pcmValue.toInt() shr 8) and 0xFF).toByte()
                    
                    // 当缓冲区满时，发送数据
                    if (offset >= maxBufferSize - 2) {
                        callback.audioAvailable(byteBuffer, 0, offset)
                        offset = 0
                    }
                }
                
                // 发送剩余数据
                if (offset > 0) {
                    callback.audioAvailable(byteBuffer, 0, offset)
                }
                
                callback.done()
                Log.d(TAG, "Synthesis completed successfully")
            } else {
                Log.e(TAG, "No audio data generated")
                callback.error()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during synthesis", e)
            callback.error()
        }
    }

    private fun initializeTtsEngine() {
        try {
            Log.d(TAG, "Initializing TTS engine with model: $defaultModelPath")
            
            // 检查模型文件是否存在
            val modelDir = File(defaultModelPath)
            if (!modelDir.exists() || !modelDir.isDirectory) {
                Log.e(TAG, "Model directory not found: $defaultModelPath")
                return
            }

            val configFile = File(modelDir, "config.json")
            if (!configFile.exists()) {
                Log.e(TAG, "config.json not found in model directory")
                return
            }

            // 初始化 TTS 服务（同步调用）
            ttsService = TtsService()
            // 注意：这里假设 init 方法有同步版本，如果没有需要使用 runBlocking
            val initResult = try {
                kotlinx.coroutines.runBlocking {
                    ttsService?.init(defaultModelPath) ?: false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during TTS init", e)
                false
            }
            
            if (initResult) {
                isInitialized = true
                Log.d(TAG, "TTS engine initialized successfully")
            } else {
                Log.e(TAG, "Failed to initialize TTS engine")
                ttsService = null
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing TTS engine", e)
            ttsService = null
            isInitialized = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "MnnTtsService destroyed")
        
        try {
            ttsService?.destroy()
            ttsService = null
            isInitialized = false
        } catch (e: Exception) {
            Log.e(TAG, "Error destroying TTS service", e)
        }
    }

    private fun buildLocaleString(lang: String?, country: String?): String {
        return when {
            lang.isNullOrEmpty() -> ""
            country.isNullOrEmpty() -> lang
            else -> "$lang-$country"
        }
    }
}
