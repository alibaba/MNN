package com.mnn.tts.demo

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import com.taobao.meta.avatar.tts.TtsService
import com.alibaba.mnn.tts.demo.SherpaTts
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.util.Locale
import org.json.JSONObject

class MnnTtsService : TextToSpeechService() {

    // 使用 SupervisorJob，这样如果一个子协程崩了，不会导致整个 Scope 失效
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // 互斥锁，确保同一时间只有一个合成任务在跑推理
    private val synthesisMutex = Mutex()
    
    // 当前正在运行的合成任务 Job，用于 onStop 时取消
    private var synthesisJob: Job? = null

    private var ttsService: TtsService? = null
    private var sherpaTts: SherpaTts? = null
    private var currentLanguage: String = "zh" // 默认中文（两字母代码，用于内部处理）
    private var currentLanguageOriginal: String = "zh" // 原始语言代码（可能三字母，用于返回给系统）
    private var currentCountry: String = "" // 当前国家代码
    private var modelPath: String? = null
    private var sampleRate: Int = 16000 // 从 config.json 读取的采样率，默认 16000

    companion object {
        private const val TAG = "MnnTtsService"
        private const val PREFS_NAME = "mnn_tts_prefs"
        private const val KEY_MODEL_PATH = "model_path"
        // 注意：/data/local/tmp 通常需要 Root 权限或 Debug 模式才能访问
        private const val DEFAULT_MODEL_PATH = "/data/local/tmp/tts_models"
        
        // 音频参数常量
        private const val DEFAULT_SAMPLE_RATE = 16000 // 默认采样率
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        private const val CHANNEL_COUNT = 1
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "MnnTtsService onCreate")
        // 异步加载模型，避免阻塞主线程导致 ANR
        serviceScope.launch {
            loadModelPath()
            initializeTtsService()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "MnnTtsService onDestroy")
        serviceScope.cancel() // 取消所有协程
        try {
            ttsService?.destroy()
            sherpaTts?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error destroying TTS service", e)
        }
        ttsService = null
    }

    // --- 核心方法：合成文本 ---
    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        // 使用最高级别的日志，确保能看到
        Log.e(TAG, "🔥🔥🔥🔥🔥 [onSynthesizeText 被调用!] Lang: ${request?.language}, Text: ${request?.charSequenceText?.take(50)}")
        if (callback == null || request == null) {
            Log.e(TAG, "❌ onSynthesizeText: callback or request is null!")
            return
        }

        val text = request.charSequenceText?.toString() ?: ""
        
        Log.d(TAG, "[合成开始] Text: \"$text\", Lang: ${request.language}")

        // 1. 如果上一个任务还在跑，先取消它
        runCatching { synthesisJob?.cancel() }

        // 2. 启动新的协程任务
        synthesisJob = serviceScope.launch {
            // 使用 Mutex 锁住，防止多线程并发调用底层 C++ 引擎导致 crash
            synthesisMutex.withLock {
                try {
                    // 检查是否已被取消 (比如用户刚点播放立刻点了暂停)
                    if (!isActive) return@withLock

                    // 检查 TTS 引擎状态
                    if (ttsService == null && sherpaTts == null) {
                        Log.e(TAG, "TTS service is null, attempting re-init")
                        initializeTtsService()
                        if (ttsService == null && sherpaTts == null) {
                            callback.error()
                            return@withLock
                        }
                    }

                    var audioData: FloatArray? = null
                    var audioShortData: ShortArray? = null

                    if (sherpaTts != null) {
                        val generated = sherpaTts?.process(text)
                        audioData = generated?.samples
                        if (generated?.sampleRate != null) {
                            sampleRate = generated.sampleRate
                        }
                    } else {
                        // 设置语言
                        val reqLang = request.language ?: currentLanguage
                        val langCode = if (reqLang.lowercase().contains("en")) "en" else "zh"
                        ttsService?.setLanguage(langCode)

                        // 等待 TTS 服务初始化完成（关键修复：确保模型已加载）
                        val isReady = ttsService?.waitForInitComplete() ?: false
                        if (!isReady) {
                            Log.e(TAG, "TTS service not ready after waiting")
                            callback.error()
                            return@withLock
                        }

                        // 执行推理 (耗时操作)
                        audioShortData = ttsService?.process(text, 0)
                    }

                    // 再次检查取消状态
                    if (!isActive) return@withLock

                    if ((audioData == null || audioData.isEmpty()) && (audioShortData == null || audioShortData.isEmpty())) {
                        Log.w(TAG, "Generated audio data is empty")
                        callback.error()
                        return@withLock
                    }

                    // 3. 开始向系统写入数据 (全程在 IO 线程，不要切换到 Main)
                    
                    // Step A: 告诉系统准备接收音频
                    // start 返回 ERROR 表示系统侧可能已断开
                    // 使用从 config.json 读取的采样率
                    if (callback.start(sampleRate, ENCODING, CHANNEL_COUNT) != TextToSpeech.SUCCESS) {
                        Log.w(TAG, "callback.start failed, system may have aborted")
                        return@withLock
                    }

                    // Step B: 格式转换 -> ByteArray (Little Endian)
                    val dataSize = audioData?.size ?: audioShortData!!.size
                    val byteArray = ByteArray(dataSize * 2)

                    if (audioData != null) {
                        for (i in audioData.indices) {
                            val s = (audioData[i] * 32767).toInt().coerceIn(-32768, 32767)
                            byteArray[i * 2] = (s and 0xFF).toByte()
                            byteArray[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
                        }
                    } else if (audioShortData != null) {
                        for (i in audioShortData.indices) {
                            val s = audioShortData[i].toInt()
                            byteArray[i * 2] = (s and 0xFF).toByte()
                            byteArray[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
                        }
                    }

                    // Step C: 写入数据
                    // 这里是一次性写入。如果数据量极大，建议分块写入(chunked)
                    val maxBufferSize = callback.maxBufferSize
                    var offset = 0
                    while (offset < byteArray.size && isActive) {
                        val bytesToWrite = Math.min(maxBufferSize, byteArray.size - offset)
                        val result = callback.audioAvailable(byteArray, offset, bytesToWrite)
                        
                        if (result != TextToSpeech.SUCCESS) {
                            Log.w(TAG, "callback.audioAvailable failed")
                            return@withLock
                        }
                        offset += bytesToWrite
                    }

                    // Step D: 结束
                    if (isActive) {
                        callback.done()
                        Log.d(TAG, "[合成完成] Sent ${byteArray.size} bytes")
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Synthesis critical error", e)
                    // 防止崩溃传递给系统
                    if (isActive) callback.error()
                }
            }
        }
    }

    // --- 核心方法：停止合成 ---
    override fun onStop() {
        Log.d(TAG, "onStop called - INTERRUPT")
        // 关键：立即取消当前的协程任务
        synthesisJob?.cancel()
        synthesisJob = null
    }

// --- 修改 1: 优化语言检查，支持三字母代码 (ISO 639-2) ---
    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        Log.e(TAG, "🔥系统询问语言检查: lang=$lang, country=$country, variant=$variant")
        
        // 简单的模糊匹配：只要包含 "zh", "cn", "en", "eng" 就认为支持
        val l = (lang ?: "").lowercase()
        if (l.contains("zh") || l.contains("cn") || l.contains("en") || l.contains("eng")) {
            return TextToSpeech.LANG_COUNTRY_AVAILABLE
        }
        
        // 暂时为了调试，还是返回成功，但正常应该返回 NOT_SUPPORTED
        return TextToSpeech.LANG_COUNTRY_AVAILABLE
    }

    // --- 修改 2: 加载语言也强制通过 ---
    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        Log.e(TAG, "🔥系统请求加载语言: lang=$lang, country=$country, variant=$variant")
        
        // 保存原始语言代码（可能三字母），用于 onGetLanguage 返回
        currentLanguageOriginal = lang ?: "zh"
        
        // 转换三字母代码为两字母代码（ISO 639-2 -> ISO 639-1），用于内部处理
        val langCode = when {
            lang == null -> "zh"
            lang.lowercase().contains("eng") || lang.lowercase().contains("en") -> "en"
            lang.lowercase().contains("zh") || lang.lowercase().contains("cn") -> "zh"
            else -> lang.take(2).lowercase() // 取前两个字符
        }
        currentLanguage = langCode
        currentCountry = country ?: ""
        Log.e(TAG, "🔥设置 currentLanguage = $currentLanguage (内部), currentLanguageOriginal = $currentLanguageOriginal (返回系统), currentCountry = $currentCountry")
        return TextToSpeech.LANG_COUNTRY_AVAILABLE
    }

    
    override fun onGetLanguage(): Array<String> {
        // 返回格式: [language, country, variant]
        // 【关键】返回当前实际使用的语言代码（可能是三字母），以匹配系统请求
        val country = when (currentLanguage) {
            "en" -> "USA"
            "zh" -> "CN"
            else -> currentCountry.ifEmpty { "CN" }
        }
        return arrayOf(currentLanguageOriginal, country, "")
    }

    // ---------------------------------------------------------
    // 新增：必须告诉系统你有具体的"发音人"，否则高版本安卓不理你
    // ---------------------------------------------------------

    override fun onGetVoices(): List<Voice> {
        // 定义一个中文发音人
        val zhVoice = Voice(
            "mnn_zh_voice",           // 唯一ID
            Locale.CHINA,             // 对应的 Locale
            Voice.QUALITY_HIGH,       // 质量
            Voice.LATENCY_NORMAL,     // 延迟
            false,                    // 是否需要网络
            setOf("male")             // 特征 (male/female)
        )
        
        // 定义一个英文发音人 (对应系统请求的 eng-USA)
        val enVoice = Voice(
            "mnn_en_voice",
            Locale.US,
            Voice.QUALITY_HIGH,
            Voice.LATENCY_NORMAL,
            false,
            setOf("female")
        )

        Log.e(TAG, "🔥系统获取 Voice 列表: [mnn_zh_voice, mnn_en_voice]")
        return listOf(zhVoice, enVoice)
    }

    override fun onIsValidVoiceName(voiceName: String?): Int {
        val result = if (voiceName == "mnn_zh_voice" || voiceName == "mnn_en_voice") {
            TextToSpeech.SUCCESS
        } else {
            TextToSpeech.ERROR
        }
        Log.d(TAG, "系统验证 Voice 名称: $voiceName -> $result")
        return result
    }

    override fun onLoadVoice(voiceName: String?): Int {
        Log.e(TAG, "🔥🔥🔥 系统请求加载 Voice: $voiceName")
        
        // 根据 Voice 名字切换你的模型或参数
        if (voiceName == "mnn_en_voice") {
            currentLanguage = "en" // 内部使用两字母
            currentLanguageOriginal = "eng" // 返回给系统使用三字母
            currentCountry = "USA" // 设置对应的国家代码
        } else {
            currentLanguage = "zh"
            currentLanguageOriginal = "zh"
            currentCountry = "CHN"
        }
        
        // 【关键修复】确保 TTS 服务已经初始化完成
        // 如果还在初始化中，等待一下（最多等待 3 秒）
        if (ttsService == null) {
            Log.i(TAG, "TTS service ready after waiting ms")
        } else {
            Log.i(TAG, "TTS service already initialized")
        }
        
        Log.e(TAG, "🔥🔥🔥 onLoadVoice 返回 SUCCESS, currentLanguage=$currentLanguage, currentCountry=$currentCountry")
        return TextToSpeech.SUCCESS
    }

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String? {
        // 当系统请求英语时，默认返回英文 Voice 的名字
        val checkLang = (lang ?: "").lowercase()
        val voiceName = if (checkLang.contains("en") || checkLang.contains("eng")) {
            "mnn_en_voice"
        } else {
            "mnn_zh_voice"
        }
        Log.d(TAG, "系统请求默认 Voice: lang=$lang -> $voiceName")
        return voiceName
    }

    // --- 初始化与路径逻辑 (保持原逻辑优化) ---
    private fun loadModelPath() {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        modelPath = prefs.getString(KEY_MODEL_PATH, null)
        
        if (modelPath.isNullOrEmpty()) {
            modelPath = findDefaultModel()
        }
        Log.i(TAG, "Model Path resolved to: $modelPath")
    }

    private fun findDefaultModel(): String? {
        // 优先检查 App 私有目录 (更安全)
        val privatePath = File(getExternalFilesDir(null), "tts_models")
        if (privatePath.exists()) return privatePath.absolutePath

        // 检查原始路径
        val legacyPath = File(DEFAULT_MODEL_PATH)
        if (legacyPath.exists() && legacyPath.isDirectory) {
             // 简单的查找逻辑
             legacyPath.listFiles()?.firstOrNull { it.isDirectory }?.let {
                return it.absolutePath
             }
        }
        return null
    }

private suspend fun initializeTtsService() {
        // 1. 【关键修复】将可变的成员变量赋值给不可变的局部变量
        // 这样编译器就确定 path 在这个函数里永远不会变了
        val path = modelPath 

        if (path.isNullOrEmpty()) {
            Log.e(TAG, "Skipping init: No model path")
            return
        }
        
        // 避免重复初始化
        if (ttsService != null || sherpaTts != null) return

        try {
            val isSherpa = File(path, "voices.bin").exists() || File(path, "model.mnn").exists()
            if (isSherpa) {
                 Log.i(TAG, "Initializing SherpaTts with $path")
                 sherpaTts = SherpaTts()
                 sherpaTts?.init(path)
                 loadSampleRateFromConfig(path)
                 Log.i(TAG, "SherpaTts initialized. Sample rate: $sampleRate Hz")
            } else {
                // 从 config.json 读取采样率
                loadSampleRateFromConfig(path)
                
                val service = TtsService()
                
                // 2. 这里使用局部变量 path，它已经被智能转换为非空 String 了
                val success = service.init(path) 
                
                if (success) {
                    ttsService = service
                    Log.i(TAG, "TTS Engine Initialized! Sample rate: $sampleRate Hz")
                } else {
                    Log.e(TAG, "TTS Engine Init Failed (return false)")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "TTS Engine Init Exception", e)
        }
    }
    
    // 从 config.json 读取采样率
    private fun loadSampleRateFromConfig(modelPath: String) {
        try {
            val configFile = File(modelPath, "config.json")
            if (configFile.exists() && configFile.isFile) {
                val configContent = configFile.readText()
                val configJson = JSONObject(configContent)
                
                if (configJson.has("sample_rate")) {
                    sampleRate = configJson.getInt("sample_rate")
                    Log.i(TAG, "Loaded sample rate from config.json: $sampleRate Hz")
                } else {
                    Log.w(TAG, "config.json does not contain 'sample_rate', using default: $DEFAULT_SAMPLE_RATE Hz")
                    sampleRate = DEFAULT_SAMPLE_RATE
                }
            } else {
                Log.w(TAG, "config.json not found at $modelPath/config.json, using default: $DEFAULT_SAMPLE_RATE Hz")
                sampleRate = DEFAULT_SAMPLE_RATE
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading sample rate from config.json", e)
            sampleRate = DEFAULT_SAMPLE_RATE
        }
    }
    
    // 供外部 Activity 调用更新模型路径
    fun updateModelPath(path: String) {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        prefs.edit().putString(KEY_MODEL_PATH, path).apply()
        
        // 重启服务逻辑
        serviceScope.launch {
            synthesisMutex.withLock {
                ttsService?.destroy()
                ttsService = null
                sherpaTts?.release()
                sherpaTts = null
                modelPath = path
                // 重新读取采样率并初始化服务
                initializeTtsService()
            }
        }
    }
}