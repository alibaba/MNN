# MNN TTS 注册为 Android 系统 TTS 服务指南

## 概述

要将 MNN TTS 注册为 Android 系统 TTS 服务，需要实现 Android 的 `TextToSpeechService` 抽象类，并正确配置相关文件。

## 实现步骤

### 1. 创建 TTS 服务类

需要创建一个继承自 `android.speech.tts.TextToSpeechService` 的服务类：

**文件位置**: `src/main/java/com/alibaba/mnn/tts/demo/MnnTtsService.kt`

```kotlin
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

            // 等待初始化完成
            val isReady = ttsService?.waitForInitComplete() ?: false
            if (!isReady) {
                Log.e(TAG, "TTS engine not ready")
                callback.error()
                return
            }

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

            // 初始化 TTS 服务
            ttsService = TtsService()
            val initResult = ttsService?.init(defaultModelPath) ?: false
            
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
```

### 2. 创建 TTS 设置 Activity

**文件位置**: `src/main/java/com/alibaba/mnn/tts/demo/MnnTtsSettingsActivity.kt`

```kotlin
package com.alibaba.mnn.tts.demo

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class MnnTtsSettingsActivity : AppCompatActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tts_settings)
        
        // 显示当前配置信息
        val infoText = findViewById<TextView>(R.id.settingsInfoText)
        val modelPath = "/data/local/tmp/tts_models/default"
        val modelExists = File(modelPath).exists()
        
        infoText.text = """
            MNN TTS Engine Settings
            
            Model Path: $modelPath
            Model Status: ${if (modelExists) "Available" else "Not Found"}
            
            Supported Languages:
            - Chinese (China): zh-CN
            
            Note: Please ensure TTS models are placed in:
            /data/local/tmp/tts_models/
        """.trimIndent()
    }
}
```

### 3. 创建设置界面布局

**文件位置**: `src/main/res/layout/activity_tts_settings.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:id="@+id/settingsInfoText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textSize="14sp"
        android:fontFamily="monospace" />

</LinearLayout>
```

### 4. 创建 TTS 引擎配置文件

**文件位置**: `src/main/res/xml/tts_engine.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<tts-engine xmlns:android="http://schemas.android.com/apk/res/android"
    android:settingsActivity="com.alibaba.mnn.tts.demo.MnnTtsSettingsActivity" />
```

**注意**: 
- 简化的配置文件只包含必需的 `settingsActivity` 属性
- `<voice>` 标签中的属性（如 `android:locale`, `android:gender` 等）在较低 API 级别不支持，已移除
- 语言支持通过 `MnnTtsService` 中的 `onIsLanguageAvailable()` 和 `onGetLanguage()` 方法实现

### 5. 更新 AndroidManifest.xml

AndroidManifest.xml 已经包含了必要的配置：

```xml
<!-- TTS 服务声明 -->
<service
    android:name="com.alibaba.mnn.tts.demo.MnnTtsService"
    android:label="MNN TTS Engine"
    android:exported="true">
    <intent-filter>
        <action android:name="android.intent.action.TTS_SERVICE" />
    </intent-filter>
    <meta-data
        android:name="android.speech.tts"
        android:resource="@xml/tts_engine" />
</service>

<!-- TTS 设置 Activity -->
<activity
    android:name="com.alibaba.mnn.tts.demo.MnnTtsSettingsActivity"
    android:label="MNN TTS Settings"
    android:exported="true">
    <intent-filter>
        <action android:name="android.intent.action.TTS_SERVICE_SETTINGS" />
    </intent-filter>
</activity>
```

### 6. 更新 strings.xml

**文件位置**: `src/main/res/values/strings.xml`

添加以下字符串资源：

```xml
<string name="tts_engine_name">MNN TTS Engine</string>
<string name="tts_settings_title">MNN TTS Settings</string>
```

## 使用方法

### 1. 准备模型文件

将 TTS 模型文件放置到设备的以下目录：
```
/data/local/tmp/tts_models/default/
├── config.json
├── model.mnn
└── (其他模型文件)
```

使用 adb 命令推送模型：
```bash
adb push /path/to/model /data/local/tmp/tts_models/default/
```

### 2. 安装应用

```bash
./gradlew installDebug
```

### 3. 在系统设置中启用

1. 打开 **设置** → **系统** → **语言和输入法** → **文字转语音输出**
2. 选择 **首选引擎** → **MNN TTS Engine**
3. 点击设置图标可以查看引擎配置信息

### 4. 测试 TTS

在系统 TTS 设置页面点击"播放"按钮测试，或使用以下代码：

```kotlin
val tts = TextToSpeech(context) { status ->
    if (status == TextToSpeech.SUCCESS) {
        tts.language = Locale.CHINA
        tts.speak("你好，这是MNN TTS测试", TextToSpeech.QUEUE_FLUSH, null, null)
    }
}
```

## 关键要点

1. **服务生命周期**: `TextToSpeechService` 由系统管理，会在需要时创建和销毁
2. **线程安全**: 合成方法可能在后台线程调用，需要注意线程安全
3. **音频格式**: 必须使用 PCM 16-bit 格式输出音频数据
4. **语言支持**: 通过 `onIsLanguageAvailable` 声明支持的语言
5. **模型路径**: 确保模型文件路径可访问且包含必要的配置文件

## 调试技巧

1. 使用 `adb logcat -s MnnTtsService` 查看 TTS 服务日志
2. 检查 `/data/local/tmp/tts_models/` 目录权限
3. 确保应用有必要的权限（已在 AndroidManifest.xml 中声明）
4. 在系统 TTS 设置中测试引擎是否正常工作

## 常见问题

### Q: 系统设置中看不到 MNN TTS Engine
**A**: 检查 AndroidManifest.xml 中的 service 配置，确保 `android:exported="true"` 且包含正确的 intent-filter

###
