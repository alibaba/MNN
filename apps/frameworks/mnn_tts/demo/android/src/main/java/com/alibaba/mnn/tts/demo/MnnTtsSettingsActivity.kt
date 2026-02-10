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
        val modelDir = File(modelPath)
        val modelExists = modelDir.exists() && modelDir.isDirectory
        val configExists = File(modelDir, "config.json").exists()
        
        infoText.text = """
            MNN TTS Engine Settings
            
            Model Path: $modelPath
            Model Directory: ${if (modelExists) "✓ Found" else "✗ Not Found"}
            Config File: ${if (configExists) "✓ Found" else "✗ Not Found"}
            
            Supported Languages:
            - Chinese (China): zh-CN
            
            Note: Please ensure TTS models are placed in:
            /data/local/tmp/tts_models/default/
            
            Required files:
            - config.json
            - model.mnn
            - (other model files)
            
            Use adb to push models:
            adb push /path/to/model /data/local/tmp/tts_models/default/
        """.trimIndent()
    }
}
