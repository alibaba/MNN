package com.mnn.tts.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnn.tts.demo.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class DemoActivity : AppCompatActivity() {
    private lateinit var ttsDemo: TtsServiceDemo
    private lateinit var textInput: EditText
    private lateinit var speakButton: Button
    private lateinit var languageGroup: RadioGroup
    private lateinit var modelRecyclerView: RecyclerView
    private lateinit var modelAdapter: ModelAdapter
    private var selectedModelPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_demo)

        textInput = findViewById(R.id.textInput)
        speakButton = findViewById(R.id.speakButton)
        languageGroup = findViewById(R.id.languageGroup)
        modelRecyclerView = findViewById(R.id.modelRecyclerView)

        ttsDemo = TtsServiceDemo(this)

        // 初始化模型列表
        initModelList()

        speakButton.setOnClickListener {
            val text = textInput.text.toString()
            if (text.isNotEmpty()) {
                val language = when (languageGroup.checkedRadioButtonId) {
                    R.id.chineseRadio -> "zh"
                    else -> "en"
                }
                ttsDemo.speak(text, language)
            } else {
                Toast.makeText(this, "Please enter text to speak", Toast.LENGTH_SHORT).show()
            }
        }

        checkPermissionAndInitialize()
    }

    private fun initModelList() {
        modelAdapter = ModelAdapter { modelPath ->
            selectedModelPath = modelPath
            loadTtsModel(modelPath)
        }
        modelRecyclerView.layoutManager = LinearLayoutManager(this)
        modelRecyclerView.adapter = modelAdapter

        lifecycleScope.launch {
            val models = scanTtsModels()
            modelAdapter.updateModels(models)
            if (models.isEmpty()) {
                Toast.makeText(this@DemoActivity, "No TTS models found in /data/local/tmp/tts_models", Toast.LENGTH_LONG).show()
            }
        }
    }

    private suspend fun scanTtsModels(): List<String> = withContext(Dispatchers.IO) {
        val modelsDir = File("/data/local/tmp/tts_models")
        val modelList = mutableListOf<String>()

        try {
            if (modelsDir.exists() && modelsDir.isDirectory) {
                modelsDir.listFiles()?.forEach { file ->
                    if (file.isDirectory) {
                        // 检查是否包含 config.json（可选，也可以直接添加所有文件夹）
                        val configFile = File(file, "config.json")
                        if (configFile.exists()) {
                            modelList.add(file.absolutePath)
                            Log.d(TAG, "Found model: ${file.absolutePath}")
                        } else {
                            // 如果没有 config.json，也添加文件夹（可能是有效的模型目录）
                            modelList.add(file.absolutePath)
                            Log.d(TAG, "Found model directory (no config.json): ${file.absolutePath}")
                        }
                    }
                }
            } else {
                Log.w(TAG, "Models directory does not exist: ${modelsDir.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error scanning models directory", e)
        }

        modelList.sorted()
    }

    private fun loadTtsModel(modelPath: String) {
        lifecycleScope.launch {
            try {
                Toast.makeText(this@DemoActivity, "Loading model: ${File(modelPath).name}...", Toast.LENGTH_SHORT).show()
                Log.d(TAG, "Loading TTS model: $modelPath")
                ttsDemo.initialize(modelPath)
                speakButton.isEnabled = true
                Toast.makeText(this@DemoActivity, "Model loaded: ${File(modelPath).name}", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "Error loading TTS model", e)
                Toast.makeText(this@DemoActivity, "Failed to load model: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun checkPermissionAndInitialize() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
        } else {
            initializeTts()
        }
    }

    private fun initializeTts() {
        // 不再在这里初始化，而是等待用户从列表中选择模型
        // 如果已经有选中的模型，则加载它
        if (selectedModelPath != null) {
            loadTtsModel(selectedModelPath!!)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initializeTts()
            } else {
                Toast.makeText(this, "Audio permission is required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsDemo.destroy()
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val TAG = "DemoActivity"
    }
} 