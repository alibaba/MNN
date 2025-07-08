package com.mnn.tts.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class DemoActivity : AppCompatActivity() {
    private lateinit var ttsDemo: TtsServiceDemo
    private lateinit var textInput: EditText
    private lateinit var speakButton: Button
    private lateinit var languageGroup: RadioGroup

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_demo)

        textInput = findViewById(R.id.textInput)
        speakButton = findViewById(R.id.speakButton)
        languageGroup = findViewById(R.id.languageGroup)

        ttsDemo = TtsServiceDemo(this)

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
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val modelDir = getExternalFilesDir(null)?.absolutePath + "/tts_models"
                ttsDemo.initialize(modelDir)
                speakButton.isEnabled = true
            } catch (e: Exception) {
                Toast.makeText(this@DemoActivity, "Failed to initialize TTS", Toast.LENGTH_SHORT).show()
            }
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
    }
} 