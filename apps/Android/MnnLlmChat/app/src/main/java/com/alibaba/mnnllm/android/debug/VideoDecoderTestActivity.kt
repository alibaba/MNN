// Created by AI Assistant on 2025/01/01.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.debug

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.FileUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

class VideoDecoderTestActivity : AppCompatActivity() {

    companion object {
        const val TAG = "VideoDecoderTestActivity"
        private const val REQUEST_PICK_VIDEO_FILE = 1003
        
        init {
            try {
                System.loadLibrary("mnnllmapp")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library: ${e.message}")
            }
        }
    }
    
    // Native method declarations
    private external fun nativeCreateDecoder(): Long
    private external fun nativeDecodeVideo(
        decoderPtr: Long, 
        videoPath: String, 
        maxFrames: Int, 
        sampleInterval: Int, 
        maxBufferCount: Int
    ): Boolean
    private external fun nativeGetDecodedFrameCount(decoderPtr: Long): Int
    private external fun nativeDestroyDecoder(decoderPtr: Long)

    private lateinit var scrollView: ScrollView
    private lateinit var logTextView: TextView
    private lateinit var selectVideoButton: Button
    private lateinit var startDecodeButton: Button
    private lateinit var stopDecodeButton: Button
    private lateinit var clearLogButton: Button
    private lateinit var selectedVideoText: TextView
    private lateinit var resultsTextView: TextView
    private lateinit var maxFramesEditText: EditText
    private lateinit var sampleIntervalEditText: EditText
    private lateinit var maxBufferCountEditText: EditText

    private var selectedVideoPath: String? = null
    private var isDecoding = false
    private var nativeDecoderPtr: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_decoder_test)

        initViews()
        setupClickListeners()
        initNativeDecoder()
        log("Video Decoder Test Activity started")
        log("Note: Video file selection uses ACTION_GET_CONTENT - no storage permissions required")
        log("Long press video name to clear selection")
    }

    private fun initViews() {
        scrollView = findViewById(R.id.scrollView)
        logTextView = findViewById(R.id.logTextView)
        selectVideoButton = findViewById(R.id.selectVideoButton)
        startDecodeButton = findViewById(R.id.startDecodeButton)
        stopDecodeButton = findViewById(R.id.stopDecodeButton)
        clearLogButton = findViewById(R.id.clearLogButton)
        selectedVideoText = findViewById(R.id.selectedVideoText)
        resultsTextView = findViewById(R.id.resultsTextView)
        maxFramesEditText = findViewById(R.id.maxFramesEditText)
        sampleIntervalEditText = findViewById(R.id.sampleIntervalEditText)
        maxBufferCountEditText = findViewById(R.id.maxBufferCountEditText)
    }

    private fun setupClickListeners() {
        selectVideoButton.setOnClickListener {
            selectVideoFile()
        }

        startDecodeButton.setOnClickListener {
            startVideoDecode()
        }

        stopDecodeButton.setOnClickListener {
            stopVideoDecode()
        }

        clearLogButton.setOnClickListener {
            clearLog()
        }
        
        // Add a method to clear selected video
        selectedVideoText.setOnLongClickListener {
            clearSelectedVideo()
            true
        }
    }

    private fun initNativeDecoder() {
        try {
            nativeDecoderPtr = nativeCreateDecoder()
            if (nativeDecoderPtr != 0L) {
                log("Native decoder initialized successfully")
            } else {
                log("Failed to initialize native decoder")
            }
        } catch (e: Exception) {
            log("Error initializing native decoder: ${e.message}")
        }
    }

    private fun selectVideoFile() {
        openVideoFilePicker()
    }

    private fun openVideoFilePicker() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "video/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        try {
            startActivityForResult(
                Intent.createChooser(intent, getString(R.string.select_video)),
                REQUEST_PICK_VIDEO_FILE
            )
        } catch (ex: ActivityNotFoundException) {
            Toast.makeText(this, R.string.file_manager_required, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_PICK_VIDEO_FILE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                handleVideoFileSelection(uri)
            }
        }
    }

    private fun handleVideoFileSelection(uri: Uri) {
        try {
            // Use the same approach as AttachmentPickerModule
            val destVideoPath = FileUtils.generateDestVideoFilePath(this, "video_decoder_test")
            val destFile = FileUtils.copyFileUriToPath(this, uri, destVideoPath)
            
            if (destFile != null) {
                selectedVideoPath = destFile.absolutePath
                selectedVideoText.text = "Selected: ${destFile.name}"
                startDecodeButton.isEnabled = true
                log("Video file selected: ${destFile.absolutePath}")
            } else {
                log("Failed to copy video file")
                Toast.makeText(this, "Failed to copy video file", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            log("Error handling video file selection: ${e.message}")
            Toast.makeText(this, "Error selecting video file", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startVideoDecode() {
        if (selectedVideoPath.isNullOrEmpty()) {
            Toast.makeText(this, "Please select a video file first", Toast.LENGTH_SHORT).show()
            return
        }

        if (isDecoding) {
            Toast.makeText(this, "Decoding already in progress", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val maxFrames = maxFramesEditText.text.toString().toIntOrNull() ?: 50
            val sampleInterval = sampleIntervalEditText.text.toString().toIntOrNull() ?: 2
            val maxBufferCount = maxBufferCountEditText.text.toString().toIntOrNull() ?: 6

            log("Starting video decode with config:")
            log("  Max Frames: $maxFrames")
            log("  Sample Interval: $sampleInterval")
            log("  Max Buffer Count: $maxBufferCount")
            log("  Video Path: $selectedVideoPath")

            isDecoding = true
            startDecodeButton.isEnabled = false
            stopDecodeButton.isEnabled = true
            selectVideoButton.isEnabled = false

            // Start decoding in background
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val result = performVideoDecode(selectedVideoPath!!, maxFrames, sampleInterval, maxBufferCount)
                    
                    withContext(Dispatchers.Main) {
                        handleDecodeResult(result)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        handleDecodeError(e)
                    }
                }
            }

        } catch (e: Exception) {
            log("Error starting video decode: ${e.message}")
            Toast.makeText(this, "Error starting decode: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private suspend fun performVideoDecode(
        videoPath: String,
        maxFrames: Int,
        sampleInterval: Int,
        maxBufferCount: Int
    ): String {
        if (nativeDecoderPtr == 0L) {
            log("Native decoder not initialized, falling back to simulation")
            return performSimulatedDecode(maxFrames, sampleInterval)
        }
        
        try {
            log("Starting native video decode...")
            
            // Call native decoder
            val success = nativeDecodeVideo(nativeDecoderPtr, videoPath, maxFrames, sampleInterval, maxBufferCount)
            
            if (success) {
                val frameCount = nativeGetDecodedFrameCount(nativeDecoderPtr)
                log("Native decode successful, extracted $frameCount frames")
                return "Successfully decoded $frameCount frames from video using native decoder"
            } else {
                log("Native decode failed")
                return "Native video decode failed"
            }
            
        } catch (e: Exception) {
            log("Error in native decode: ${e.message}, falling back to simulation")
            return performSimulatedDecode(maxFrames, sampleInterval)
        }
    }
    
    private suspend fun performSimulatedDecode(maxFrames: Int, sampleInterval: Int): String {
        // Simulate video decoding process
        log("Starting video decode simulation...")
        
        // Simulate processing time
        kotlinx.coroutines.delay(2000)
        
        // Simulate frame extraction
        val totalFrames = 120 // Simulated total frames
        val extractedFrames = (totalFrames / sampleInterval).coerceAtMost(maxFrames)
        
        log("Simulated decode complete:")
        log("  Total frames in video: $totalFrames")
        log("  Extracted frames: $extractedFrames")
        log("  Sample interval: $sampleInterval")
        
        return "Successfully decoded $extractedFrames frames from video (simulated)"
    }

    private fun handleDecodeResult(result: String) {
        isDecoding = false
        startDecodeButton.isEnabled = true
        stopDecodeButton.isEnabled = false
        selectVideoButton.isEnabled = true
        
        resultsTextView.text = result
        log("Decode completed: $result")
        Toast.makeText(this, "Video decode completed", Toast.LENGTH_SHORT).show()
    }

    private fun handleDecodeError(error: Exception) {
        isDecoding = false
        startDecodeButton.isEnabled = true
        stopDecodeButton.isEnabled = false
        selectVideoButton.isEnabled = true
        
        val errorMessage = "Decode failed: ${error.message}"
        resultsTextView.text = errorMessage
        log(errorMessage)
        Toast.makeText(this, errorMessage, Toast.LENGTH_SHORT).show()
    }

    private fun stopVideoDecode() {
        if (!isDecoding) {
            Toast.makeText(this, "No decode in progress", Toast.LENGTH_SHORT).show()
            return
        }

        log("Stopping video decode...")
        isDecoding = false
        startDecodeButton.isEnabled = true
        stopDecodeButton.isEnabled = false
        selectVideoButton.isEnabled = true
        
        resultsTextView.text = "Decode stopped by user"
        Toast.makeText(this, "Video decode stopped", Toast.LENGTH_SHORT).show()
    }

    // No permission handling needed for ACTION_GET_CONTENT

    private fun log(message: String) {
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())
        val logMessage = "[$timestamp] $message\n"
        
        runOnUiThread {
            logTextView.append(logMessage)
            scrollView.post {
                scrollView.fullScroll(View.FOCUS_DOWN)
            }
        }
        
        Log.d(TAG, message)
    }

    private fun clearSelectedVideo() {
        selectedVideoPath?.let { path ->
            try {
                val file = File(path)
                if (file.exists()) {
                    file.delete()
                    log("Cleared selected video file: $path")
                }
            } catch (e: Exception) {
                log("Error clearing video file: ${e.message}")
            }
        }
        
        selectedVideoPath = null
        selectedVideoText.text = getString(R.string.no_video_selected)
        startDecodeButton.isEnabled = false
        log("Video selection cleared")
    }

    private fun clearLog() {
        logTextView.text = ""
        log("Log cleared")
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isDecoding) {
            stopVideoDecode()
        }
        
        // Clean up native decoder
        if (nativeDecoderPtr != 0L) {
            try {
                nativeDestroyDecoder(nativeDecoderPtr)
                nativeDecoderPtr = 0L
                log("Native decoder cleaned up")
            } catch (e: Exception) {
                log("Error cleaning up native decoder: ${e.message}")
            }
        }
        
        // Clean up copied video file
        selectedVideoPath?.let { path ->
            try {
                val file = File(path)
                if (file.exists()) {
                    file.delete()
                    log("Cleaned up temporary video file: $path")
                }
            } catch (e: Exception) {
                log("Error cleaning up video file: ${e.message}")
            }
        }
    }
}
