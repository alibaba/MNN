// Created by ruoyi.sjd on 2025/6/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.lifecycle.lifecycleScope
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.mainsettings.MainSettings.isStopDownloadOnChatEnabled
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.qnn.QnnModule
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Progress dialog that can update download progress in real-time
 */
class ProgressDialog(
    private val context: Context,
    private val modelId: String? = null
) {
    private var dialog: Dialog? = null
    private var progressIndicator: CircularProgressIndicator? = null
    private var progressText: TextView? = null
    private var downloadListener: DownloadListener? = null
    private var downloadManager: ModelDownloadManager? = null
    
    fun show(): ProgressDialog {
        val inflater = LayoutInflater.from(context)
        val view = inflater.inflate(R.layout.dialog_progress_material3, null)
        
        progressIndicator = view.findViewById(R.id.progress_indicator)
        progressText = view.findViewById(R.id.progress_text)
        
        // Set progress indicator to determinate mode
        progressIndicator?.isIndeterminate = false
        progressIndicator?.max = 100
        progressIndicator?.progress = 0
        
        dialog = MaterialAlertDialogBuilder(context)
            .setView(view)
            .setCancelable(false)
            .create()
        
        dialog?.show()
        
        // Set up download listener if modelId is provided
        modelId?.let { setupDownloadListener(it) }
        
        return this
    }
    
    fun setupDownloadListener(modelId: String) {
        downloadManager = ModelDownloadManager.getInstance(context)
        downloadListener = object : DownloadListener {
            override fun onDownloadStart(modelId: String) {
                updateProgress(0.0)
            }
            
            override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
                updateProgress(downloadInfo.progress)
            }
            
            override fun onDownloadFinished(modelId: String, path: String) {
                updateProgress(1.0)
            }
            
            override fun onDownloadFailed(modelId: String, e: Exception) {
                updateProgress(0.0)
            }
            
            override fun onDownloadPaused(modelId: String) {
                updateProgress(0.0)
            }
            
            override fun onDownloadFileRemoved(modelId: String) {}
            override fun onDownloadTotalSize(modelId: String, totalSize: Long) {}
            override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {}
        }
        
        downloadManager?.addListener(downloadListener!!)
    }
    
    private fun updateProgress(progress: Double) {
        MainScope().launch {
            val progressPercent = (progress * 100).toInt()
            progressIndicator?.progress = progressPercent
            progressText?.text = "$progressPercent%"
        }
    }
    
    fun dismiss() {
        downloadListener?.let { downloadManager?.removeListener(it) }
        dialog?.dismiss()
        dialog = null
    }
    
    fun setOnDismissListener(listener: () -> Unit) {
        dialog?.setOnDismissListener { listener() }
    }
}

object ChatRouter {

    fun startRun(context: Context, modelIdParam: String, destModelDir:String?, sessionId: String?) {
        Log.d(TAG, "startRun modelIdParam: $modelIdParam destModelDir: $destModelDir sessionId: $sessionId")
        val isDiffusion = ModelTypeUtils.isDiffusionModel(modelIdParam)
        var modelId:String? = modelIdParam
        val downloadManager = ModelDownloadManager.getInstance(context)
        if (!modelIdParam.startsWith("local") && !modelIdParam.startsWith("Builtin") && !modelIdParam.contains("/") && !isDiffusion) {
            modelId = ModelUtils.getValidModelIdFromName(downloadManager, modelIdParam)
        }
        Log.d(TAG, "modelId: $modelId")
        if (modelId == null) {
            Toast.makeText(context, context.getString(R.string.model_not_found, modelIdParam), Toast.LENGTH_LONG).show()
            return
        }
        
        // Check if this is a QNN model and handle QNN library download
        if (ModelTypeUtils.isQnnModel(modelId)) {
            Log.d(TAG, "QNN model detected: $modelId")
            if (QnnModule.deviceSupported()) {
                checkAndDownloadQnnLibs(context, modelId, destModelDir, sessionId, isDiffusion)
            } else {
                Log.w(TAG, "QNN model detected but device does not support QNN acceleration")
                Toast.makeText(context, context.getString(R.string.qnn_device_not_supported), Toast.LENGTH_LONG).show()
            }
            return
        }
        
        // Continue with normal flow for non-QNN models
        proceedToStartChat(context, modelId, destModelDir, sessionId, isDiffusion)
    }
    
    private fun checkAndDownloadQnnLibs(context: Context, modelId: String, destModelDir: String?, sessionId: String?, isDiffusion: Boolean) {
        // Check if QNN libs are already copied
        MainScope().launch {
            try {
                val qnnLibsDownloaded = withContext(Dispatchers.IO) {
                    QnnModule.isQnnLibsDownloaded(context)
                }
                
                if (qnnLibsDownloaded) {
                    Log.d(TAG, "QNN libs already downloaded, loading QNN libraries")
                    
                    // Load QNN libraries before starting chat
                    val loadSuccess = withContext(Dispatchers.IO) {
                        QnnModule.loadQnnLibs(context)
                    }
                    
                    if (loadSuccess) {
                        Log.d(TAG, "QNN libraries loaded successfully, proceeding to start chat")
                        proceedToStartChat(context, modelId, destModelDir, sessionId, isDiffusion)
                    } else {
                        Log.e(TAG, "Failed to load QNN libraries")
                        Toast.makeText(context, context.getString(R.string.qnn_libs_load_failed), Toast.LENGTH_LONG).show()
                    }
                    return@launch
                }
                
                // Show confirmation dialog
                showQnnDownloadConfirmationDialog(context, modelId, destModelDir, sessionId, isDiffusion)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error checking QNN libs status", e)
                Toast.makeText(context, context.getString(R.string.qnn_libs_download_failed), Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun showQnnDownloadConfirmationDialog(context: Context, modelId: String, destModelDir: String?, sessionId: String?, isDiffusion: Boolean) {
        val dialog = MaterialAlertDialogBuilder(context)
            .setTitle(context.getString(R.string.qnn_libs_download_title))
            .setMessage(context.getString(R.string.qnn_libs_download_message))
            .setPositiveButton(context.getString(R.string.download)) { _, _ ->
                downloadQnnLibsAndStartChat(context, modelId, destModelDir, sessionId, isDiffusion)
            }
            .setNegativeButton(context.getString(R.string.cancel)) { dialog, _ ->
                dialog.dismiss()
            }
            .create()
        
        dialog.show()
    }
    
    private fun showMaterial3ProgressDialog(
        context: Context,
        message: String,
        onStart: (dismissDialog: () -> Unit) -> Unit
    ): Dialog {
        // Create a custom layout for the progress dialog
        val inflater = LayoutInflater.from(context)
        val view = inflater.inflate(R.layout.dialog_progress_material3, null)
        
        val progressIndicator = view.findViewById<CircularProgressIndicator>(R.id.progress_indicator)
        val progressText = view.findViewById<TextView>(R.id.progress_text)
        
        // Set progress indicator to indeterminate mode for this simple dialog
        progressIndicator.isIndeterminate = true
        progressText.text = message
        
        val dialog = MaterialAlertDialogBuilder(context)
            .setView(view)
            .setCancelable(false)
            .create()
        
        dialog.show()
        
        // Start the operation
        onStart { dialog.dismiss() }
        
        return dialog
    }
    
    private fun downloadQnnLibsAndStartChat(context: Context, modelId: String, destModelDir: String?, sessionId: String?, isDiffusion: Boolean) {
        val progressDialog = ProgressDialog(
            context = context,
            modelId = null // We'll set this up in the coroutine
        )
        
        progressDialog.show()
        
        MainScope().launch {
            try {
                val libs = ModelRepository.getMarketDataSuspend().libs
                val qnnLibsItem = libs.find { it.modelName.equals("qnn_arm64_libs", ignoreCase = true) }
                qnnLibsItem?.modelId?.let { qnnModelId ->
                    progressDialog.setupDownloadListener(qnnModelId)
                }
                
                val downloadSuccess = withContext(Dispatchers.IO) {
                    val downloadManager = ModelDownloadManager.getInstance(context)
                    downloadManager.downloadQnnLibs()
                }
                
                progressDialog.dismiss()
                
                if (downloadSuccess) {
                    Log.d(TAG, "QNN libs downloaded successfully, loading QNN libraries")
                    
                    // Load QNN libraries before starting chat
                    val loadSuccess = withContext(Dispatchers.IO) {
                        QnnModule.loadQnnLibs(context)
                    }
                    
                    if (loadSuccess) {
                        Log.d(TAG, "QNN libraries loaded successfully, starting chat")
                        proceedToStartChat(context, modelId, destModelDir, sessionId, isDiffusion)
                    } else {
                        Log.e(TAG, "Failed to load QNN libraries")
                        Toast.makeText(context, context.getString(R.string.qnn_libs_load_failed), Toast.LENGTH_LONG).show()
                    }
                } else {
                    Log.e(TAG, "Failed to download QNN libs")
                    Toast.makeText(context, context.getString(R.string.qnn_libs_download_failed), Toast.LENGTH_LONG).show()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error downloading QNN libs", e)
                progressDialog.dismiss()
                Toast.makeText(context, context.getString(R.string.qnn_libs_download_failed), Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun proceedToStartChat(context: Context, modelId: String, destModelDir: String?, sessionId: String?, isDiffusion: Boolean) {
        val downloadManager = ModelDownloadManager.getInstance(context)
        if (isStopDownloadOnChatEnabled(context)) {
            downloadManager.pauseAllDownloads()
        }
        val configFilePath: String? = destModelDir ?: ModelUtils.getConfigPathForModel(modelId)
        val configFileExists = configFilePath?.let { File(it).exists() } ?: false
        if (!configFileExists) {
            Toast.makeText(
                context,
                context.getString(R.string.config_file_not_found, configFilePath?: modelId),
                Toast.LENGTH_LONG
            ).show()
            return
        }
        Log.d(TAG, "isDiffusion: ${isDiffusion}, configFilePath: $configFilePath")
        val intent = Intent(context, ChatActivity::class.java)
        intent.putExtra("chatSessionId", sessionId)
        if (isDiffusion) {
            intent.putExtra("diffusionDir", configFilePath)
        } else {
            intent.putExtra("configFilePath", configFilePath)
        }
        intent.putExtra("modelId", modelId)
        intent.putExtra("modelName", ModelUtils.getModelName(modelId))

        if (context !is Activity) {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        }
        context.startActivity(intent)
    }

    private const val TAG = "ChatRouter"

}