package com.alibaba.mnnllm.android.download

import android.content.Intent
import android.os.Build
import android.util.Log
import com.alibaba.mls.api.download.DownloadForegroundService
import com.alibaba.mnnllm.android.MnnLlmApplication
import java.util.LinkedHashMap

class DownloadForegroundServiceCoordinator(
    private val gateway: Gateway
) {
    interface Gateway {
        fun show(downloadCount: Int, modelName: String?): Boolean
        fun stop()
    }

    private val activeDownloads = LinkedHashMap<String, String?>()
    private var lastDispatchedState: NotificationState? = null

    fun onDownloadStateChanged(
        modelId: String,
        modelName: String?,
        isDownloading: Boolean
    ) {
        if (isDownloading) {
            activeDownloads[modelId] = modelName ?: activeDownloads[modelId]
        } else {
            activeDownloads.remove(modelId)
        }

        val nextState = activeDownloads.toNotificationState()
        if (nextState == null) {
            if (lastDispatchedState != null) {
                gateway.stop()
                lastDispatchedState = null
            }
            return
        }

        if (nextState == lastDispatchedState) {
            return
        }

        if (gateway.show(nextState.downloadCount, nextState.modelName)) {
            lastDispatchedState = nextState
        }
    }

    private fun LinkedHashMap<String, String?>.toNotificationState(): NotificationState? {
        if (isEmpty()) {
            return null
        }
        return NotificationState(
            downloadCount = size,
            modelName = values.firstOrNull()
        )
    }

    private data class NotificationState(
        val downloadCount: Int,
        val modelName: String?
    )
}

object DownloadForegroundServiceManager {
    private const val TAG = "DownloadFgServiceMgr"

    private val coordinator = DownloadForegroundServiceCoordinator(
        object : DownloadForegroundServiceCoordinator.Gateway {
            override fun show(downloadCount: Int, modelName: String?): Boolean {
                return runCatching {
                    val service = DownloadForegroundService.getInstance()
                    if (service != null) {
                        Log.d(
                            TAG,
                            "Updating download foreground service count=$downloadCount modelName=$modelName"
                        )
                        service.updateNotification(downloadCount, modelName)
                    } else {
                        val context = MnnLlmApplication.getAppContext()
                        val intent = Intent(context, DownloadForegroundService::class.java).apply {
                            putExtra(DownloadForegroundService.EXTRA_DOWNLOAD_COUNT, downloadCount)
                            putExtra(DownloadForegroundService.EXTRA_MODEL_NAME, modelName)
                        }
                        Log.d(
                            TAG,
                            "Starting download foreground service count=$downloadCount modelName=$modelName"
                        )
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            context.startForegroundService(intent)
                        } else {
                            context.startService(intent)
                        }
                    }
                    true
                }.getOrElse { error ->
                    Log.e(
                        TAG,
                        "Failed to show download foreground service count=$downloadCount modelName=$modelName",
                        error
                    )
                    false
                }
            }

            override fun stop() {
                runCatching {
                    val context = MnnLlmApplication.getAppContext()
                    Log.d(TAG, "Stopping download foreground service")
                    context.stopService(Intent(context, DownloadForegroundService::class.java))
                }.onFailure { error ->
                    Log.e(TAG, "Failed to stop download foreground service", error)
                }
            }
        }
    )

    @Synchronized
    fun onDownloadStateChanged(
        modelId: String,
        modelName: String?,
        isDownloading: Boolean
    ) {
        coordinator.onDownloadStateChanged(modelId, modelName, isDownloading)
    }
}
