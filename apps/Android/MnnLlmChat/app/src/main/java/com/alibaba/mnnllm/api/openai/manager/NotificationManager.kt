package com.alibaba.mnnllm.api.openai.manager

import android.R
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import androidx.core.app.NotificationCompat
import timber.log.Timber

/**
 * Class for managing API service related notifications
 *
 * Responsible for creating, updating and canceling system notifications related to API service
 * @property context Android context object
 */
class ApiNotificationManager(private val context: Context) {
    private val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    
    companion object {
        /** Notification ID constant */
        const val NOTIFICATION_ID = 1001
        
        /** Notification channel ID */
        private const val CHANNEL_ID = "api_service_channel"
    }

    init {
        // Initialize notification channel during initialization
        initializeNotificationChannel()
    }

    /**
     * Initialize notification channel
     * 
     * Create high priority notification channel for displaying API service status
     */
    private fun initializeNotificationChannel() {
        val channelName = context.getString(com.alibaba.mnnllm.android.R.string.api_service_channel_name)
        val channel = NotificationChannel(
            CHANNEL_ID, 
            channelName, 
            NotificationManager.IMPORTANCE_HIGH
        )
        notificationManager.createNotificationChannel(channel)
    }

    /**
     * Build API service notification
     *
     * @param contentTitle Notification title, uses default from string resources if not provided
     * @param contentText Notification content, uses default from string resources if not provided
     * @return Built Notification object
     */
    fun buildNotification(
        contentTitle: String? = null, 
        contentText: String? = null
    ): Notification {
        val title = contentTitle ?: context.getString(com.alibaba.mnnllm.android.R.string.api_service_running)
        val text = contentText ?: context.getString(com.alibaba.mnnllm.android.R.string.server_running_message)
        
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .setAutoCancel(false)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .build()
    }

    /**
     * Update current displayed notification
     *
     * @param contentTitle New notification title
     * @param contentText New notification content
     */
    fun updateNotification(contentTitle: String, contentText: String) {
        val notification = buildNotification(contentTitle, contentText)
        Timber.tag("NotificationManager").i("Updating notification: $contentTitle - $contentText")
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    /**
     * Cancel current displayed notification
     *
     * If cancellation fails, it will record a warning log
     */
    fun cancelNotification() {
        try {
            notificationManager.cancel(NOTIFICATION_ID)
            Timber.tag("NotificationManager").i("Notification cancelled")
        } catch (e: Exception) {
            Timber.tag("NotificationManager").w("Failed to cancel notification: ${e.message}")
        }
    }
}