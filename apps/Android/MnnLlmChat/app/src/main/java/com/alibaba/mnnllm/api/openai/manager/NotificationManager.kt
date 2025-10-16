package com.alibaba.mnnllm.api.openai.manager

import android.R
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
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
     * Build API service notification with IP address and action buttons
     *
     * @param contentTitle Notification title, uses default from string resources if not provided
     * @param contentText Notification content, uses default from string resources if not provided
     * @param port Server port number
     * @return Built Notification object
     */
    fun buildNotification(
        contentTitle: String? = null, 
        contentText: String? = null,
        port: Int = 8080
    ): Notification {
        val title = contentTitle ?: context.getString(com.alibaba.mnnllm.android.R.string.api_service_running)
        val ipAddress = ApiServerConfig.getIpAddress(context)
        val url = "http://$ipAddress:$port"
        val text = if (contentText.isNullOrBlank()) {
            context.getString(com.alibaba.mnnllm.android.R.string.api_service_running_on, ipAddress, port)
        } else {
            contentText
        }
        
        Timber.tag("ApiNotificationManager").i("Building notification - Config IP: $ipAddress, Port: $port, Text: $text")
        
        //createstopservice PendingIntent
        val stopIntent = Intent(ApiServiceActionReceiver.ACTION_STOP_SERVICE).apply {
            `package` = context.packageName
        }
        Timber.tag("ApiNotificationManager").i("Creating stop intent with action: ${ApiServiceActionReceiver.ACTION_STOP_SERVICE}")
        Timber.tag("ApiNotificationManager").i("Stop intent package: ${context.packageName}")
        val stopPendingIntent = PendingIntent.getBroadcast(
            context, 
            1001, 
            stopIntent, 
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        Timber.tag("ApiNotificationManager").i("Stop PendingIntent created: $stopPendingIntent")
        
        //createcopy URL PendingIntent
        val copyIntent = Intent(ApiServiceActionReceiver.ACTION_COPY_URL).apply {
            `package` = context.packageName
            putExtra(ApiServiceActionReceiver.EXTRA_URL, url)
        }
        Timber.tag("ApiNotificationManager").i("Creating copy intent with action: ${ApiServiceActionReceiver.ACTION_COPY_URL}, URL: $url")
        Timber.tag("ApiNotificationManager").i("Copy intent package: ${context.packageName}")
        val copyPendingIntent = PendingIntent.getBroadcast(
            context, 
            System.currentTimeMillis().toInt(), 
            copyIntent, 
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        Timber.tag("ApiNotificationManager").i("Copy PendingIntent created: $copyPendingIntent")
        
        //createopentestpage PendingIntent
        val testIntent = Intent(ApiServiceActionReceiver.ACTION_TEST_PAGE).apply {
            `package` = context.packageName
            putExtra(ApiServiceActionReceiver.EXTRA_URL, url)
        }
        Timber.tag("ApiNotificationManager").i("Creating test intent with action: ${ApiServiceActionReceiver.ACTION_TEST_PAGE}, URL: $url")
        Timber.tag("ApiNotificationManager").i("Test intent package: ${context.packageName}")
        val testPendingIntent = PendingIntent.getBroadcast(
            context, 
            (System.currentTimeMillis() + 1).toInt(), 
            testIntent, 
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        Timber.tag("ApiNotificationManager").i("Test PendingIntent created: $testPendingIntent")
        
        //createclicknotificationwhenopen mainActivityPendingIntent
        val mainActivityIntent = Intent(context, com.alibaba.mnnllm.android.main.MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val mainActivityPendingIntent = PendingIntent.getActivity(
            context,
            (System.currentTimeMillis() + 2).toInt(),
            mainActivityIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        Timber.tag("ApiNotificationManager").i("MainActivity PendingIntent created: $mainActivityPendingIntent")
        
        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .setAutoCancel(false)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            // .addAction(
            //     R.drawable.ic_dialog_alert,
            //     context.getString(com.alibaba.mnnllm.android.R.string.api_service_stop),
            //     stopPendingIntent
            // )
            .addAction(
                R.drawable.ic_dialog_info,
                context.getString(com.alibaba.mnnllm.android.R.string.api_service_copy_url),
                copyPendingIntent
            )
            .addAction(
                R.drawable.ic_dialog_info,
                context.getString(com.alibaba.mnnllm.android.R.string.api_service_test_page),
                testPendingIntent
            )
            .build()
        
        Timber.tag("ApiNotificationManager").i("Notification built successfully with ${notification.actions?.size ?: 0} actions")
        notification.actions?.forEachIndexed { index, action ->
            Timber.tag("ApiNotificationManager").i("Action $index: ${action.title}, Intent: ${action.actionIntent}")
        }
        
        return notification
    }

    /**
     * Update current displayed notification
     *
     * @param contentTitle New notification title
     * @param contentText New notification content
     * @param port Server port number
     */
    fun updateNotification(contentTitle: String, contentText: String, port: Int = 8080) {
        Timber.tag("ApiNotificationManager").i("updateNotification called - Title: $contentTitle, Text: $contentText, Port: $port")
        val notification = buildNotification(contentTitle, contentText, port)
        Timber.tag("ApiNotificationManager").i("Updating notification: $contentTitle - $contentText")
        notificationManager.notify(NOTIFICATION_ID, notification)
        Timber.tag("ApiNotificationManager").i("Notification updated with ID: $NOTIFICATION_ID")
    }

    /**
     * Cancel current displayed notification
     *
     * If cancellation fails, it will record a warning log
     */
    fun cancelNotification() {
        try {
            notificationManager.cancel(NOTIFICATION_ID)
            Timber.tag("ApiNotificationManager").i("Notification cancelled")
        } catch (e: Exception) {
            Timber.tag("ApiNotificationManager").w("Failed to cancel notification: ${e.message}")
        }
    }
    
}