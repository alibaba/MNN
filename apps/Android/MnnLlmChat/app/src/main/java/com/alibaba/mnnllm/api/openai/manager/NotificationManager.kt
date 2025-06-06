package com.alibaba.mnnllm.api.openai.manager

import android.R
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import androidx.core.app.NotificationCompat
import timber.log.Timber

/**
 * 管理API服务相关通知的类
 *
 * 负责创建、更新和取消与API服务相关的系统通知
 * @property context Android上下文对象
 */
class ApiNotificationManager(private val context: Context) {
    private val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    
    companion object {
        /** 通知ID常量 */
        const val NOTIFICATION_ID = 1001
        
        /** 通知通道ID */
        private const val CHANNEL_ID = "api_service_channel"
        
        /** 通知通道名称 */
        private const val NOTIFICATION_CHANNEL_NAME = "API 服务通道"
    }

    init {
        // 初始化时创建通知通道
        initializeNotificationChannel()
    }

    /**
     * 初始化通知通道
     * 
     * 创建高优先级通知通道用于显示API服务状态
     */
    private fun initializeNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, 
            NOTIFICATION_CHANNEL_NAME, 
            NotificationManager.IMPORTANCE_HIGH
        )
        notificationManager.createNotificationChannel(channel)
    }

    /**
     * 构建API服务通知
     *
     * @param contentTitle 通知标题，默认为"API 服务运行中"
     * @param contentText 通知内容，默认为"正在监听端口：8080"
     * @return 构建好的Notification对象
     */
    fun buildNotification(
        contentTitle: String = "API 服务运行中", 
        contentText: String = "正在监听端口：8080"
    ): Notification {
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle(contentTitle)
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .setAutoCancel(false)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .build()
    }

    /**
     * 更新当前显示的通知
     *
     * @param contentTitle 新的通知标题
     * @param contentText 新的通知内容
     */
    fun updateNotification(contentTitle: String, contentText: String) {
        val notification = buildNotification(contentTitle, contentText)
        Timber.tag("NotificationManager").i("Updating notification: $contentTitle - $contentText")
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    /**
     * 取消当前显示的通知
     *
     * 如果取消失败会记录警告日志
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