package com.alibaba.mnnllm.api.openai.manager

import android.content.Context
import com.alibaba.mnnllm.api.openai.service.OpenAIService
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import timber.log.Timber

/**
 * API服务管理器，为UI层提供统一的API服务管理接口
 * 封装了服务启动、停止、状态查询等操作
 */
object ApiServiceManager {
    private val TAG = this::class.java.simpleName
    
    /**
     * 启动API服务
     * @param context 上下文，必须是ChatActivity实例
     * @return 是否成功启动
     */
    fun startApiService(context: Context): Boolean {
        return try {
            OpenAIService.startService(context)
            Timber.tag(TAG).i("API service start requested")
            true
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to start API service")
            false
        }
    }
    
    /**
     * 停止API服务
     * @param context 上下文
     * @return 是否成功停止
     */
    fun stopApiService(context: Context): Boolean {
        return try {
            OpenAIService.releaseService(context)
            Timber.tag(TAG).i("API service stop requested")
            true
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to stop API service")
            false
        }
    }
    
    /**
     * 重启API服务
     * @param context 上下文
     * @return 是否成功重启
     */
    fun restartApiService(context: Context): Boolean {
        return try {
            stopApiService(context)
            // 重置ServerEventManager状态
          //  ServerEventManager.getInstance().resetRuntimeState()
            // 给一点时间让服务完全停止
            Thread.sleep(500)
            startApiService(context)
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to restart API service")
            false
        }
    }
    
    /**
     * 检查API服务状态
     * 使用ServerEventManager获取准确的服务器状态
     */
    fun isApiServiceRunning(): Boolean {
        return try {
            ServerEventManager.getInstance().isServerRunning()
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Failed to check API service status")
            false
        }
    }
    
    /**
     * 检查API服务是否就绪
     */
    fun isApiServiceReady(): Boolean {
        return try {
            ServerEventManager.getInstance().isServerReady()
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Failed to check API service ready status")
            false
        }
    }
    
    /**
     * 获取服务器状态
     */
    fun getServerState(): ServerEventManager.ServerState {
        return ServerEventManager.getInstance().getCurrentState()
    }
    
    /**
     * 获取服务器信息
     */
    fun getServerInfo(): ServerEventManager.ServerInfo {
        return ServerEventManager.getInstance().getCurrentInfo()
    }
    
    /**
     * 获取API服务端口
     * @param context 上下文，用于获取配置
     * @return 服务端口
     */
    fun getApiServicePort(context: Context): Int {
        // 确保配置已初始化
        ApiServerConfig.initializeConfig(context)
        return ApiServerConfig.getPort(context)
    }
    
    /**
     * 获取API服务基础URL
     * @param context 上下文，用于获取配置
     * @return 服务基础URL
     */
    fun getApiServiceBaseUrl(context: Context): String {
        // 确保配置已初始化
        ApiServerConfig.initializeConfig(context)
        val port = ApiServerConfig.getPort(context)
        val ipAddress = ApiServerConfig.getIpAddress(context)
        return "http://$ipAddress:$port"
    }
}