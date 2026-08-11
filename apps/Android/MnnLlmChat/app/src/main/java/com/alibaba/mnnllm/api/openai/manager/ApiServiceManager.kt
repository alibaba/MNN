package com.alibaba.mnnllm.api.openai.manager

import android.content.Context
import com.alibaba.mnnllm.api.openai.network.compat.EndpointUrlBuilder
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.alibaba.mnnllm.api.openai.service.OpenAIService
import timber.log.Timber

/** * APIservicemanager,asUIlayerprovideunifiedAPIservicemanaginginterface * encapsulateservicestart, stop,statequeryetc.operations*/
object ApiServiceManager {
    private val TAG = this::class.java.simpleName

    /** * startAPIservice * @param context context，must beChatActivityinstance * @param modelId currentmodelID * @return whethersuccessstart*/
    fun startApiService(context: Context, modelId: String? = null): Boolean {
        return try {
            OpenAIService.startService(context, modelId)
            Timber.tag(TAG).i("API service start requested with modelId: $modelId")
            true
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to start API service")
            false
        }
    }

    /** * stopAPIservice * @param context context * @return whethersuccessstop*/
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

    /** * restartAPIservice * @param context context * @return whethersuccessrestart
     * Note: Do not call from main thread. Use lifecycleScope.launch { stop+delay+start } instead. */
    fun restartApiService(context: Context): Boolean {
        return try {
            stopApiService(context)
            Thread.sleep(500)
            startApiService(context)
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to restart API service")
            false
        }
    }

    /** * checkAPIservicestate * useServerEventManagergetaccurateservicestate*/
    fun isApiServiceRunning(): Boolean {
        return try {
            ServerEventManager.getInstance().isServerRunning()
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Failed to check API service status")
            false
        }
    }

    /** * checkAPIservicewhetherready*/
    fun isApiServiceReady(): Boolean {
        return try {
            ServerEventManager.getInstance().isServerReady()
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Failed to check API service ready status")
            false
        }
    }

    /** * getservicestate*/
    fun getServerState(): ServerEventManager.ServerState {
        return ServerEventManager.getInstance().getCurrentState()
    }

    /** * getserviceinfo*/
    fun getServerInfo(): ServerEventManager.ServerInfo {
        return ServerEventManager.getInstance().getCurrentInfo()
    }

    /** * getAPIserviceport * @param context context，forgetconfig * @return serviceport*/
    fun getApiServicePort(context: Context): Int {
        ApiServerConfig.initializeConfig(context)
        return ApiServerConfig.getPort(context)
    }

    /** * getAPIservicebaseURL * @param context context，forgetconfig * @return servicebaseURL*/
    fun getApiServiceBaseUrl(context: Context): String {
        ApiServerConfig.initializeConfig(context)
        val port = ApiServerConfig.getPort(context)
        val ipAddress = ApiServerConfig.getIpAddress(context)
        val useHttps = ApiServerConfig.useHttpsUrl(context)
        return EndpointUrlBuilder.buildBaseUrl(ipAddress, port, useHttps)
    }
}
