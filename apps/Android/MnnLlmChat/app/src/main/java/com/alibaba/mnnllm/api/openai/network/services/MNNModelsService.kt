package com.alibaba.mnnllm.api.openai.network.services

import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.models.ModelData
import com.alibaba.mnnllm.api.openai.network.models.ModelPermission
import com.alibaba.mnnllm.api.openai.network.models.ModelsResponse
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.MnnLlmApplication
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.response.respond
import kotlinx.coroutines.runBlocking
import timber.log.Timber

/**
 * MNN 模型服务
 * 负责处理模型相关的业务逻辑
 */
class MNNModelsService {
    private val logger = ChatLogger()

    suspend fun getAvailableModels(call: io.ktor.server.application.ApplicationCall, traceId: String) {
        try {
            logger.logRequestStart(traceId, call)
            
            val context = MnnLlmApplication.getAppContext()
            val availableModels = runBlocking {
                ModelListManager.loadAvailableModels(context)
            }
            val modelDataList = availableModels.map { modelWrapper ->
                ModelData(
                    id = modelWrapper.modelItem.modelId ?: "unknown",
                    created = System.currentTimeMillis() / 1000, // Unix timestamp
                    permission = listOf(
                        ModelPermission(
                            id = "modelperm-${modelWrapper.modelItem.modelId}",
                            created = System.currentTimeMillis() / 1000
                        )
                    )
                )
            }
            val response = ModelsResponse(data = modelDataList)
            
            call.respond(response)
            logger.logInfo(traceId, "Models list returned successfully")
            
        } catch (e: Exception) {
            Timber.tag("MNNModelsService").e(e, "Error getting available models")
            logger.logError(traceId, e, "Failed to get available models")
            call.respond(HttpStatusCode.InternalServerError, mapOf("error" to "Failed to get available models"))
        }
    }
}
