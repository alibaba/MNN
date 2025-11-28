package com.alibaba.mnnllm.api.openai.network.services

import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.models.ModelData
import com.alibaba.mnnllm.api.openai.network.models.ModelPermission
import com.alibaba.mnnllm.api.openai.network.models.ModelsResponse
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.api.openai.manager.CurrentModelManager
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.response.respond
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import timber.log.Timber

/** * MNN modelservice * responsible forprocessmodelrelatedbusinesslogic*/
class MNNModelsService {
    private val logger = ChatLogger()

    suspend fun getAvailableModels(call: io.ktor.server.application.ApplicationCall, traceId: String) {
        try {
            logger.logRequestStart(traceId, call)
            
            val currentModelId = CurrentModelManager.getCurrentModelId()
            if (currentModelId == null) {
                Timber.tag("MNNModelsService").w("No current model ID available")
                logger.logError(traceId, Exception("No current model ID available"), "No current model ID available")
                call.respond(HttpStatusCode.InternalServerError, mapOf("error" to "No current model available"))
                return
            }
            
            val availableModels = ModelListManager.getCurrentModels()
            //onlyreturncurrentcurrentlyusemodel
            val currentModelWrapper = availableModels?.find { 
                it.modelItem.modelId == currentModelId 
            }
            
            val modelDataList = if (currentModelWrapper != null) {
                listOf(ModelData(
                    id = currentModelWrapper.modelItem.modelId ?: "unknown",
                    created = System.currentTimeMillis() / 1000, // Unix timestamp
                    permission = listOf(
                        ModelPermission(
                            id = "modelperm-${currentModelWrapper.modelItem.modelId}",
                            created = System.currentTimeMillis() / 1000
                        )
                    )
                ))
            } else {
                //ifcannot findcurrentmodel，returnonedefaultmodeldata
                listOf(ModelData(
                    id = currentModelId,
                    created = System.currentTimeMillis() / 1000,
                    permission = listOf(
                        ModelPermission(
                            id = "modelperm-$currentModelId",
                            created = System.currentTimeMillis() / 1000
                        )
                    )
                ))
            }
            
            val response = ModelsResponse(data = modelDataList)
            
            call.respond(response)
            logger.logInfo(traceId, "Current model returned successfully: $currentModelId")
            
        } catch (e: Exception) {
            Timber.tag("MNNModelsService").e(e, "Error getting current model")
            logger.logError(traceId, e, "Failed to get current model")
            call.respond(HttpStatusCode.InternalServerError, mapOf("error" to "Failed to get current model"))
        }
    }
}
