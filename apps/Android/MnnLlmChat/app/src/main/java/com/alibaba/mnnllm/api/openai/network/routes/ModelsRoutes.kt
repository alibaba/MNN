package com.alibaba.mnnllm.api.openai.network.routes

import com.alibaba.mnnllm.api.openai.network.models.ModelData
import com.alibaba.mnnllm.api.openai.network.models.ModelPermission
import com.alibaba.mnnllm.api.openai.network.models.ModelsResponse
import com.alibaba.mnnllm.api.openai.network.services.MNNModelsService
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import java.util.UUID

/** * modelroutedefine * responsible fordefine /v1/models APIroute*/

/** * registermodelrelatedroute*/
fun Route.modelsRoutes() {
    val modelsService = MNNModelsService()

    get("/v1/models") {
        val traceId = UUID.randomUUID().toString()
        modelsService.getAvailableModels(call, traceId)
    }
}
