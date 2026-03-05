package com.alibaba.mnnllm.api.openai.network.auth

import android.content.Context
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond

object ApiAuthUtils {
    fun isAuthorized(call: ApplicationCall, context: Context): Boolean {
        if (!ApiServerConfig.isAuthEnabled(context)) {
            return true
        }

        val configuredApiKey = ApiServerConfig.getApiKey(context)
        val bearerToken = call.request.headers[HttpHeaders.Authorization]
            ?.removePrefix("Bearer ")
            ?.trim()
        val xApiKey = call.request.headers["x-api-key"]?.trim()

        return bearerToken == configuredApiKey || xApiKey == configuredApiKey
    }

    suspend fun respondUnauthorized(call: ApplicationCall) {
        call.respond(
            HttpStatusCode.Unauthorized,
            mapOf(
                "type" to "error",
                "error" to mapOf(
                    "type" to "authentication_error",
                    "message" to "Invalid API key"
                )
            )
        )
    }
}
