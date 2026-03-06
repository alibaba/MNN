package com.alibaba.mnnllm.api.openai.network.auth

import android.content.Context
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.header
import io.ktor.server.response.respond
import io.ktor.server.response.respondText

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
        call.response.header(HttpHeaders.WWWAuthenticate, "Bearer")
        call.respondText(
            text = """{"error":{"type":"authentication_error","message":"Invalid API key"}}""",
            contentType = ContentType.Application.Json,
            status = HttpStatusCode.Unauthorized
        )
    }
}
