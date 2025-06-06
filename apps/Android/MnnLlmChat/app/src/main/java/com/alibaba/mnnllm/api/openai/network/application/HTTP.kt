package com.alibaba.mnnllm.api.openai.network.application

import android.content.Context
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.doublereceive.DoubleReceive
import io.ktor.server.plugins.requestvalidation.RequestValidation
import kotlinx.serialization.json.Json
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import io.ktor.server.auth.Authentication
import io.ktor.server.auth.UserIdPrincipal
import io.ktor.server.auth.bearer



fun Application.configureHTTP(context: Context) {
    install(DoubleReceive)

    // 根据配置决定是否启用CORS
    if (ApiServerConfig.isCorsEnabled(context)) {
        install(CORS) {
            val corsOrigins = ApiServerConfig.getCorsOrigins(context)
            // 解析CORS来源配置
            if (corsOrigins.isNotEmpty()) {
                corsOrigins.split(",").forEach { origin ->
                    val trimmedOrigin = origin.trim()
                    if (trimmedOrigin.isNotEmpty()) {
                        allowHost(trimmedOrigin.removePrefix("http://").removePrefix("https://"))
                    }
                }
            } else {
                // 如果没有配置具体来源，允许所有主机
                anyHost()
                allowMethod(HttpMethod.Post)
                allowMethod(HttpMethod.Get)
                allowHeader(HttpHeaders.Authorization)
                allowHeader(HttpHeaders.ContentType)
                allowHeader("x-api-key") // 显式允许 x-api-key
                allowCredentials = true
            }
        }
    }

    install(RequestValidation)

    install(ContentNegotiation) {
        json(
            Json {
                prettyPrint = false
                ignoreUnknownKeys = true
            }
        )

    }

    // 根据配置决定是否启用认证
    install(Authentication) {
        bearer("auth-bearer") {
            realm = "Access to the '/' path"
            skipWhen {
                // 当认证未启用时，跳过认证逻辑
                !ApiServerConfig.isAuthEnabled(context)
            }
            authenticate { tokenCredential ->
                val configuredApiKey = ApiServerConfig.getApiKey(context)
                if (tokenCredential.token == configuredApiKey) {
                    UserIdPrincipal("api-user")
                } else {
                    null
                }
            }
        }
    }



}


