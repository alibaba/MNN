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

    //according toconfigdecidewhetherenableCORS
    if (ApiServerConfig.isCorsEnabled(context)) {
        install(CORS) {
            val corsOrigins = ApiServerConfig.getCorsOrigins(context)
            //parseCORSsourceconfig
            if (corsOrigins.isNotEmpty()) {
                corsOrigins.split(",").forEach { origin ->
                    val trimmedOrigin = origin.trim()
                    if (trimmedOrigin.isNotEmpty()) {
                        allowHost(trimmedOrigin.removePrefix("http://").removePrefix("https://"))
                    }
                }
            } else {
                //if not availableconfigspecificsource，allowallhosts
                anyHost()
                allowMethod(HttpMethod.Post)
                allowMethod(HttpMethod.Get)
                allowHeader(HttpHeaders.Authorization)
                allowHeader(HttpHeaders.ContentType)
                allowHeader("x-api-key") //explicitlyallow x-api-key
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

    //according toconfigdecidewhetherenableauthentication
    install(Authentication) {
        bearer("auth-bearer") {
            realm = "Access to the '/' path"
            skipWhen {
                //when authenticationnotenabledwhen,skipauthenticationlogic
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


