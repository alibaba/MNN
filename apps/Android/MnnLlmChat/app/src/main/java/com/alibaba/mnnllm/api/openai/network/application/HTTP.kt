package com.alibaba.mnnllm.api.openai.network.application

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

fun Application.configureHTTP() {
    install(DoubleReceive)

    install(CORS) {
        allowHost("localhost")
        allowHost("0.0.0.0:8080")
        allowMethod(HttpMethod.Post)
        allowHeader(HttpHeaders.Authorization)
        allowHeader(HttpHeaders.ContentType)
        allowCredentials = true
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

    /***
    val validPassword = "12345678"

    install(Authentication) {
        basic(name = "openai-auth") {
            realm = "OpenAI API"
            validate { credentials ->
                if (credentials.password == validPassword) {
                    // 允许任何用户名，密码正确即可
                    UserIdPrincipal(credentials.name)
                } else {
                    null
                }
            }
        }
    }
    install(RoutingRoot) {
        // 示例：在某个路由下启用认证
        authenticate("openai-auth") {
            get("/v1/models") {
                call.respondText("Valid models list", contentType = ContentType.Text.Plain)
            }
        }
    }
    ***/

}


