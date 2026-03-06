package com.alibaba.mnnllm.api.openai.network.compat

object EndpointUrlBuilder {
    fun buildBaseUrl(host: String, port: Int, useHttps: Boolean): String {
        val scheme = if (useHttps) "https" else "http"
        return "$scheme://$host:$port"
    }
}
