package com.alibaba.mls.api.download

object ModelNameUtils {
    @JvmStatic
    fun getModelName(modelId: String?): String? {
        if (modelId != null && modelId.contains("/")) {
            return modelId.substring(modelId.lastIndexOf("/") + 1)
        }
        return modelId
    }

    fun safeModelId(modelId: String): String {
        return modelId.replace("/".toRegex(), "_")
    }

    // split "Huggingface/taobao-mnn/Qwen-1.5B" to ["Huggingface", "taobao-mnn/Qwen-1.5B"]
    fun splitSource(modelId: String): Array<String> {
        val firstSlashIndex = modelId.indexOf('/')
        if (firstSlashIndex == -1) {
            return arrayOf(modelId)
        }
        val source = modelId.substring(0, firstSlashIndex)
        val path = modelId.substring(firstSlashIndex + 1)
        return arrayOf(source, path)
    }

    fun getSource(modelId: String): String? {
        val firstSlashIndex = modelId.indexOf('/')
        if (firstSlashIndex == -1) {
            return null
        }
        return modelId.substring(0, firstSlashIndex)
    }

    fun getRepositoryPath(modelId: String): String {
        return splitSource(modelId)[1]
    }
}
