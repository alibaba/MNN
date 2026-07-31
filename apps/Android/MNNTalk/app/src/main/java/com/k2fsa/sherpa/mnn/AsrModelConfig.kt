package com.k2fsa.sherpa.mnn

import java.io.File
import org.json.JSONObject

private const val CONFIG_FILE_NAME = "config.json"

fun getModelConfigFromDirectory(modelDirectory: String): OnlineModelConfig? {
    val directory = File(modelDirectory)
    if (!directory.isDirectory) return null

    val config = runCatching {
        val json = JSONObject(File(directory, CONFIG_FILE_NAME).readText())
        val transducer = json.getJSONObject("transducer")
        OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = File(directory, transducer.getString("encoder")).absolutePath,
                decoder = File(directory, transducer.getString("decoder")).absolutePath,
                joiner = File(directory, transducer.getString("joiner")).absolutePath
            ),
            tokens = File(directory, json.getString("tokens")).absolutePath,
            modelType = json.optString("modelType", "zipformer")
        )
    }.getOrNull()

    return config ?: OnlineModelConfig(
        transducer = OnlineTransducerModelConfig(
            encoder = File(directory, "encoder-epoch-99-avg-1.int8.mnn").absolutePath,
            decoder = File(directory, "decoder-epoch-99-avg-1.int8.mnn").absolutePath,
            joiner = File(directory, "joiner-epoch-99-avg-1.int8.mnn").absolutePath
        ),
        tokens = File(directory, "tokens.txt").absolutePath,
        modelType = "zipformer"
    )
}

fun getOnlineLMConfigFromDirectory(modelDirectory: String): OnlineLMConfig {
    val directory = File(modelDirectory)
    val fromConfig = runCatching {
        val json = JSONObject(File(directory, CONFIG_FILE_NAME).readText())
        if (!json.has("lm")) return@runCatching null
        val lm = json.getJSONObject("lm")
        OnlineLMConfig(
            model = File(directory, lm.getString("model")).absolutePath,
            scale = lm.optDouble("scale", 0.5).toFloat()
        )
    }.getOrNull()
    if (fromConfig != null) return fromConfig

    val fallback = File(directory, "with-state-epoch-99-avg-1.int8.onnx")
    return if (fallback.isFile) OnlineLMConfig(fallback.absolutePath, 0.5f) else OnlineLMConfig()
}
