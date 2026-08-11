package com.k2fsa.sherpa.mnn

import android.util.Log
import org.json.JSONObject
import java.io.File

data class AsrModelConfig(
    val modelType: String,
    val transducer: TransducerConfig,
    val tokens: String,
    val language: List<String>? = null,
    val description: String? = null,
    val lm: LmConfig? = null
)

data class TransducerConfig(
    val encoder: String,
    val decoder: String,
    val joiner: String
)

data class LmConfig(
    val model: String,
    val scale: Float = 0.5f
)

object AsrConfigManager {
    private const val TAG = "AsrConfigManager"
    private const val CONFIG_FILE_NAME = "config.json"

    fun getModelConfigFromDirectory(modelDir: String): OnlineModelConfig? {
        return try {
            val configFile = File(modelDir, CONFIG_FILE_NAME)
            if (!configFile.exists()) {
                Log.w(TAG, "Config file not found at ${configFile.absolutePath}, using fallback")
                return getFallbackConfig(modelDir)
            }

            val asrConfig = parseConfigJson(configFile.readText())
            Log.i(TAG, "Using ASR config from JSON: ${asrConfig.description ?: "Unknown model"}")
            convertToOnlineModelConfig(modelDir, asrConfig)
        } catch (e: Exception) {
            Log.e(TAG, "Error reading config file from $modelDir", e)
            getFallbackConfig(modelDir)
        }
    }

    fun getLmConfigFromDirectory(modelDir: String): OnlineLMConfig {
        return try {
            val configFile = File(modelDir, CONFIG_FILE_NAME)
            if (!configFile.exists()) {
                return getDefaultLmConfig(modelDir)
            }

            val asrConfig = parseConfigJson(configFile.readText())
            if (asrConfig.lm != null) {
                OnlineLMConfig(
                    model = File(modelDir, asrConfig.lm.model).absolutePath,
                    scale = asrConfig.lm.scale
                )
            } else {
                getDefaultLmConfig(modelDir)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading LM config from $modelDir", e)
            getDefaultLmConfig(modelDir)
        }
    }

    private fun parseConfigJson(jsonContent: String): AsrModelConfig {
        val configJson = JSONObject(jsonContent)
        val transducerJson = configJson.getJSONObject("transducer")
        val languages = if (configJson.has("language")) {
            mutableListOf<String>().apply {
                val languageArray = configJson.getJSONArray("language")
                for (index in 0 until languageArray.length()) {
                    add(languageArray.getString(index))
                }
            }
        } else {
            null
        }

        val lmConfig = if (configJson.has("lm")) {
            val lmJson = configJson.getJSONObject("lm")
            LmConfig(
                model = lmJson.getString("model"),
                scale = lmJson.optDouble("scale", 0.5).toFloat()
            )
        } else {
            null
        }

        return AsrModelConfig(
            modelType = configJson.getString("modelType"),
            transducer = TransducerConfig(
                encoder = transducerJson.getString("encoder"),
                decoder = transducerJson.getString("decoder"),
                joiner = transducerJson.getString("joiner")
            ),
            tokens = configJson.getString("tokens"),
            language = languages,
            description = configJson.optString("description", null),
            lm = lmConfig
        )
    }

    private fun convertToOnlineModelConfig(modelDir: String, config: AsrModelConfig): OnlineModelConfig {
        return OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = File(modelDir, config.transducer.encoder).absolutePath,
                decoder = File(modelDir, config.transducer.decoder).absolutePath,
                joiner = File(modelDir, config.transducer.joiner).absolutePath
            ),
            tokens = File(modelDir, config.tokens).absolutePath,
            modelType = config.modelType
        )
    }

    private fun getFallbackConfig(modelDir: String): OnlineModelConfig {
        val dirName = File(modelDir).name.lowercase()
        val suffix = if (dirName.contains("bilingual") || dirName.contains("zh")) ".int8.mnn" else ".mnn"
        return OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = "$modelDir/encoder-epoch-99-avg-1$suffix",
                decoder = "$modelDir/decoder-epoch-99-avg-1$suffix",
                joiner = "$modelDir/joiner-epoch-99-avg-1$suffix"
            ),
            tokens = "$modelDir/tokens.txt",
            modelType = "zipformer"
        )
    }

    private fun getDefaultLmConfig(modelDir: String): OnlineLMConfig {
        val lmPath = "$modelDir/with-state-epoch-99-avg-1.int8.onnx"
        return if (File(lmPath).exists()) {
            OnlineLMConfig(model = lmPath, scale = 0.5f)
        } else {
            OnlineLMConfig()
        }
    }
}
