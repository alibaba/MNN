// Created by ruoyi.sjd on 2025/01/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.k2fsa.sherpa.mnn

import android.util.Log
import org.json.JSONObject
import java.io.File

/** * ASR modelconfigdataclass - correspond tosinglemodeldirectoryunder config.json*/
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

/** * ASR configmanager*/
object AsrConfigManager {
    private const val TAG = "AsrConfigManager"
    private const val CONFIG_FILE_NAME = "config.json"
    
    /** * fromspecifiedmodeldirectoryreadconfigfileandreturnmodelconfig * @param modelDir ASRmodeldirectorypath (such as: /path/to/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20) * @return OnlineModelConfig or null*/
    fun getModelConfigFromDirectory(modelDir: String): OnlineModelConfig? {
        return try {
            val configFile = File(modelDir, CONFIG_FILE_NAME)
            Log.d(TAG, "Looking for config file at: ${configFile.absolutePath}")
            
            if (!configFile.exists()) {
                Log.w(TAG, "Config file not found at ${configFile.absolutePath}, using fallback")
                return getFallbackConfig(modelDir)
            }
            
            val configContent = configFile.readText()
            Log.d(TAG, "Read config file content: ${configContent.take(200)}...")
            
            val asrConfig = parseConfigJson(configContent)
            Log.i(TAG, "Using ASR config from JSON: ${asrConfig.description ?: "Unknown model"}")
            return convertToOnlineModelConfig(modelDir, asrConfig)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error reading config file from $modelDir", e)
            getFallbackConfig(modelDir)
        }
    }
    
    /** * parseJSONconfigfile - singlemodelconfig*/
    private fun parseConfigJson(jsonContent: String): AsrModelConfig {
        val configJson = JSONObject(jsonContent)
        val transducerJson = configJson.getJSONObject("transducer")
        
        //parseoptionallanguagearray
        val languages = if (configJson.has("language")) {
            val languageArray = configJson.getJSONArray("language")
            val langList = mutableListOf<String>()
            for (i in 0 until languageArray.length()) {
                langList.add(languageArray.getString(i))
            }
            langList
        } else null
        
        val transducerConfig = TransducerConfig(
            encoder = transducerJson.getString("encoder"),
            decoder = transducerJson.getString("decoder"),
            joiner = transducerJson.getString("joiner")
        )
        
        //parseoptionalLMconfig
        val lmConfig = if (configJson.has("lm")) {
            val lmJson = configJson.getJSONObject("lm")
            LmConfig(
                model = lmJson.getString("model"),
                scale = lmJson.optDouble("scale", 0.5).toFloat()
            )
        } else null
        
        return AsrModelConfig(
            modelType = configJson.getString("modelType"),
            transducer = transducerConfig,
            tokens = configJson.getString("tokens"),
            language = languages,
            description = configJson.optString("description", null),
            lm = lmConfig
        )
    }
    
    /** * convertconfigconvertas OnlineModelConfig*/
    private fun convertToOnlineModelConfig(modelDir: String, config: AsrModelConfig): OnlineModelConfig {
        return OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = File(modelDir, config.transducer.encoder).absolutePath,
                decoder = File(modelDir, config.transducer.decoder).absolutePath,
                joiner = File(modelDir, config.transducer.joiner).absolutePath,
            ),
            tokens = File(modelDir, config.tokens).absolutePath,
            modelType = config.modelType,
        )
    }
    
    /** * getfallbackconfig (based onoriginal hardencodinglogic)*/
    private fun getFallbackConfig(modelDir: String): OnlineModelConfig? {
        Log.w(TAG, "Using fallback configuration for modelDir: $modelDir")
        
        //according tomodeldirectorynameinferconfig
        val dirName = File(modelDir).name.lowercase()
        
        return when {
            dirName.contains("bilingual") || dirName.contains("zh") -> {
                Log.d(TAG, "Using bilingual/Chinese fallback config")
                OnlineModelConfig(
                    transducer = OnlineTransducerModelConfig(
                        encoder = "$modelDir/encoder-epoch-99-avg-1.int8.mnn",
                        decoder = "$modelDir/decoder-epoch-99-avg-1.int8.mnn",
                        joiner = "$modelDir/joiner-epoch-99-avg-1.int8.mnn",
                    ),
                    tokens = "$modelDir/tokens.txt",
                    modelType = "zipformer",
                )
            }
            dirName.contains("en") -> {
                Log.d(TAG, "Using English fallback config")
                OnlineModelConfig(
                    transducer = OnlineTransducerModelConfig(
                        encoder = "$modelDir/encoder-epoch-99-avg-1.mnn",
                        decoder = "$modelDir/decoder-epoch-99-avg-1.mnn",
                        joiner = "$modelDir/joiner-epoch-99-avg-1.mnn",
                    ),
                    tokens = "$modelDir/tokens.txt",
                    modelType = "zipformer",
                )
            }
            else -> {
                Log.d(TAG, "Using default fallback config")
                OnlineModelConfig(
                    transducer = OnlineTransducerModelConfig(
                        encoder = "$modelDir/encoder-epoch-99-avg-1.mnn",
                        decoder = "$modelDir/decoder-epoch-99-avg-1.mnn",
                        joiner = "$modelDir/joiner-epoch-99-avg-1.mnn",
                    ),
                    tokens = "$modelDir/tokens.txt",
                    modelType = "zipformer",
                )
            }
        }
    }
    
    /** * fromspecifiedmodeldirectorygetlanguagemodelconfig * @param modelDir ASRmodeldirectorypath * @return OnlineLMConfig*/
    fun getLmConfigFromDirectory(modelDir: String): OnlineLMConfig {
        return try {
            val configFile = File(modelDir, CONFIG_FILE_NAME)
            
            if (!configFile.exists()) {
                Log.w(TAG, "Config file not found, using default LM config")
                return getDefaultLmConfig(modelDir)
            }
            
            val configContent = configFile.readText()
            val asrConfig = parseConfigJson(configContent)
            
            if (asrConfig.lm != null) {
                val fullModelPath = File(modelDir, asrConfig.lm.model).absolutePath
                Log.i(TAG, "Using LM config from JSON: ${asrConfig.lm.model} with scale ${asrConfig.lm.scale}")
                OnlineLMConfig(
                    model = fullModelPath,
                    scale = asrConfig.lm.scale
                )
            } else {
                Log.d(TAG, "No LM config found in configuration, using default")
                getDefaultLmConfig(modelDir)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error reading LM config from $modelDir", e)
            getDefaultLmConfig(modelDir)
        }
    }
    
    /** * getdefaultlanguagemodelconfig*/
    private fun getDefaultLmConfig(modelDir: String): OnlineLMConfig {
        //according tomodeldirectorynamedeterminewhethershoulduseLM
        val dirName = File(modelDir).name.lowercase()
        val shouldUseLm = dirName.contains("zh") || dirName.contains("bilingual") || dirName.contains("chinese")
        
        return if (shouldUseLm) {
            val lmPath = "$modelDir/with-state-epoch-99-avg-1.int8.onnx"
            if (File(lmPath).exists()) {
                Log.d(TAG, "Using default LM config with model: $lmPath")
                OnlineLMConfig(
                    model = lmPath,
                    scale = 0.5f
                )
            } else {
                Log.d(TAG, "LM file not found at $lmPath, using empty LM config")
                OnlineLMConfig()
            }
        } else {
            Log.d(TAG, "No LM needed for model: $dirName")
            OnlineLMConfig()
        }
    }

} 