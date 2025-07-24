// Created by ruoyi.sjd on 2025/01/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.k2fsa.sherpa.mnn

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * ASR 模型配置数据类 - 对应单个模型目录下的 config.json
 */
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

/**
 * ASR 配置管理器
 */
object AsrConfigManager {
    private const val TAG = "AsrConfigManager"
    private const val CONFIG_FILE_NAME = "config.json"
    
    /**
     * 从指定模型目录读取配置文件并返回模型配置
     * @param modelDir ASR模型目录路径 (如: /path/to/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20)
     * @return OnlineModelConfig 或 null
     */
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
    
    /**
     * 解析JSON配置文件 - 单个模型配置
     */
    private fun parseConfigJson(jsonContent: String): AsrModelConfig {
        val configJson = JSONObject(jsonContent)
        val transducerJson = configJson.getJSONObject("transducer")
        
        // 解析可选的语言数组
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
        
        // 解析可选的LM配置
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
    
    /**
     * 将配置转换为 OnlineModelConfig
     */
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
    
    /**
     * 获取回退配置（基于原有的硬编码逻辑）
     */
    private fun getFallbackConfig(modelDir: String): OnlineModelConfig? {
        Log.w(TAG, "Using fallback configuration for modelDir: $modelDir")
        
        // 根据模型目录名称推断配置
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
    
    /**
     * 从指定模型目录获取语言模型配置
     * @param modelDir ASR模型目录路径
     * @return OnlineLMConfig
     */
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
    
    /**
     * 获取默认的语言模型配置
     */
    private fun getDefaultLmConfig(modelDir: String): OnlineLMConfig {
        // 根据模型目录名称判断是否应该使用LM
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