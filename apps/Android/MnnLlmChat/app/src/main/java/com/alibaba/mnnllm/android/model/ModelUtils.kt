// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.model

import android.annotation.SuppressLint
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.R
import java.io.File
import java.util.Locale
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import android.widget.Toast
import com.alibaba.mnnllm.android.modelsettings.ModelConfig

object ModelUtils {
    @Deprecated("Use ModelMarketItem.vendor field instead for market models")
    fun getVendor(modelName: String):String {
        val modelLower = modelName.lowercase(Locale.getDefault())
        if (modelLower.contains("deepseek")) {
            return ModelVendors.DeepSeek
        } else if (modelLower.contains("qwen") || modelLower.contains("qwq")) {
            return ModelVendors.Qwen
        } else if (modelLower.contains("llama") || modelLower.contains("mobilellm")) {
            return ModelVendors.Llama
        } else if (modelLower.contains("smo")) {
            return ModelVendors.Smo
        } else if (modelLower.contains("phi")) {
            return ModelVendors.Phi
        } else if (modelLower.contains("baichuan")) {
            return ModelVendors.Baichuan
        } else if (modelLower.contains("yi")) {
            return ModelVendors.Yi
        } else if (modelLower.contains("glm") || modelLower.contains("codegeex")) {
            return ModelVendors.Glm
        } else if (modelLower.contains("reader")) {
            return ModelVendors.Jina
        } else if (modelLower.contains("internlm")) {
            return ModelVendors.Internlm
        } else if (modelLower.contains("gemma")) {
            return ModelVendors.Gemma
        }  else if (modelLower.contains("mimo")) {
            return ModelVendors.Mimo
        } else if (modelLower.contains("fastvlm")) {
            return ModelVendors.FastVlm
        } else if (modelLower.contains("openelm")) {
            return ModelVendors.OpenElm
        } else {
            return ModelVendors.Others
        }
    }

    fun getDrawableId(modelName: String?): Int {
        if (modelName == null) {
            return 0
        }
        val modelLower = modelName.lowercase(Locale.getDefault())
        if (modelLower.contains("deepseek")) {
            return R.drawable.deepseek_icon
        } else if (modelLower.contains("qwen") || modelLower.contains("qwq")) {
            return R.drawable.qwen_icon
        } else if (modelLower.contains("llama") || modelLower.contains("mobilellm")) {
            return R.drawable.llama_icon
        } else if (modelLower.contains("smo")) {
            return R.drawable.smolm_icon
        } else if (modelLower.contains("phi")) {
            return R.drawable.phi_icon
        } else if (modelLower.contains("baichuan")) {
            return R.drawable.baichuan_icon
        } else if (modelLower.contains("yi")) {
            return R.drawable.yi_icon
        } else if (modelLower.contains("glm") || modelLower.contains("codegeex")) {
            return R.drawable.chatglm_icon
        } else if (modelLower.contains("reader")) {
            return R.drawable.jina_icon
        } else if (modelLower.contains("internlm")) {
            return R.drawable.internlm_icon
        } else if (modelLower.contains("gemma")) {
            return R.drawable.gemma_icon
        }
        return 0
    }

    @SuppressLint("DefaultLocale")
    fun generateBenchMarkString(metrics: HashMap<String, Any>): String {
        if (metrics.containsKey("total_timeus")) {
            return generateDiffusionBenchMarkString(metrics)
        }
        val promptLen = metrics.getOrDefault("prompt_len", 0L) as Long
        val decodeLen = metrics.getOrDefault("decode_len", 0L) as Long
        val prefillTimeUs = metrics.getOrDefault("prefill_time", 0L) as Long
        val decodeTimeUs = metrics.getOrDefault("decode_time", 0L) as Long
        var visionTimeUs = if (metrics.containsKey("vision_time")) metrics["vision_time"] as Long else 0L
        var audioTimeUs = if (metrics.containsKey("audio_time")) metrics["audio_time"] as Long else 0L
        if (promptLen == 0L || decodeLen == 0L) {
            return "generateBenchMarkString error"
        }
        // Calculate speeds in tokens per second
        var totalPrefillTimeUs = prefillTimeUs + visionTimeUs + audioTimeUs
        val promptSpeed =
            if ((totalPrefillTimeUs > 0)) (promptLen / (totalPrefillTimeUs / 1000000.0)) else 0.0
        val decodeSpeed = if ((decodeTimeUs > 0)) (decodeLen / (decodeTimeUs / 1000000.0)) else 0.0
        return String.format(
            "Prefill: %.2fs, %d tokens, %.2f tokens/s \nDecode: %.2fs, %d tokens, %.2f tokens/s",
            totalPrefillTimeUs.toFloat() / 1000000, promptLen, promptSpeed,
            decodeTimeUs.toFloat() / 1000000,decodeLen, decodeSpeed,
        )
    }

    @SuppressLint("DefaultLocale")
    fun generateDiffusionBenchMarkString(metrics: HashMap<String, Any>): String {
        val totalDuration = metrics["total_timeus"] as Long * 1.0 / 1000000.0
        return String.format("Generate time: %.2f s", totalDuration)
    }
    /**
     * you can add ModelItem.fromLocalModel("Qwen-Omni-7B", "/data/local/tmp/omni_test/model")
     * to load local models
     */
     val localModelList: MutableList<ModelItem> by lazy {
        val result = mutableListOf<ModelItem>()
        try {
            val modelsDir = File("/data/local/tmp/mnn_models/")
            if (modelsDir.exists() && modelsDir.isDirectory) {
                modelsDir.listFiles()?.forEach { modelDir ->
                    if (modelDir.isDirectory && File(modelDir, "config.json").exists()) {
                        val modelPath = modelDir.absolutePath
                        val modelId = "local/${modelPath}"
                        result.add(ModelItem.fromLocalModel(modelId, modelPath))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("ModelUtils", "Failed to load models from /data/local/tmp/mnn_models/", e)
        }
        result
    }

    private fun isQwen3(modelName: String):Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("qwen3")
    }

    fun isAudioModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("audio") || isOmni(modelName)
    }

    fun isMultiModalModel(modelName: String): Boolean {
        return isAudioModel(modelName) || isVisualModel(modelName) || isDiffusionModel(modelName) || isOmni(modelName)
    }

    fun isDiffusionModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("stable-diffusion")
    }

    @JvmStatic
    fun getModelName(modelId: String?): String? {
        if (modelId != null && modelId.contains("/")) {
            return modelId.substring(modelId.lastIndexOf("/") + 1)
        }
        return modelId
    }

    @JvmStatic
    fun generateSimpleTags(modelName: String, modelItem: ModelItem): ArrayList<String> {
        val splits = modelName.split("-".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        val tags = ArrayList<String>()
        val isDiffusion = isDiffusionModel(modelName)
        if (splits.size > 1 && !isDiffusion) {
            val brand = splits[0]
            tags.add(brand.lowercase(Locale.getDefault()))
        }
        for (i in 1 until splits.size) {
            val tag = splits[i]
            if (tag.lowercase(Locale.getDefault()).matches("^[\\\\.0-9]+[mb]$".toRegex())) {
                tags.add(tag.lowercase(Locale.getDefault()))
            }
        }
        if (isDiffusion) {
            tags.add("diffusion")
        } else {
            tags.add("text")
            if (isAudioModel(modelName)) {
                tags.add("audio")
            } else if (isVisualModel(modelName)) {
                tags.add("visual")
            }
        }
        if (modelItem.isLocal) {
            tags.add("local")
        } else {
            tags.add(ModelItem.sourceToTag(getSource(modelItem.modelId!!)!!))
        }
        return tags
    }

    fun isVisualModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("vl") || isOmni(modelName)
    }

    fun isR1Model(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("deepseek-r1")
    }

    fun safeModelId(modelId: String): String {
        return modelId.replace("/".toRegex(), "_")
    }

    fun isOmni(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("omni")
    }

    fun isSupportThinkingSwitch(modelName: String): Boolean {
        return isQwen3(modelName)
    }

    fun supportAudioOutput(modelName: String): Boolean {
        return isOmni(modelName)
    }

    /**
     * Check if the model is a TTS (Text-to-Speech) model
     */
    fun isTtsModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("bert-vits") ||
               modelName.lowercase(Locale.getDefault()).contains("tts")
    }

    /**
     * Check if the model is a TTS model based on tags
     */
    fun isTtsModelByTags(tags: List<String>): Boolean {
        return tags.any { it.equals("TTS", ignoreCase = true) }
    }

    /**
     * Check if the model is an ASR (Automatic Speech Recognition) model based on tags
     */
    fun isAsrModelByTags(tags: List<String>): Boolean {
        return tags.any { it.equals("ASR", ignoreCase = true) }
    }

    //split "Huggingface/taobao-mnn/Qwen-1.5B" to ["Huggingface", "taobao-mnn/Qwen-1.5B"]
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

    fun getRepositoryPath(modelId: String):String {
        return splitSource(modelId)[1]
    }

    fun getValidModelIdFromName(instance: ModelDownloadManager, modelName: String): String? {
        listOf(
            ModelSources.sourceModelers,
            ModelSources.sourceHuffingFace,
            ModelSources.sourceModelScope
        ).forEach { source ->
            val modelId = if (source == ModelSources.sourceHuffingFace)
                "$source/taobao-mnn/$modelName" else "$source/MNN/$modelName"
            val downloadFile = instance.getDownloadedFile(modelId)
            Log.d(TAG, "getValidModelIdFromName: modelId: $modelId downloadFile: ${downloadFile} exists: ${downloadFile?.exists()}")
            if (instance.getDownloadedFile(modelId) != null) {
                return modelId
            }
        }
        return null
    }

    fun getConfigPathForModel(modelId: String): String? {
        return if (isDiffusionModel(modelId)) {
            ModelDownloadManager.getInstance(ApplicationProvider.get())
                .getDownloadedFile(modelId)?.absolutePath
        } else {
            ModelConfig.getDefaultConfigFile(modelId)
        }
    }

    fun getConfigPathForModel(modelItem: ModelItem): String? {
        val modelId = modelItem.modelId!!
        val modelName = modelItem.modelName!!

        return if (isDiffusionModel(modelName)) {
            if (modelItem.isLocal) {
                // For local models, use the local path directly
                modelItem.localPath
            } else {
                ModelDownloadManager.getInstance(ApplicationProvider.get())
                    .getDownloadedFile(modelId)?.absolutePath
            }
        } else {
            if (modelItem.isLocal) {
                // For local models, look for config.json in the same directory
                val localPath = modelItem.localPath
                if (!localPath.isNullOrEmpty()) {
                    val configFile = File(localPath, "config.json")
                    if (configFile.exists()) {
                        configFile.absolutePath
                    } else {
                        // Fallback to default config
                        ModelConfig.getDefaultConfigFile(modelId)
                    }
                } else {
                    ModelConfig.getDefaultConfigFile(modelId)
                }
            } else {
                ModelConfig.getDefaultConfigFile(modelId)
            }
        }
    }

    fun openModelCard(context: Context, modelItem: ModelItem) {
        val modelId = modelItem.modelId ?: return
        
        try {
            val source = getSource(modelId)
            val repoPath = getRepositoryPath(modelId)
            
            if (source == null) {
                Toast.makeText(context, context.getString(R.string.unable_to_determine_model_source), Toast.LENGTH_SHORT).show()
                return
            }
            
            val url = when (source) {
                "HuggingFace" -> "https://huggingface.co/$repoPath"
                "ModelScope" -> "https://modelscope.cn/models/$repoPath"
                "Modelers" -> "https://www.modelers.cn/models/$repoPath"
                else -> {
                    Toast.makeText(context, context.getString(R.string.unknown_source, source), Toast.LENGTH_SHORT).show()
                    return
                }
            }
            
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            context.startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(context, context.getString(R.string.failed_to_open_model_card, e.message), Toast.LENGTH_SHORT).show()
        }
    }

    fun getModelSource(context: Context, modelId: String?): String? {
        return when {
            modelId == null -> null
            modelId.startsWith("HuggingFace/") || modelId.contains("taobao-mnn") -> context.getString(R.string.huggingface)
            modelId.startsWith("ModelScope/") -> context.getString(R.string.modelscope)
            modelId.startsWith("Modelers/") -> context.getString(R.string.modelers)
            else -> null
        }
    }

    private const val TAG = "ModelUtils"
}