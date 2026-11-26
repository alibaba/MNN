package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import timber.log.Timber
import java.util.Locale

object TagMapper {
    
    private var tagMap: Map<String, Tag> = emptyMap()

    private val capabilityKeys = listOf(
        "Chat", "Think", "Vision", "ImageGen", "Audio", "AudioGen", "Code", "Math", "Video", "Safety"
    ).associateBy { it.lowercase(Locale.ROOT) }

    private val taskAliases = mapOf(
        "text-generation" to "Chat",
        "text2text-generation" to "Chat",
        "textgeneration" to "Chat",
        "conversational" to "Chat",
        "image-text-to-text" to "Vision",
        "image-to-text" to "Vision",
        "visual-question-answering" to "Vision",
        "text-to-image" to "ImageGen",
        "automatic-speech-recognition" to "Audio",
        "audio-to-text" to "Audio",
        "asr" to "Audio",
        "text-to-speech" to "AudioGen",
        "text-to-audio" to "AudioGen",
        "tts" to "AudioGen",
        "video-text-to-text" to "Video"
    )

    fun getCapabilityTags(rawTags: List<String>): List<Tag> {
        return rawTags.mapNotNull { rawTag ->
            val value = rawTag.trim().let {
                if (it.startsWith("custom_tag:", ignoreCase = true)) it.substringAfter(':').trim() else it
            }
            val key = (tagMap[value]?.key ?: value).lowercase(Locale.ROOT)
            val capabilityKey = capabilityKeys[key] ?: taskAliases[key] ?: return@mapNotNull null
            tagMap[capabilityKey] ?: Tag(capabilityKey, capabilityKey)
        }.distinctBy { it.key }
    }
    
    fun initializeFromConfig(config: ModelMarketConfig) {
        val mappings = mutableMapOf<String, Tag>()
        config.tagTranslations.forEach { (key, chineseTranslation) ->
            mappings[chineseTranslation] = Tag(chineseTranslation, key)
            mappings[key] = Tag(chineseTranslation, key)
        }
        tagMap = mappings
        Timber.d("initializeFromConfig: loaded ${tagMap.size} tag mappings")
    }
    
    fun getTag(stringTag: String): Tag {
        if (stringTag.equals("local", ignoreCase = true)) {
            return Tag("本地", "local")
        }
        if (stringTag.equals("builtin", ignoreCase = true)) {
            return Tag("内置", "builtin")
        }
        getCapabilityTags(listOf(stringTag)).firstOrNull()?.let { return it }
        val tag = tagMap[stringTag] ?: Tag(stringTag, stringTag) // Fallback for unmapped tags
        if (tagMap[stringTag] == null && tagMap.isNotEmpty()) {
            Timber.w("getTag: '$stringTag' not found in tagMap (size=${tagMap.size}), using fallback")
        }
        return tag
    }
    
    fun getAllTags(): List<Tag> {
        return getCapabilityTags(tagMap.values.map { it.key }).sortedBy { it.getDisplayText() }
    }
    
    fun getQuickFilterTags(quickFilterTagNames: List<String>): List<Tag> {
        return getCapabilityTags(quickFilterTagNames)
    }

    fun getDisplayTagList(tagKeys: List<String>): List<String> {
        return getCapabilityTags(tagKeys).map { it.getDisplayText() }
    }

    /**
     * Context-aware version for ViewHolder bind. Uses the View's context for locale
     * so tags display correctly on Chinese devices without requiring scroll.
     */
    fun getDisplayTagList(tagKeys: List<String>, context: Context): List<String> {
        return getCapabilityTags(tagKeys).map { it.getDisplayText(context) }
    }
} 