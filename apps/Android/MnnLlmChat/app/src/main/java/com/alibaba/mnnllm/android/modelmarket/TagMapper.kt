package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import com.alibaba.mnnllm.android.utils.DeviceUtils

object TagMapper {
    
    private var tagMap: Map<String, Tag> = emptyMap()
    
    fun initializeFromConfig(config: ModelMarketConfig) {
        val mappings = mutableMapOf<String, Tag>()
        config.tagTranslations.forEach { (key, chineseTranslation) ->
            mappings[chineseTranslation] = Tag(chineseTranslation, key)
            mappings[key] = Tag(chineseTranslation, key)
        }
        tagMap = mappings
    }
    
    fun getTag(stringTag: String): Tag {
        if (stringTag.equals("local", ignoreCase = true)) {
            return Tag("本地", "local")
        }
        if (stringTag.equals("builtin", ignoreCase = true)) {
            return Tag("内置", "builtin")
        }
        return tagMap[stringTag] ?: Tag(stringTag, stringTag) // Fallback for unmapped tags
    }
    
    fun getAllTags(): List<Tag> {
        return tagMap.values.toList().distinctBy { it.key }.sortedBy { it.getDisplayText() }
    }
    
    fun getQuickFilterTags(quickFilterTagNames: List<String>): List<Tag> {
        return quickFilterTagNames.mapNotNull { tagName ->
            tagMap[tagName]
        }
    }

    fun getDisplayTagList(tagKeys: List<String>): List<String> {
        return tagKeys.map { getTag(it).getDisplayText() }
    }

    /**
     * Context-aware version for ViewHolder bind. Uses the View's context for locale
     * so tags display correctly on Chinese devices without requiring scroll.
     */
    fun getDisplayTagList(tagKeys: List<String>, context: Context): List<String> {
        val useChinese = DeviceUtils.isChinese(context)
        return tagKeys.map { getTag(it).getDisplayText(useChinese) }
    }
} 