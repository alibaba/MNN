package com.alibaba.mnnllm.android.modelmarket

object TagMapper {
    
    private var tagMap: Map<String, Tag> = emptyMap()
    
    fun initializeFromData(modelMarketData: ModelMarketData) {
        val mappings = mutableMapOf<String, Tag>()
        modelMarketData.tagTranslations.forEach { (key, chineseTranslation) ->
            mappings[chineseTranslation] = Tag(chineseTranslation, key)
            mappings[key] = Tag(chineseTranslation, key)
        }
        tagMap = mappings
    }
    
    fun getTag(stringTag: String): Tag {
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
} 