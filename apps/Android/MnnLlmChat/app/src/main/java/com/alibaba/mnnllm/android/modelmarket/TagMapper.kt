package com.alibaba.mnnllm.android.modelmarket

object TagMapper {
    
    private var tagMap: Map<String, Tag> = emptyMap()
    
    // Initialize with default mappings (fallback)
    private val defaultTagMap = mapOf(
        "对话" to Tag("对话", "Chat"),
        "文本生成" to Tag("文本生成", "TextGeneration"),
        "多模态" to Tag("多模态", "Multimodal"),
        "图片理解" to Tag("图片理解", "ImageUnderstanding"),
        "视频理解" to Tag("视频理解", "VideoUnderstanding"),
        "音频理解" to Tag("音频理解", "AudioUnderstanding"),
        "代码生成" to Tag("代码生成", "CodeGeneration"),
        "数学" to Tag("数学", "Math"),
        "文档理解" to Tag("文档理解", "DocumentUnderstanding"),
        "文生图" to Tag("文生图", "TextToImage"),
        "深度思考" to Tag("深度思考", "DeepThinking")
    )
    
    init {
        tagMap = defaultTagMap
    }
    
    fun initializeFromData(modelMarketData: ModelMarketData) {
        val mappings = mutableMapOf<String, Tag>()
        modelMarketData.tagMappings.forEach { (chineseTag, tagInfo) ->
            mappings[chineseTag] = Tag(tagInfo.ch, tagInfo.key)
        }
        tagMap = if (mappings.isNotEmpty()) mappings else defaultTagMap
    }
    
    fun getTag(stringTag: String): Tag {
        return tagMap[stringTag] ?: Tag(stringTag, stringTag) // Fallback for unmapped tags
    }
    
    fun getAllTags(): List<Tag> {
        return tagMap.values.toList().sortedBy { it.getDisplayText() }
    }
    
    fun getQuickFilterTags(quickFilterTagNames: List<String>): List<Tag> {
        return quickFilterTagNames.mapNotNull { tagName ->
            tagMap[tagName]
        }
    }
} 