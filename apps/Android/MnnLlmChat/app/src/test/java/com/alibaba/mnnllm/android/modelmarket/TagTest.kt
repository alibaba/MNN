package com.alibaba.mnnllm.android.modelmarket

import org.junit.Test
import org.junit.Assert.*
import org.junit.Before

class TagTest {

    @Before
    fun setup() {
        TagMapper.initializeFromConfig(
            ModelMarketConfig(
                version = "1",
                tagTranslations = mapOf(
                    "Chat" to "对话",
                    "TextGeneration" to "文本生成",
                    "Multimodal" to "多模态",
                    "Think" to "深度思考",
                    "Vision" to "图像理解",
                    "Video" to "视频理解",
                    "Audio" to "音频理解",
                    "AudioGen" to "音频生成",
                    "ImageGen" to "文生图",
                    "Code" to "代码",
                    "Math" to "数学",
                    "Safety" to "内容安全",
                    "mnn" to "推理框架",
                    "custom metadata" to "自定义信息"
                ),
                quickFilterTags = emptyList(),
                vendorOrder = emptyList(),
                llmModels = emptyList(),
                ttsModels = emptyList(),
                asrModels = emptyList(),
                libs = emptyList()
            )
        )
    }

    @Test
    fun testTagCreation() {
        val tag = Tag("对话", "Chat")
        assertEquals("对话", tag.ch)
        assertEquals("Chat", tag.key)
    }

    @Test
    fun testTagMapper() {
        val chatTag = TagMapper.getTag("对话")
        assertEquals("对话", chatTag.ch)
        assertEquals("Chat", chatTag.key)
        
        val unknownTag = TagMapper.getTag("Unknown Tag")
        assertEquals("Unknown Tag", unknownTag.ch)
        assertEquals("Unknown Tag", unknownTag.key)
    }

    @Test
    fun testGetAllTags() {
        val tags = TagMapper.getAllTags()
        assertTrue(tags.isNotEmpty())
        assertTrue(tags.any { it.key == "Chat" })
        assertEquals(
            setOf("Chat", "Think", "Vision", "ImageGen", "Audio", "AudioGen", "Code", "Math", "Video", "Safety"),
            tags.map { it.key }.toSet()
        )
    }

    @Test
    fun testDisplayTextFallback() {
        // In test environment, should fallback to key
        val tag = Tag("对话", "Chat")
        val displayText = tag.getDisplayText()
        // In test environment, this should return the key as fallback
        assertEquals("Chat", displayText)
    }

    // ==================== TDD: 中文 locale 标签显示测试 ====================

    @Test
    fun `getDisplayText with useChinese true should return Chinese text`() {
        val tag = Tag("对话", "Chat")
        val displayText = tag.getDisplayText(useChinese = true)
        assertEquals("对话", displayText)
    }

    @Test
    fun `getDisplayText with useChinese false should return English text`() {
        val tag = Tag("对话", "Chat")
        val displayText = tag.getDisplayText(useChinese = false)
        assertEquals("Chat", displayText)
    }

    @Test
    fun `TagMapper getTag by English key should return correct Tag`() {
        val tag = TagMapper.getTag("Chat")
        assertEquals("对话", tag.ch)
        assertEquals("Chat", tag.key)
    }

    @Test
    fun `TagMapper getTag by Chinese key should return correct Tag`() {
        val tag = TagMapper.getTag("对话")
        assertEquals("对话", tag.ch)
        assertEquals("Chat", tag.key)
    }

    @Test
    fun `TagMapper getDisplayTagList should return Chinese when useChinese is true`() {
        val tagKeys = listOf("Chat", "Multimodal")
        // 模拟中文环境：直接使用带 useChinese 参数的版本
        val displayTags = tagKeys.map { TagMapper.getTag(it).getDisplayText(useChinese = true) }
        assertEquals(listOf("对话", "多模态"), displayTags)
    }

    @Test
    fun `TagMapper getDisplayTagList should return English when useChinese is false`() {
        val tagKeys = listOf("Chat", "Multimodal")
        val displayTags = tagKeys.map { TagMapper.getTag(it).getDisplayText(useChinese = false) }
        assertEquals(listOf("Chat", "Multimodal"), displayTags)
    }

    @Test
    fun `local tag should display Chinese when useChinese is true`() {
        val tag = TagMapper.getTag("local")
        assertEquals("本地", tag.ch)
        assertEquals("local", tag.key)
        assertEquals("本地", tag.getDisplayText(useChinese = true))
    }

    @Test
    fun `builtin tag should display Chinese when useChinese is true`() {
        val tag = TagMapper.getTag("builtin")
        assertEquals("内置", tag.ch)
        assertEquals("builtin", tag.key)
        assertEquals("内置", tag.getDisplayText(useChinese = true))
    }

    @Test
    fun `capability tags normalize explicit tasks and deduplicate aliases`() {
        val rawTags = listOf(
            " text-generation ", "CHAT", "TextGeneration", "对话", "text2text-generation", "conversational",
            "image-text-to-text", "Vision", "image-to-text", "visual-question-answering",
            "text-to-image", "ImageGen", "automatic-speech-recognition", "audio-to-text", "ASR", "Audio",
            "text-to-speech", "text-to-audio", "TTS", "AudioGen", "video-text-to-text", "Video"
        )

        assertEquals(
            listOf("Chat", "Vision", "ImageGen", "Audio", "AudioGen", "Video"),
            TagMapper.getCapabilityTags(rawTags).map { it.key }
        )
        assertEquals("Chat", TagMapper.getTag("text-generation").key)
        assertEquals("Vision", TagMapper.getTag("image-text-to-text").key)
    }

    @Test
    fun `capability tags keep curated abilities and exclude technical metadata`() {
        val abilities = listOf("Think", "Vision", "ImageGen", "Audio", "AudioGen", "Code", "Math", "Video", "Safety")
        val metadata = listOf(
            "mnn", "MNN-Pocket", "int4", "hqq", "en", "custom metadata", "custom_tag:int4",
            "custom_tag:mnn-pocket", "license:apache-2.0", "NPU", "QNN", "ThinkingSwitch",
            "local", "builtin", "Multimodal", "text-to-video", "image-to-image", "", "  "
        )
        val rawTags = metadata + abilities + listOf("图像理解", "custom_tag:Think", " CUSTOM_TAG:Safety ")

        assertEquals(abilities, TagMapper.getCapabilityTags(rawTags).map { it.key })
        assertEquals(emptyList<Tag>(), TagMapper.getCapabilityTags(metadata))
        assertEquals(emptyList<String>(), TagMapper.getDisplayTagList(metadata))
        assertEquals(abilities, TagMapper.getQuickFilterTags(rawTags).map { it.key })
    }

    @Test
    fun `capabilities work without CDN translations`() {
        TagMapper.initializeFromConfig(
            ModelMarketConfig("1", emptyMap(), emptyList(), emptyList(), emptyList(), emptyList(), emptyList(), emptyList())
        )

        assertEquals(
            listOf("Chat", "Vision", "Safety"),
            TagMapper.getCapabilityTags(listOf("text-generation", "image-text-to-text", "Safety", "mnn")).map { it.key }
        )
        assertEquals(listOf("Chat"), TagMapper.getQuickFilterTags(listOf("text-generation", "Chat", "int4")).map { it.key })
    }

    @Test
    fun `cached market items expose normalized filter keys without changing raw metadata`() {
        fun cachedItem(vendor: String) = com.google.gson.Gson().fromJson(
            """{"modelName":"MNN-Pocket-0.3B-MNN","vendor":"$vendor","size_gb":0.3,
               |"tags":["mnn","int4","text-generation","Chat","image-text-to-text","Vision","Safety"],
               |"categories":["text-generation"],"sources":{},"extra_tags":["ThinkingSwitch"]}""".trimMargin(),
            ModelMarketItem::class.java
        )
        val item = cachedItem("MNN-Pocket")
        val originalTags = item.tags.toList()
        val keys = item.structuredTags.map { it.key }

        assertEquals(listOf("Chat", "Vision", "Safety"), keys)
        assertTrue(keys.containsAll(TagMapper.getQuickFilterTags(listOf("text-generation", "Vision")).map { it.key }))
        assertEquals(originalTags, item.tags)
        assertEquals(listOf("text-generation"), item.categories)
        assertEquals(listOf("ThinkingSwitch"), item.extraTags)
        assertEquals("MNN-Pocket", item.vendor)
        assertEquals("MNN", item.displayVendor)
        assertNull(item.currentSource)
        assertEquals("Qwen", cachedItem("Qwen").displayVendor)
        assertEquals("MNN", cachedItem("Other").displayVendor)
    }

    @Test
    fun `cached Pocket models recover MNN vendor without overwriting curated vendors`() {
        fun cachedItem(name: String, vendor: String) = com.google.gson.Gson().fromJson(
            """{"modelName":"$name","vendor":"$vendor","size_gb":0.3,
               |"tags":[],"categories":[],"sources":{"ModelScope":"MNN/$name"}}""".trimMargin(),
            ModelMarketItem::class.java
        )
        val item = cachedItem("MNN-Pocket-0.3B-MNN", "Other")

        assertEquals("MNN", item.displayVendor)
        assertEquals("Other", item.vendor)
        assertNull(item.extraTags)
        assertNull(item.currentSource)
        for (name in listOf("MNN-Pocket-0.3B-MNN", "mnn-pocket-1b-qat", "MNN-Pocket-0.3B-DFlash")) {
            for (vendor in listOf("", "  ", "Other", "other", "MNN-Pocket", "mnn-pocket")) {
                assertEquals("MNN", cachedItem(name, vendor).displayVendor)
            }
            for (vendor in listOf("Qwen", "CuratedVendor", "MNN")) {
                assertEquals(vendor, cachedItem(name, vendor).displayVendor)
            }
        }
    }

    @Test
    fun `non Pocket models do not become MNN vendors by organization or substring`() {
        val item = ModelMarketItem(
            modelName = "MNN-Unknown-MNN", vendor = "Other", sizeB = 1.0,
            tags = emptyList(), categories = emptyList(), sources = mapOf("ModelScope" to "MNN/MNN-Unknown-MNN")
        )

        for (name in listOf("MNN-Unknown-MNN", "Qwen3-MNN-Pocket-MNN", "PocketBook-not-ours-MNN")) {
            for (vendor in listOf("", "Other", "MNN-Pocket", "Qwen")) {
                assertEquals(vendor, item.copy(modelName = name, vendor = vendor).displayVendor)
            }
        }
    }

    @Test
    fun `unknown tag should fallback to original string for both Chinese and English`() {
        val unknownTag = TagMapper.getTag("SomeUnknownTag")
        assertEquals("SomeUnknownTag", unknownTag.ch)
        assertEquals("SomeUnknownTag", unknownTag.key)
        assertEquals("SomeUnknownTag", unknownTag.getDisplayText(useChinese = true))
        assertEquals("SomeUnknownTag", unknownTag.getDisplayText(useChinese = false))
    }
}
