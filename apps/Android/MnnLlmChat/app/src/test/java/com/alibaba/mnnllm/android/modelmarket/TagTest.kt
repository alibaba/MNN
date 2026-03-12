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
                    "Multimodal" to "多模态"
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
        assertTrue(tags.any { it.key == "TextGeneration" })
        assertTrue(tags.any { it.key == "Multimodal" })
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
    fun `unknown tag should fallback to original string for both Chinese and English`() {
        val unknownTag = TagMapper.getTag("SomeUnknownTag")
        assertEquals("SomeUnknownTag", unknownTag.ch)
        assertEquals("SomeUnknownTag", unknownTag.key)
        assertEquals("SomeUnknownTag", unknownTag.getDisplayText(useChinese = true))
        assertEquals("SomeUnknownTag", unknownTag.getDisplayText(useChinese = false))
    }
}
