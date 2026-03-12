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
}
