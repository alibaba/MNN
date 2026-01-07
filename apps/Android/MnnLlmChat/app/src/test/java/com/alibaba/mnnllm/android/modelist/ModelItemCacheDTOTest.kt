package com.alibaba.mnnllm.android.modelist

import android.content.Context
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mnnllm.android.modelist.LocalModelsProvider
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE)
class ModelItemCacheDTOTest {

    @Before
    fun setup() {
        mockkObject(LocalModelsProvider)
    }

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun `toWrapper should apply cached tags to local model`() {
        // 1. Setup a raw local model (simulating what LocalModelsProvider finds on disk)
        val modelId = "local/path/to/model"
        val rawLocalModel = ModelItem().apply {
            this.modelId = modelId
            this.localPath = "/path/to/model"
            this.tags = mutableListOf() // Initially empty tags
        }
        
        // Mock LocalModelsProvider to return this raw model
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(rawLocalModel)

        // 2. Create a CacheDTO with enriched tags (simulating what we saved previously)
        val expectedTags = listOf("tag1", "favorite", "local")
        val marketItem = ModelMarketItem(
            modelName = "Cool Local Model",
            vendor = "Vendor",
            sizeB = 7.0,
            tags = expectedTags,
            categories = emptyList(),
            sources = emptyMap()
        )
        
        val cacheDTO = ModelListManager.ModelItemCacheDTO(
            modelId = modelId,
            modelPath = "/path/to/model",
            downloadSize = 1000L,
            isPinned = true,
            lastChatTime = 123456789L,
            downloadTime = 123456000L,
            isLocal = true,
            modelName = "Cool Local Model",
            tags = expectedTags,
            modelMarketItem = marketItem
        )

        // 3. Call toWrapper
        val context = mockk<Context>(relaxed = true)
        val wrapper = cacheDTO.toWrapper(context)

        // 4. Verification
        assertNotNull("Wrapper should not be null", wrapper)
        val resultingModel = wrapper!!.modelItem
        
        assertEquals("Model ID should match", modelId, resultingModel.modelId)
        
        // This assertion checks if the bug is present (currently fails) or fixed
        // The bug is that tags are NOT applied to the raw local model
        // So initially, this test should FAIL if we assert they ARE equal.
        // We want to prove the fix works, so we assert true equality.
        
        println("Resulting tags: ${resultingModel.tags}")
        assertTrue("Tags should contain cached tags. Actual: ${resultingModel.tags}", 
            resultingModel.tags.containsAll(expectedTags))
            
        // Check ModelMarketItem
        assertNotNull("ModelMarketItem should be restored", resultingModel.modelMarketItem)
        assertEquals("ModelName in MarketItem should match", "Cool Local Model", 
            (resultingModel.modelMarketItem as ModelMarketItem).modelName)

        // Check ModelName on the item itself (Critical for start-up display)
        assertEquals("ModelName should be restored from cache", "Cool Local Model", resultingModel.modelName)
    }
}
