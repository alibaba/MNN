package com.alibaba.mnnllm.android.modelmarket

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-JVM tests for the featured-model pinning of the market list.
 * The stable sort itself is exercised through the same comparator the
 * ViewModel applies so ordering regressions are caught here.
 */
class ModelMarketPinningTest {

    private fun item(name: String) = ModelMarketItem(
        modelName = name,
        vendor = "",
        sizeB = 0.0,
        tags = emptyList(),
        categories = emptyList(),
        sources = mapOf("ModelScope" to "MNN/$name")
    )

    @Test
    fun pocketFamilyIsPinnedByPrefix() {
        assertTrue(ModelMarketViewModel.isPinnedModel(item("MNN-Pocket-0.3B-MNN")))
        assertTrue(ModelMarketViewModel.isPinnedModel(item("MNN-Pocket-0.3B-DFlash")))
        assertTrue(ModelMarketViewModel.isPinnedModel(item("mnn-pocket-1b-MNN")))
    }

    @Test
    fun otherModelsAreNotPinned() {
        assertFalse(ModelMarketViewModel.isPinnedModel(item("Qwen2.5-0.5B-Instruct-MNN")))
        assertFalse(ModelMarketViewModel.isPinnedModel(item("PocketBook-not-ours-MNN")))
        assertFalse(ModelMarketViewModel.isPinnedModel(item("DeepSeek-R1-1.5B-Qwen-MNN")))
    }

    @Test
    fun pinnedModelsSortFirstAndOthersKeepCatalogOrder() {
        val catalog = listOf(
            "DeepSeek-R1-1.5B-Qwen-MNN",
            "MNN-Pocket-0.3B-MNN",
            "ERNIE-4.5-0.3B-PT-MNN",
            "MNN-Pocket-0.3B-DFlash",
            "Qwen2.5-0.5B-Instruct-MNN"
        ).map(::item)

        val sorted = catalog.sortedBy { if (ModelMarketViewModel.isPinnedModel(it)) 0 else 1 }

        assertEquals(
            listOf(
                "MNN-Pocket-0.3B-MNN",       // pinned, catalog order preserved
                "MNN-Pocket-0.3B-DFlash",    // pinned
                "DeepSeek-R1-1.5B-Qwen-MNN", // rest unchanged
                "ERNIE-4.5-0.3B-PT-MNN",
                "Qwen2.5-0.5B-Instruct-MNN"
            ),
            sorted.map { it.modelName }
        )
    }
}
