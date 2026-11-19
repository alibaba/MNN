package com.alibaba.mnnllm.android.modelmarket

import com.alibaba.mls.api.source.ModelSources
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-JVM tests for the ModelScope/HuggingFace catalog merge (alibaba/MNN#5132
 * Android counterpart). No Robolectric needed: only data classes and pure logic.
 */
class RemoteModelCatalogTest {

    private val llmFiles = setOf("config.json", "llm.mnn", "llm.mnn.weight", "tokenizer.txt")
    private val sanaFiles = setOf("llm", "connector.mnn", "transformer.mnn", "vae_encoder.mnn", "vae_decoder.mnn")
    private val sdFiles = setOf("text_encoder.mnn", "unet.mnn", "vae_decoder.mnn")

    private fun ms(name: String, fileSize: Long = 500_000_000L, params: Long = 1_500_000_000L, gated: Any? = false) =
        RemoteModelCatalog.MsRepo(id = "MNN/$name", fileSize = fileSize, params = params, gated = gated)

    private fun hf(name: String, gated: Any? = false) =
        RemoteModelCatalog.HfRepo(id = "taobao-mnn/$name", gated = gated)

    // ==================== format check ====================

    @Test
    fun supportsFormat_llmNeedsConfigAndLlmMnn() {
        assertTrue(RemoteModelCatalog.supportsFormat("Qwen2.5-0.5B-Instruct-MNN", llmFiles))
        assertFalse(RemoteModelCatalog.supportsFormat("MNN-Pocket-0.3B", setOf("config.json", "model.safetensors")))
        assertFalse(RemoteModelCatalog.supportsFormat("empty", emptySet()))
    }

    @Test
    fun supportsFormat_sanaNeedsEditPipelineParts() {
        assertTrue(RemoteModelCatalog.supportsFormat("MNN-Sana-Edit-V2", sanaFiles))
        assertFalse(RemoteModelCatalog.supportsFormat("MNN-Sana-Edit-V2-FAILTEST", setOf("config.json", "llm.mnn")))
    }

    @Test
    fun supportsFormat_stableDiffusionNeedsThreeModules() {
        assertTrue(RemoteModelCatalog.supportsFormat("stable-diffusion-v1-5-mnn-opencl", sdFiles))
        assertFalse(RemoteModelCatalog.supportsFormat("stable-diffusion-v1-5-mnn", setOf("config.json", "llm.mnn")))
    }

    // ==================== eligibility ====================

    @Test
    fun isEligible_rejectsGatedPrivateOversizeNpu() {
        assertTrue(RemoteModelCatalog.isEligible("good-MNN", 1_000, gated = false, isPrivate = false))
        assertFalse(RemoteModelCatalog.isEligible("gated-MNN", 1_000, gated = true, isPrivate = false))
        assertFalse(RemoteModelCatalog.isEligible("gated-auto-MNN", 1_000, gated = true, isPrivate = false))
        assertFalse(RemoteModelCatalog.isEligible("priv-MNN", 1_000, gated = false, isPrivate = true))
        assertFalse(RemoteModelCatalog.isEligible("huge-MNN", 17L * 1024 * 1024 * 1024, gated = false, isPrivate = false))
        assertFalse(RemoteModelCatalog.isEligible("Some-NPU", 1_000, gated = false, isPrivate = false))
    }

    // ==================== merge ====================

    @Test
    fun merge_unionOfBothHubsSortedAndDeduplicated() {
        val files = mapOf(
            "A-MNN" to llmFiles, "B-MNN" to llmFiles, "C-MNN" to llmFiles
        )
        val merged = RemoteModelCatalog.merge(
            msRepos = listOf(ms("B-MNN"), ms("A-MNN")),
            hfRepos = listOf(hf("C-MNN"), hf("A-MNN")),
            filesByName = files,
            labelsByName = emptyMap()
        )
        assertEquals(listOf("A-MNN", "B-MNN", "C-MNN"), merged.map { it.modelName })
    }

    @Test
    fun merge_setsBothSourceKeysWhenBothHubsListTheRepo() {
        val merged = RemoteModelCatalog.merge(
            msRepos = listOf(ms("Qwen2.5-0.5B-Instruct-MNN")),
            hfRepos = listOf(hf("Qwen2.5-0.5B-Instruct-MNN")),
            filesByName = mapOf("Qwen2.5-0.5B-Instruct-MNN" to llmFiles),
            labelsByName = emptyMap()
        )
        assertEquals(1, merged.size)
        val sources = merged[0].sources
        assertEquals("MNN/Qwen2.5-0.5B-Instruct-MNN", sources[ModelSources.sourceModelScope])
        assertEquals("taobao-mnn/Qwen2.5-0.5B-Instruct-MNN", sources[ModelSources.sourceHuffingFace])
    }

    @Test
    fun merge_dropsUnsupportedAndGatedRepos() {
        val merged = RemoteModelCatalog.merge(
            msRepos = listOf(
                ms("safetensors-only"),                 // no llm.mnn
                ms("gated-MNN", gated = true),
                ms("good-MNN")
            ),
            hfRepos = listOf(hf("safetensors-only"), hf("good-MNN")),
            filesByName = mapOf(
                "safetensors-only" to setOf("config.json", "model.safetensors"),
                "gated-MNN" to llmFiles,
                "good-MNN" to llmFiles
            ),
            labelsByName = emptyMap()
        )
        assertEquals(listOf("good-MNN"), merged.map { it.modelName })
    }

    @Test
    fun merge_oneHubDownStillListsWithTheOther() {
        val merged = RemoteModelCatalog.merge(
            msRepos = emptyList(),                       // ModelScope unreachable
            hfRepos = listOf(hf("B-MNN"), hf("A-MNN")),
            filesByName = mapOf("A-MNN" to llmFiles, "B-MNN" to llmFiles),
            labelsByName = emptyMap()
        )
        assertEquals(listOf("A-MNN", "B-MNN"), merged.map { it.modelName })
        assertEquals(
            mapOf<String, String>(ModelSources.sourceHuffingFace to "taobao-mnn/A-MNN"),
            merged[0].sources
        )
    }

    @Test
    fun merge_cdnLabelsWinForCuratedFieldsAndCarryModelers() {
        val label = ModelMarketItem(
            modelName = "gemma-4-E2B-it-MNN",
            vendor = "Gemma",
            sizeB = 2.0,
            tags = listOf("Vision", "Audio"),
            categories = listOf("chat", "gemma"),
            sources = mapOf("Modelers" to "some/modelers-path"),
            minAppVersion = "0.8.3",
            description = "curated desc",
            fileSize = 3_739_999_029L
        )
        val merged = RemoteModelCatalog.merge(
            msRepos = listOf(ms("gemma-4-E2B-it-MNN", fileSize = 3_700_000_000L, params = 4_000_000_000L)),
            hfRepos = listOf(hf("gemma-4-E2B-it-MNN")),
            filesByName = mapOf("gemma-4-E2B-it-MNN" to llmFiles),
            labelsByName = mapOf("gemma-4-E2B-it-MNN" to label)
        )
        val item = merged.single()
        assertEquals("Gemma", item.vendor)
        assertEquals(listOf("Vision", "Audio"), item.tags)
        assertEquals(listOf("chat", "gemma"), item.categories)
        assertEquals("0.8.3", item.minAppVersion)
        assertEquals("curated desc", item.description)
        assertEquals(2.0, item.sizeB, 0.001)
        // API refreshes the size; the Modelers source survives from the label.
        assertEquals(3_700_000_000L, item.fileSize)
        assertEquals("some/modelers-path", item.sources["Modelers"])
        assertEquals("MNN/gemma-4-E2B-it-MNN", item.sources[ModelSources.sourceModelScope])
    }

    @Test
    fun toItem_apiFillsSizeFromParamsWhenNoLabel() {
        val item = RemoteModelCatalog.toItem(
            "LFM3-1.7B-MNN",
            ms = RemoteModelCatalog.MsRepo(id = "MNN/LFM3-1.7B-MNN", fileSize = 1_200_000_000L, params = 1_700_000_000L),
            hf = null,
            label = null
        )
        assertEquals(1.7, item.sizeB, 0.01)
        assertEquals(1_200_000_000L, item.fileSize)
        assertEquals("LFM3-1.7B-MNN", item.modelName)
        // No CDN label -> vendor is inferred from the family prefix.
        assertEquals("LFM", item.vendor)
        assertNull(item.minAppVersion)
    }

    // ==================== vendor inference ====================

    @Test
    fun inferVendor_matchesFamilyPrefixes() {
        assertEquals("Qwen", RemoteModelCatalog.inferVendor("Qwen3-4B-MNN"))
        assertEquals("Qwen", RemoteModelCatalog.inferVendor("QwQ-32B-MNN"))
        assertEquals("DeepSeek", RemoteModelCatalog.inferVendor("DeepSeek-R1-7B-Qwen-MNN"))
        assertEquals("LFM", RemoteModelCatalog.inferVendor("LFM2-1.2B-MNN"))
        assertEquals("stability.ai", RemoteModelCatalog.inferVendor("stable-diffusion-v1-5-mnn-opencl"))
        assertEquals("stability.ai", RemoteModelCatalog.inferVendor("MNN-Sana-Edit-V2"))
        assertEquals("InternLM", RemoteModelCatalog.inferVendor("InternVL2_5-1B-MNN"))
        assertEquals("Other", RemoteModelCatalog.inferVendor("Unknown-Family-MNN"))
    }

    @Test
    fun toItem_blankLabelVendorFallsBackToInference() {
        val label = ModelMarketItem(
            modelName = "X-MNN", vendor = "", sizeB = 1.0, tags = emptyList(),
            categories = emptyList(), sources = emptyMap()
        )
        val item = RemoteModelCatalog.toItem(
            "X-MNN",
            ms = RemoteModelCatalog.MsRepo(id = "MNN/X-MNN", fileSize = 1_000L),
            hf = null,
            label = label
        )
        assertEquals("Other", item.vendor)
    }

    @Test
    fun toItem_extractsCustomTagsFromModelScope() {
        val repo = RemoteModelCatalog.MsRepo(
            id = "MNN/MNN-Pocket-0.3B",
            tasks = listOf("text-generation"),
            tags = listOf("license:apache-2.0", "custom_tag:mnn-pocket", "custom_tag:int4", "library:safetensors")
        )
        val item = RemoteModelCatalog.toItem("MNN-Pocket-0.3B", repo, null, null)
        assertEquals(setOf("text-generation", "mnn-pocket", "int4"), item.tags.toSet())
        assertEquals(listOf("text-generation"), item.categories)
    }

    @Test
    fun merge_repoNameIsLastPathSegmentOnly() {
        val merged = RemoteModelCatalog.merge(
            msRepos = listOf(RemoteModelCatalog.MsRepo(id = "MNN/deep/nested-MNN")),
            hfRepos = emptyList(),
            filesByName = mapOf("nested-MNN" to llmFiles),
            labelsByName = emptyMap()
        )
        assertEquals(listOf("nested-MNN"), merged.map { it.modelName })
        assertEquals("MNN/deep/nested-MNN", merged[0].sources[ModelSources.sourceModelScope])
    }

    // ==================== DTO parsing shape ====================

    @Test
    fun msRepo_gatedAndPrivateParseFromJsonBooleans() {
        val repo = com.google.gson.Gson().fromJson(
            """{"id":"MNN/X-MNN","last_modified":"2026-11-10T06:47:01Z","file_size":739624210,
               |"params":361551232,"tasks":["text-generation"],"gated":false,"private":false}""".trimMargin(),
            RemoteModelCatalog.MsRepo::class.java
        )
        assertEquals("X-MNN", repo.name)
        assertEquals(739624210L, repo.fileSize)
        assertEquals(361551232L, repo.params)
        assertFalse(repo.gated as Boolean)
    }

    @Test
    fun hfRepo_fileNamesComeFromSiblings() {
        val repo = com.google.gson.Gson().fromJson(
            """{"id":"taobao-mnn/Y-MNN","gated":"auto","private":false,
               |"siblings":[{"rfilename":"config.json"},{"rfilename":"llm.mnn"}]}""".trimMargin(),
            RemoteModelCatalog.HfRepo::class.java
        )
        assertEquals("Y-MNN", repo.name)
        assertEquals(setOf("config.json", "llm.mnn"), repo.fileNames())
        assertEquals("auto", repo.gated)
    }
}
