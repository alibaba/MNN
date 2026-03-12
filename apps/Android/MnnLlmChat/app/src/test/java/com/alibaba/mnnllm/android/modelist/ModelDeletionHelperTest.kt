package com.alibaba.mnnllm.android.modelist

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.ApplicationProvider as MlsApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.MmapUtils
import io.mockk.every
import io.mockk.mockkStatic
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * Unit tests for ModelDeletionHelper, especially tmps mmap entries discovery:
 * - tmps/<base>/modelers and tmps/<base>/modelscope must be discovered (per MmapUtils.getMmapDir)
 * - getStorageAnalysis and findOrphanMmapCaches must include these nested paths
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ModelDeletionHelperTest {

    private lateinit var application: Application
    private lateinit var filesDir: File
    private lateinit var tmpsDir: File

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        filesDir = application.filesDir
        tmpsDir = File(filesDir, "tmps")
        if (tmpsDir.exists()) tmpsDir.deleteRecursively()
        tmpsDir.mkdirs()

        mockkObject(ChatDataManager)
        mockkObject(ModelDownloadManager)
        every { ModelDownloadManager.getInstance(application) } returns mockk<ModelDownloadManager>(relaxed = true).apply {
            every { getDownloadedFile(any<String>()) } returns null
        }
        mockkStatic(MlsApplicationProvider::class)
        every { MlsApplicationProvider.get() } returns application
    }

    @After
    fun tearDown() {
        if (tmpsDir.exists()) tmpsDir.deleteRecursively()
        unmockkAll()
    }

    @Test
    fun getStorageAnalysis_discovers_tmps_base_modelers_as_separate_entry() {
        // Simulate path: tmps/taobao-mnn_Qwen3.5-0.8B-MNN/modelers (Modelers source)
        val baseDir = File(tmpsDir, "taobao-mnn_Qwen3.5-0.8B-MNN")
        baseDir.mkdirs()
        val modelersDir = File(baseDir, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "cache.bin").writeBytes(ByteArray(100))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns listOf(
                ChatDataManager.DownloadedModelInfo("Modelers/MNN/Qwen3.5-0.8B-MNN", 0L, "", 0L)
            )
        }

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)

        val modelersEntry = analysis.mmapCacheEntries.find {
            it.modelId == "taobao-mnn_Qwen3.5-0.8B-MNN/modelers"
        }
        assertTrue("must discover tmps/<base>/modelers as separate entry", modelersEntry != null)
        assertEquals(100L, modelersEntry!!.sizeBytes)
        assertTrue("path must end with /modelers", modelersEntry.path.endsWith("modelers"))
        assertFalse("Modelers model is tracked, so not orphan", modelersEntry.isOrphan)
    }

    @Test
    fun getStorageAnalysis_discovers_tmps_base_modelscope_as_separate_entry() {
        val baseDir = File(tmpsDir, "taobao-mnn_SomeModel")
        baseDir.mkdirs()
        val modelscopeDir = File(baseDir, "modelscope")
        modelscopeDir.mkdirs()
        File(modelscopeDir, "data.bin").writeBytes(ByteArray(200))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns listOf(
                ChatDataManager.DownloadedModelInfo("ModelScope/MNN/SomeModel", 0L, "", 0L)
            )
        }

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)

        val msEntry = analysis.mmapCacheEntries.find { it.modelId == "taobao-mnn_SomeModel/modelscope" }
        assertTrue("must discover tmps/<base>/modelscope as separate entry", msEntry != null)
        assertEquals(200L, msEntry!!.sizeBytes)
        assertFalse("ModelScope model is tracked", msEntry.isOrphan)
    }

    @Test
    fun getStorageAnalysis_tmps_base_without_source_subdir_treated_as_single_entry() {
        // HuggingFace style: tmps/taobao-mnn_XXX only (no modelers/modelscope)
        val baseDir = File(tmpsDir, "taobao-mnn_Other")
        baseDir.mkdirs()
        File(baseDir, "file.bin").writeBytes(ByteArray(50))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns emptyList()
        }

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)

        val entry = analysis.mmapCacheEntries.find { it.modelId == "taobao-mnn_Other" }
        assertTrue("must discover tmps/<base> when no source subdir", entry != null)
        assertEquals(50L, entry!!.sizeBytes)
        assertTrue("no tracked model, so orphan", entry.isOrphan)
    }

    @Test
    fun getStorageAnalysis_totalMmapCacheSize_equals_sum_of_discovered_entries() {
        val base1 = File(tmpsDir, "base1")
        base1.mkdirs()
        File(base1, "a.bin").writeBytes(ByteArray(10))
        val base2 = File(tmpsDir, "base2")
        base2.mkdirs()
        val modelersDir = File(base2, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "b.bin").writeBytes(ByteArray(20))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns emptyList()
        }

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)

        val sumEntries = analysis.mmapCacheEntries.sumOf { it.sizeBytes }
        assertEquals(
            "totalMmapCacheSize must equal sum of discovered mmap entries (no double-count)",
            sumEntries,
            analysis.totalMmapCacheSize
        )
        assertEquals(30L, analysis.totalMmapCacheSize)
    }

    @Test
    fun findOrphanMmapCaches_includes_untracked_tmps_base_modelers_dir() {
        val baseDir = File(tmpsDir, "taobao-mnn_Qwen3.5-0.8B-MNN")
        baseDir.mkdirs()
        val modelersDir = File(baseDir, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "x.bin").writeBytes(ByteArray(1))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns emptyList()
        }

        val orphans = ModelDeletionHelper.findOrphanMmapCaches(application)

        val orphanPaths = orphans.map { it.absolutePath }
        assertTrue(
            "orphans must include tmps/.../modelers when no Modelers model tracked",
            orphanPaths.any { it.endsWith("taobao-mnn_Qwen3.5-0.8B-MNN${File.separator}modelers") }
        )
    }

    @Test
    fun findOrphanMmapCaches_excludes_tmps_base_modelers_when_modelers_model_tracked() {
        val baseDir = File(tmpsDir, "taobao-mnn_Qwen3.5-0.8B-MNN")
        baseDir.mkdirs()
        val modelersDir = File(baseDir, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "x.bin").writeBytes(ByteArray(1))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns listOf(
                ChatDataManager.DownloadedModelInfo("Modelers/MNN/Qwen3.5-0.8B-MNN", 0L, "", 0L)
            )
        }

        val orphans = ModelDeletionHelper.findOrphanMmapCaches(application)

        val orphanPaths = orphans.map { it.absolutePath }
        assertFalse(
            "orphans must NOT include tmps/.../modelers when Modelers model is tracked",
            orphanPaths.any { it.endsWith("modelers") && it.contains("Qwen3.5-0.8B-MNN") }
        )
    }

    // --- getStorageDetailForEntry / getStorageAnalysisWithDetails (drill-down) ---

    @Test
    fun getStorageDetailForEntry_orphan_entry_has_null_resolved_model_and_lists_mmap_files() {
        val baseDir = File(tmpsDir, "orphan_base")
        baseDir.mkdirs()
        val modelersDir = File(baseDir, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "a.bin").writeBytes(ByteArray(10))
        File(modelersDir, "b.bin").writeBytes(ByteArray(20))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns emptyList()
        }

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)
        val entry = analysis.mmapCacheEntries.find { it.modelId == "orphan_base/modelers" }
            ?: analysis.mmapCacheEntries.find { it.path == modelersDir.absolutePath }
        assertNotNull("entry for orphan_base/modelers must exist", entry)

        val detail = ModelDeletionHelper.getStorageDetailForEntry(application, entry!!)

        assertNull("orphan entry must have null resolvedModelId", detail.resolvedModelId)
        assertEquals(entry.modelId, detail.entryModelId)
        assertEquals(entry.path, detail.mmapPath)
        assertEquals(30L, detail.mmapSize)
        assertTrue("mmapFiles must list files under mmap dir", detail.mmapFiles.size >= 2)
        assertTrue(detail.mmapFiles.any { it.first.endsWith("a.bin") && it.second == 10L })
        assertTrue(detail.mmapFiles.any { it.first.endsWith("b.bin") && it.second == 20L })
        assertTrue("orphan entry must be marked orphan", detail.isOrphan)
    }

    @Test
    fun getStorageDetailForEntry_tracked_entry_resolves_model_id() {
        // Path that MmapUtils.getMmapDir("Modelers/MNN/Base") produces: tmps/taobao-mnn_Base/modelers
        val baseName = "taobao-mnn_Base"
        val baseDir = File(tmpsDir, baseName)
        baseDir.mkdirs()
        val modelersDir = File(baseDir, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "x.bin").writeBytes(ByteArray(5))

        val modelId = "Modelers/MNN/Base"
        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns listOf(
                ChatDataManager.DownloadedModelInfo(modelId, 0L, "", 0L)
            )
        }

        val expectedMmapPath = MmapUtils.getMmapDir(modelId)
        assertEquals(
            "test setup: MmapUtils path must match our dir",
            modelersDir.absolutePath,
            expectedMmapPath
        )

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)
        val entry = analysis.mmapCacheEntries.find { it.path == expectedMmapPath }
            ?: analysis.mmapCacheEntries.find { it.modelId == "$baseName/modelers" }
        assertNotNull("entry must exist", entry)

        val detail = ModelDeletionHelper.getStorageDetailForEntry(application, entry!!)

        assertEquals("tracked entry must resolve to modelId", modelId, detail.resolvedModelId)
        assertEquals(entry.modelId, detail.entryModelId)
        assertEquals(expectedMmapPath, detail.mmapPath)
        assertFalse("tracked entry must not be orphan", detail.isOrphan)
    }

    @Test
    fun getStorageAnalysisWithDetails_returns_one_detail_per_entry() {
        val base1 = File(tmpsDir, "base1")
        base1.mkdirs()
        File(base1, "f.bin").writeBytes(ByteArray(7))
        val base2 = File(tmpsDir, "base2")
        base2.mkdirs()
        val modelersDir = File(base2, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "g.bin").writeBytes(ByteArray(11))

        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns emptyList()
        }

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)
        val details = ModelDeletionHelper.getStorageAnalysisWithDetails(application)

        assertEquals(
            "getStorageAnalysisWithDetails must return one detail per mmap entry",
            analysis.mmapCacheEntries.size,
            details.size
        )
        analysis.mmapCacheEntries.forEach { entry ->
            val detail = details.find { it.entryModelId == entry.modelId }
            assertNotNull("detail for entry ${entry.modelId} must exist", detail)
            assertEquals(entry.path, detail!!.mmapPath)
            assertEquals(entry.sizeBytes, detail.mmapSize)
            assertEquals(entry.isOrphan, detail.isOrphan)
        }
    }

    @Test
    fun getStorageDetailForEntry_includes_config_dir_and_files_when_config_exists() {
        val baseName = "taobao-mnn_ConfModel"
        val baseDir = File(tmpsDir, baseName)
        baseDir.mkdirs()
        val modelersDir = File(baseDir, "modelers")
        modelersDir.mkdirs()
        File(modelersDir, "cache.bin").writeBytes(ByteArray(3))

        val modelId = "Modelers/MNN/ConfModel"
        every { ChatDataManager.getInstance(application) } returns mockk<ChatDataManager>(relaxed = true).apply {
            every { getAllDownloadedModels() } returns listOf(
                ChatDataManager.DownloadedModelInfo(modelId, 0L, "", 0L)
            )
        }

        val configDir = File(File(filesDir, "configs"), ModelUtils.safeModelId(modelId))
        configDir.mkdirs()
        File(configDir, "custom_config.json").writeBytes(ByteArray(64))

        val analysis = ModelDeletionHelper.getStorageAnalysis(application)
        val entry = analysis.mmapCacheEntries.find { it.modelId == "$baseName/modelers" }
            ?: analysis.mmapCacheEntries.find { it.path == modelersDir.absolutePath }
        assertNotNull("entry must exist", entry)

        val detail = ModelDeletionHelper.getStorageDetailForEntry(application, entry!!)

        assertEquals(modelId, detail.resolvedModelId)
        assertNotNull("config dir should be set when config dir exists", detail.configDirPath)
        assertTrue("config dir path should contain configs", detail.configDirPath!!.contains("configs"))
        assertTrue("configFiles should list custom_config.json", detail.configFiles.any { it.first.endsWith("custom_config.json") })
        assertEquals(64L, detail.configFiles.find { it.first.endsWith("custom_config.json") }!!.second)
    }
}
