package com.alibaba.mnnllm.android.debug

import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.modelist.ModelListManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class ModelListDumperPluginTest {
    private class FakeController : ModelListDebugController {
        var modelMap: Map<String, ModelItem> = emptyMap()
        var modelTags: Map<String, List<String>> = emptyMap()
        var modelExtraTags: Map<String, List<String>> = emptyMap()
        var storageEntries: List<StorageEntry> = emptyList()
        var deleteResult: Boolean = false
        var lastDeletedPath: String? = null

        override fun getModelListState(): ModelListManager.ModelListState = ModelListManager.ModelListState.Loading

        override suspend fun notifyModelListMayChange(reason: ModelListManager.ChangeReason) {
            // no-op for unit test
        }

        override fun getModelIdModelMap(): Map<String, ModelItem> = modelMap

        override fun getModelTags(modelId: String): List<String> = modelTags[modelId] ?: emptyList()

        override fun getExtraTags(modelId: String): List<String> = modelExtraTags[modelId] ?: emptyList()

        override fun isThinkingModel(modelId: String): Boolean = false

        override fun isVisualModel(modelId: String): Boolean = false

        override fun isAudioModel(modelId: String): Boolean = false

        override fun isVideoModel(modelId: String): Boolean = false

        override fun scanStorageEntries(): List<StorageEntry> = storageEntries

        override fun deleteSymlink(absolutePath: String): Boolean {
            lastDeletedPath = absolutePath
            return deleteResult
        }

        override fun getTagLocaleInfo(): TagLocaleInfo = TagLocaleInfo(
            deviceLocale = "en_US",
            isChinese = false,
            sampleTags = listOf(
                TagDisplaySample("Chat", "对话", "Chat", "Chat"),
                TagDisplaySample("Multimodal", "多模态", "Multimodal", "Multimodal")
            )
        )
    }

    @Test
    fun `tags without modelId should dump all models with tags and extra tags`() {
        val modelA = ModelItem().apply {
            modelId = "model/a"
            modelName = "Model A"
        }
        val modelB = ModelItem().apply {
            modelId = "model/b"
            modelName = "Model B"
        }

        val controller = FakeController().apply {
            modelMap = linkedMapOf(
                "model/a" to modelA,
                "model/b" to modelB
            )
            modelTags = mapOf(
                "model/a" to listOf("Think", "Vision"),
                "model/b" to emptyList()
            )
            modelExtraTags = mapOf(
                "model/a" to listOf("QNN"),
                "model/b" to listOf("Audio")
            )
        }

        val plugin = ModelListDumperPlugin(controller)
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("tags"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("All model tags (2)"))
        assertTrue(output.contains("- model/a"))
        assertTrue(output.contains("tags: [Think, Vision]"))
        assertTrue(output.contains("extraTags: [QNN]"))
        assertTrue(output.contains("- model/b"))
        assertTrue(output.contains("tags: []"))
        assertTrue(output.contains("extraTags: [Audio]"))
    }

    @Test
    fun `files should list storage entries grouped by container`() {
        val controller = FakeController().apply {
            storageEntries = listOf(
                StorageEntry(
                    "Qwen2.5-0.5B-MNN", "/data/.mnnmodels/modelscope/Qwen2.5-0.5B-MNN",
                    true, "/data/downloads/abc", true, true,
                    "ModelScope/MNN/Qwen2.5-0.5B-MNN", "modelscope"
                ),
                StorageEntry(
                    "sd-v1-5", "/data/.mnnmodels/modelscope/sd-v1-5",
                    true, "/data/downloads/def", false, true,
                    "ModelScope/MNN/sd-v1-5", "modelscope"
                ),
                StorageEntry(
                    "builtin-model", "/data/.mnnmodels/builtin/builtin-model",
                    false, null, true, true,
                    "Builtin/MNN/builtin-model", "builtin"
                )
            )
        }

        val plugin = ModelListDumperPlugin(controller)
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("files"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Storage Entries (3)"))
        assertTrue(output.contains("[modelscope]"))
        assertTrue(output.contains("Qwen2.5-0.5B-MNN [SYMLINK]"))
        assertTrue(output.contains("Target Exists: true"))
        assertTrue(output.contains("Target Exists: false"))
        assertTrue(output.contains("[builtin]"))
        assertTrue(output.contains("builtin-model [DIR]"))
    }

    @Test
    fun `files should show empty message when no entries`() {
        val plugin = ModelListDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("files"), PrintStream(out))

        assertTrue(out.toString().contains("No entries found"))
    }

    @Test
    fun `unlink should remove symlink entry by modelId`() {
        val controller = FakeController().apply {
            storageEntries = listOf(
                StorageEntry(
                    "sd-v1-5", "/data/.mnnmodels/modelscope/sd-v1-5",
                    true, "/data/downloads/def", true, true,
                    "ModelScope/MNN/sd-v1-5", "modelscope"
                )
            )
            deleteResult = true
        }

        val plugin = ModelListDumperPlugin(controller)
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("unlink", "ModelScope/MNN/sd-v1-5"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Unlinked: sd-v1-5"))
        assertTrue(output.contains("Target preserved"))
        assertEquals("/data/.mnnmodels/modelscope/sd-v1-5", controller.lastDeletedPath)
    }

    @Test
    fun `unlink should match by directory name`() {
        val controller = FakeController().apply {
            storageEntries = listOf(
                StorageEntry(
                    "sd-v1-5", "/data/.mnnmodels/modelscope/sd-v1-5",
                    true, "/data/downloads/def", true, true,
                    "ModelScope/MNN/sd-v1-5", "modelscope"
                )
            )
            deleteResult = true
        }

        val plugin = ModelListDumperPlugin(controller)
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("unlink", "sd-v1-5"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Unlinked: sd-v1-5"))
    }

    @Test
    fun `unlink should refuse non-symlink entries`() {
        val controller = FakeController().apply {
            storageEntries = listOf(
                StorageEntry(
                    "builtin-model", "/data/.mnnmodels/builtin/builtin-model",
                    false, null, true, true,
                    "Builtin/MNN/builtin-model", "builtin"
                )
            )
        }

        val plugin = ModelListDumperPlugin(controller)
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("unlink", "Builtin/MNN/builtin-model"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("not a symlink"))
    }

    @Test
    fun `unlink without args should print usage`() {
        val plugin = ModelListDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("unlink"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Usage: dumpapp models unlink"))
    }

    @Test
    fun `unlink should show error for unknown model`() {
        val plugin = ModelListDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()
        plugin.execute(listOf("unlink", "nonexistent-model"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("No entry found for: nonexistent-model"))
    }
}
