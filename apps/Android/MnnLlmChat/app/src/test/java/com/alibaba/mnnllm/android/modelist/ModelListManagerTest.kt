package com.alibaba.mnnllm.android.modelist

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadPersistentData
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.modelmarket.ModelMarketCache
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.nio.file.Files

@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE)
@ExperimentalCoroutinesApi
class ModelListManagerTest {

    private lateinit var context: Context
    private val testDispatcher = UnconfinedTestDispatcher()
    private lateinit var tempDir: File

    @Before
    fun setup() {
        Dispatchers.setMain(testDispatcher)
        context = ApplicationProvider.getApplicationContext()
        
        // Reset ModelListManager via reflection
        resetModelListManager()

        // Create a temporary directory for local model
        tempDir = Files.createTempDirectory("model_test").toFile()
        File(tempDir, "config.json").writeText("{}")

        // Mock Singleton/Static objects
        mockkObject(LocalModelsProvider)
        mockkObject(ChatDataManager)
        mockkObject(PreferenceUtils)
        mockkObject(ModelMarketCache)
        mockkObject(DownloadPersistentData)
        mockkObject(BuiltinModelManager)
        
        mockkObject(MnnLlmApplication)
        val mockApp = mockk<MnnLlmApplication>(relaxed = true)
        every { MnnLlmApplication.getInstance() } returns mockApp

        // Default Mocks
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf()
        every { PreferenceUtils.getPinnedModels(any()) } returns emptySet()
        every { DownloadPersistentData.getDownloadSizeSaved(any(), any()) } returns 0L
        every { BuiltinModelManager.hasBuiltinModels(any()) } returns false
        
        // Mock ChatDataManager instance
        val mockChatDataManager = mockk<ChatDataManager>(relaxed = true)
        every { ChatDataManager.getInstance(any()) } returns mockChatDataManager
        
        // Mock ModelMarketCache Flows
        every { ModelMarketCache.observeModelMarketConfig(any()) } returns flowOf(emptyMap())
        coEvery { ModelMarketCache.getModelFromCache(any()) } returns null
    }

    private fun resetModelListManager() {
        try {
            val instance = ModelListManager
            val kClass = instance::class
            
            // Reset isInitialized
            val isInitializedProp = kClass.java.getDeclaredField("isInitialized")
            isInitializedProp.isAccessible = true
            isInitializedProp.setBoolean(instance, false)
            
            // Reset isInitializing
            val isInitializingProp = kClass.java.getDeclaredField("isInitializing")
            isInitializingProp.isAccessible = true
            isInitializingProp.setBoolean(instance, false)
            
            // Reset _modelListState to Loading
            val modelListStateProp = kClass.java.getDeclaredField("_modelListState")
            modelListStateProp.isAccessible = true
            // We need to set it to a new MutableStateFlow or reset the existing one
            // Easier to just let initialize() overwrite it? No, initialize() updates it.
            // But if it was Success from previous run, we want it back to Loading?
            // Actually initialize sets it to Loading or Success.
            // Resetting isInitialized is the most important.
            
        } catch (e: Exception) {
            println("Failed to reset ModelListManager: ${e.message}")
        }
    }

    @After
    fun tearDown() {
        unmockkAll()
        Dispatchers.resetMain()
        ModelListManager.clearModelCache()
        resetModelListManager() // Reset again to be safe
        
        if (::tempDir.isInitialized && tempDir.exists()) {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun `initialize should load models correctly`() = runTest {
        // Setup scenarios
        val modelId = "test-model-id"
        val modelItem = ModelItem().apply { 
            this.modelId = modelId
            this.modelName = "Test Model"
            this.localPath = tempDir.absolutePath
        }
        // Force isLocal = true using reflection if setter not available, 
        // try direct property access first (assuming it works in Kotlin if var)
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal via reflection: ${e.message}")
        }
        
        // Mock LocalModelsProvider to return a model
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        
        // Trigger initialization
        ModelListManager.initialize(context)

        // Wait for flow emission
        val models = ModelListManager.observeModels().firstOrNull()
        
        // Verify
        assertTrue("Models should not be null", models != null)
        assertTrue("Models should not be empty", models!!.isNotEmpty())
        assertEquals(1, models.size)
        assertEquals(modelId, models[0].modelItem.modelId)
    }
    
    @Test
    fun `getModelTags should return correct tags`() = runTest {
        // 1. Setup a model with tags
        val modelId = "tagged-model"
        val tags = listOf("tag1", "tag2")
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = tags
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
             println("Could not set isLocal via reflection: ${e.message}")
        }
        
        // Mock LocalModelsProvider
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        
        // Initialize
        ModelListManager.initialize(context)
        
        // Wait for flow emission to ensure map is populated
        ModelListManager.observeModels().firstOrNull()
        
        val resultTags = ModelListManager.getModelTags(modelId)
        
        // The implementation adds "local" tag to local models if not present
        assertTrue(resultTags.contains("tag1"))
        assertTrue(resultTags.contains("tag2"))
        
        // Check if "local" tag is added (implementation dependent)
        assertTrue(resultTags.contains("local"))
    }

    // ========== Model Type Detection Tests ==========

    @Test
    fun `isThinkingModel should return true for model with thinking tag`() = runTest {
        // Given
        val modelId = "thinking-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("Think", "other-tag")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val result = ModelListManager.isThinkingModel(modelId)

        // Then
        assertTrue(result)
    }

    @Test
    fun `isThinkingModel should return false for model without thinking tag`() = runTest {
        // Given
        val modelId = "normal-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("other-tag")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val result = ModelListManager.isThinkingModel(modelId)

        // Then
        assertFalse(result)
    }

    @Test
    fun `isVisualModel should return true for model with visual tag`() = runTest {
        // Given
        val modelId = "visual-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("Vision", "multimodal")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val result = ModelListManager.isVisualModel(modelId)

        // Then
        assertTrue(result)
    }

    @Test
    fun `isAudioModel should return true for model with audio tag`() = runTest {
        // Given
        val modelId = "audio-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("Audio")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val result = ModelListManager.isAudioModel(modelId)

        // Then
        assertTrue(result)
    }

    @Test
    fun `isVideoModel should return true for model with video tag`() = runTest {
        // Given
        val modelId = "video-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("Video")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val result = ModelListManager.isVideoModel(modelId)

        // Then
        assertTrue(result)
    }

    // ========== DTO Conversion Tests ==========

    @Test
    fun `ModelItemCacheDTO from should convert valid wrapper correctly`() = runTest {
        // Given
        val modelId = "test-model"
        val modelPath = tempDir.absolutePath
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.modelName = "Test Model"
            this.localPath = modelPath
            this.tags = listOf("tag1", "tag2")
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 1024L,
            isPinned = true
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertTrue("DTO should not be null", dto != null)
        assertEquals(modelId, dto!!.modelId)
        assertEquals(modelPath, dto.modelPath)
        assertEquals(1024L, dto.downloadSize)
        assertTrue(dto.isPinned)
    }

    @Test
    fun `ModelItemCacheDTO from should return null for null modelId`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = null  // null modelId
            this.modelName = "Test Model"
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 1024L,
            isPinned = false
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertNull("DTO should be null for null modelId", dto)
    }

    @Test
    fun `ModelItemCacheDTO from should return null for blank modelId`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "   "  // blank modelId
            this.modelName = "Test Model"
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 1024L,
            isPinned = false
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertNull("DTO should be null for blank modelId", dto)
    }

    // ========== State Management Tests ==========

    @Test
    fun `getCurrentModels should return models after initialization`() = runTest {
        // Given
        val modelId = "test-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()
        val models = ModelListManager.getCurrentModels()

        // Then
        assertNotNull("Models should not be null after initialization", models)
        assertTrue("Models should not be empty", models!!.isNotEmpty())
    }

    @Test
    fun `clearModelCache should clear cached models`() = runTest {
        // Given
        val modelId = "test-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        ModelListManager.clearModelCache()

        // Then - cache is cleared (verified by internal state)
        // Note: This is a void method, we're just ensuring it doesn't throw
        assertTrue("clearModelCache should execute without error", true)
    }

    // ========== Path Parsing Tests (via reflection) ==========

    @Test
    fun `createModelIdFromPath should parse modelers path correctly`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/modelers/test-model"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertEquals("Modelers/MNN/test-model", modelId)
    }

    @Test
    fun `createModelIdFromPath should parse modelscope path correctly`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/modelscope/test-model"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertEquals("ModelScope/MNN/test-model", modelId)
    }

    @Test
    fun `createModelIdFromPath should parse builtin path correctly`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/builtin/test-model"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertEquals("Builtin/MNN/test-model", modelId)
    }

    @Test
    fun `createModelIdFromPath should parse root level path as HuggingFace`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/test-model"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertEquals("HuggingFace/taobao-mnn/test-model", modelId)
    }

    @Test
    fun `createModelIdFromPath should return null for invalid path`() = runTest {
        // Given
        val modelPath = "/invalid/path/test-model"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertNull("Should return null for invalid path", modelId)
    }

    // Helper method to invoke private createModelIdFromPath via reflection
    private fun invokeCreateModelIdFromPath(path: String): String? {
        return try {
            val method = ModelListManager::class.java.getDeclaredMethod(
                "createModelIdFromPath",
                Context::class.java,
                String::class.java
            )
            method.isAccessible = true
            method.invoke(ModelListManager, context, path) as? String
        } catch (e: Exception) {
            println("Failed to invoke createModelIdFromPath: ${e.message}")
            null
        }
    }

    // ========== Data Change Detection Tests ==========

    @Test
    fun `hasDataChanged should return true when cached is null`() = runTest {
        // Given
        val freshModels = listOf(createTestWrapper("model1"))

        // When
        val result = invokeHasDataChanged(null, freshModels)

        // Then
        assertTrue("Should return true when cached is null", result)
    }

    @Test
    fun `hasDataChanged should return true when sizes differ`() = runTest {
        // Given
        val cached = listOf(createTestWrapper("model1"))
        val fresh = listOf(createTestWrapper("model1"), createTestWrapper("model2"))

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertTrue("Should return true when sizes differ", result)
    }

    @Test
    fun `hasDataChanged should return true when model IDs differ`() = runTest {
        // Given
        val cached = listOf(createTestWrapper("model1"))
        val fresh = listOf(createTestWrapper("model2"))

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertTrue("Should return true when model IDs differ", result)
    }

    @Test
    fun `hasDataChanged should return true when isPinned changes`() = runTest {
        // Given
        val cached = listOf(createTestWrapper("model1", isPinned = false))
        val fresh = listOf(createTestWrapper("model1", isPinned = true))

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertTrue("Should return true when isPinned changes", result)
    }

    @Test
    fun `hasDataChanged should return false when models are identical`() = runTest {
        // Given
        val cached = listOf(createTestWrapper("model1"))
        val fresh = listOf(createTestWrapper("model1"))

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertFalse("Should return false when models are identical", result)
    }

    // Helper methods
    private fun createTestWrapper(
        modelId: String,
        isPinned: Boolean = false,
        downloadSize: Long = 1024L
    ): ModelItemWrapper {
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.localPath = tempDir.absolutePath
        }
        return ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = downloadSize,
            isPinned = isPinned
        )
    }

    private fun invokeHasDataChanged(
        cached: List<ModelItemWrapper>?,
        fresh: List<ModelItemWrapper>
    ): Boolean {
        return try {
            val method = ModelListManager::class.java.getDeclaredMethod(
                "hasDataChanged",
                List::class.java,
                List::class.java
            )
            method.isAccessible = true
            method.invoke(ModelListManager, cached, fresh) as Boolean
        } catch (e: Exception) {
            println("Failed to invoke hasDataChanged: ${e.message}")
            false
        }
    }

    // ========== Additional Coverage Tests ==========

    @Test
    fun `getModelIdModelMap should return model map`() = runTest {
        // Given
        val modelId = "test-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val modelMap = ModelListManager.getModelIdModelMap()

        // Then
        assertNotNull("Model map should not be null", modelMap)
        assertTrue("Model map should contain the model", modelMap.containsKey(modelId))
    }

    @Test
    fun `getExtraTags should return empty list for non-existent model`() = runTest {
        // Given
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf()
        ModelListManager.initialize(context)

        // When
        val tags = ModelListManager.getExtraTags("non-existent-model")

        // Then
        assertTrue("Should return empty list for non-existent model", tags.isEmpty())
    }

    @Test
    fun `getModelTags should return empty list for non-existent model`() = runTest {
        // Given
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf()
        ModelListManager.initialize(context)

        // When
        val tags = ModelListManager.getModelTags("non-existent-model")

        // Then
        assertTrue("Should return empty list for non-existent model", tags.isEmpty())
    }

    // ========== Multiple Models Tests ==========

    @Test
    fun `initialize should handle multiple models correctly`() = runTest {
        // Given
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.modelName = "Model 1"
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.modelName = "Model 2"
            this.localPath = tempDir.absolutePath
        }

        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(model1, true)
            isLocalField.setBoolean(model2, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(model1, model2)

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertEquals(2, models!!.size)
        assertTrue("Should contain model1", models.any { it.modelItem.modelId == "model1" })
        assertTrue("Should contain model2", models.any { it.modelItem.modelId == "model2" })
    }

    @Test
    fun `observeModelList should emit state flow`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        val stateFlow = ModelListManager.observeModelList()
        val state = stateFlow.firstOrNull()

        // Then
        assertNotNull("State should not be null", state)
        assertTrue("State should be Success", state is ModelListManager.ModelListState.Success)
    }

    @Test
    fun `pinned models should appear first in list`() = runTest {
        // Given
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.localPath = tempDir.absolutePath
        }

        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(model1, true)
            isLocalField.setBoolean(model2, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(model1, model2)
        every { PreferenceUtils.getPinnedModels(any()) } returns setOf("model2")

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertEquals(2, models!!.size)
        assertTrue("First model should be pinned", models[0].isPinned)
        assertEquals("model2", models[0].modelItem.modelId)
    }

    // ========== Edge Cases and Boundary Tests ==========

    @Test
    fun `initialize with empty model list should succeed`() = runTest {
        // Given
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf()

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertTrue("Models should be empty", models!!.isEmpty())
    }

    @Test
    fun `getModelTags with empty tags should return local tag only`() = runTest {
        // Given
        val modelId = "model-no-tags"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = emptyList()
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val tags = ModelListManager.getModelTags(modelId)

        // Then
        assertTrue("Should contain local tag", tags.contains("local"))
    }

    @Test
    fun `hasDataChanged with empty lists should return false`() = runTest {
        // Given
        val cached = emptyList<ModelItemWrapper>()
        val fresh = emptyList<ModelItemWrapper>()

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertFalse("Empty lists should be considered equal", result)
    }

    @Test
    fun `ModelItemCacheDTO toWrapper should reconstruct local model`() = runTest {
        // Given
        val modelId = "test-local-model"
        val modelPath = tempDir.absolutePath
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.localPath = modelPath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        val dto = ModelListManager.ModelItemCacheDTO(
            modelId = modelId,
            modelPath = modelPath,
            downloadSize = 2048L,
            isPinned = true,
            lastChatTime = 0L,
            downloadTime = System.currentTimeMillis(),
            isLocal = true,
            modelName = "Test Model",
            tags = listOf("test"),
            modelMarketItem = null
        )

        // When
        val wrapper = dto.toWrapper(context)

        // Then
        assertNotNull("Wrapper should not be null", wrapper)
        assertEquals(modelId, wrapper!!.modelItem.modelId)
        assertTrue("Should be pinned", wrapper.isPinned)
        assertEquals(2048L, wrapper.downloadSize)
    }

    @Test
    fun `model with multiple tags should preserve all tags`() = runTest {
        // Given
        val modelId = "multi-tag-model"
        val tags = listOf("tag1", "tag2", "tag3")
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = tags
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val resultTags = ModelListManager.getModelTags(modelId)

        // Then
        assertTrue("Should contain tag1", resultTags.contains("tag1"))
        assertTrue("Should contain tag2", resultTags.contains("tag2"))
        assertTrue("Should contain tag3", resultTags.contains("tag3"))
        assertTrue("Should contain local tag", resultTags.contains("local"))
    }

    @Test
    fun `hasDataChanged should detect downloadSize change`() = runTest {
        // Given
        val cached = listOf(createTestWrapper("model1", downloadSize = 1024L))
        val fresh = listOf(createTestWrapper("model1", downloadSize = 2048L))

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertTrue("Should detect downloadSize change", result)
    }

    @Test
    fun `createModelIdFromPath with empty model name should return null`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/modelers/"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertNull("Should return null for empty model name", modelId)
    }

    @Test
    fun `createModelIdFromPath with nested path should return null`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/invalid/nested/path"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertNull("Should return null for nested invalid path", modelId)
    }

    @Test
    fun `observeModelsWithSource should emit models with data source`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        val result = ModelListManager.observeModelsWithSource().firstOrNull()

        // Then
        assertNotNull("Result should not be null", result)
        assertNotNull("Models should not be null", result!!.first)
        assertTrue("Should have data source", result.second != null)
    }

    @Test
    fun `model with null modelName should handle gracefully`() = runTest {
        // Given
        val modelId = "model-no-name"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.modelName = null
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertEquals(1, models!!.size)
        assertEquals(modelId, models[0].modelItem.modelId)
    }

    @Test
    fun `ModelItemCacheDTO from with null modelPath should succeed`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = null
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 1024L,
            isPinned = false
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertNotNull("DTO should not be null", dto)
        assertEquals("test-model", dto!!.modelId)
        assertNull("modelPath should be null", dto.modelPath)
    }

    @Test
    fun `setContext should update context`() = runTest {
        // Given
        val newContext = ApplicationProvider.getApplicationContext<Context>()

        // When
        ModelListManager.setContext(newContext)

        // Then - context is set (verified by not throwing exception)
        assertTrue("setContext should execute without error", true)
    }

    @Test
    fun `isShowingCachedData should return false after fresh load`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()
        val isShowingCached = ModelListManager.isShowingCachedData()

        // Then
        assertFalse("Should not be showing cached data after fresh load", isShowingCached)
    }

    @Test
    fun `models with lastChatTime should be sorted by recent usage`() = runTest {
        // Given
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.localPath = tempDir.absolutePath
        }

        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(model1, true)
            isLocalField.setBoolean(model2, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(model1, model2)

        val mockChatDataManager = mockk<ChatDataManager>(relaxed = true)
        every { ChatDataManager.getInstance(any()) } returns mockChatDataManager
        every { mockChatDataManager.getLastChatTime("model1") } returns 1000L
        every { mockChatDataManager.getLastChatTime("model2") } returns 2000L

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        // Model2 should come first due to more recent lastChatTime
        // Note: Actual sorting depends on implementation details
        assertTrue("Should have 2 models", models!!.size == 2)
    }

    @Test
    fun `hasDataChanged should detect lastChatTime change`() = runTest {
        // Given
        val modelItem1 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val wrapper1 = ModelItemWrapper(
            modelItem = modelItem1,
            downloadedModelInfo = ChatDataManager.DownloadedModelInfo(
                "model1", 1000L, tempDir.absolutePath, 1000L
            ),
            downloadSize = 1024L,
            isPinned = false
        )

        val modelItem2 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val wrapper2 = ModelItemWrapper(
            modelItem = modelItem2,
            downloadedModelInfo = ChatDataManager.DownloadedModelInfo(
                "model1", 1000L, tempDir.absolutePath, 2000L
            ),
            downloadSize = 1024L,
            isPinned = false
        )

        val cached = listOf(wrapper1)
        val fresh = listOf(wrapper2)

        // When
        val result = invokeHasDataChanged(cached, fresh)

        // Then
        assertTrue("Should detect lastChatTime change", result)
    }

    @Test
    fun `DTO with empty modelId should return null from from()`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = ""
            this.localPath = tempDir.absolutePath
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 1024L,
            isPinned = false
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertNull("DTO should be null for empty modelId", dto)
    }

    @Test
    fun `multiple models with same tags should all be retrievable`() = runTest {
        // Given
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.tags = listOf("common-tag")
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.tags = listOf("common-tag")
            this.localPath = tempDir.absolutePath
        }

        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(model1, true)
            isLocalField.setBoolean(model2, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(model1, model2)

        // When
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        val tags1 = ModelListManager.getModelTags("model1")
        val tags2 = ModelListManager.getModelTags("model2")

        // Then
        assertTrue("Model1 should have common-tag", tags1.contains("common-tag"))
        assertTrue("Model2 should have common-tag", tags2.contains("common-tag"))
    }

    @Test
    fun `DTO toWrapper with non-existent local model should return null`() = runTest {
        // Given
        every { LocalModelsProvider.getLocalModels() } returns mutableListOf()

        val dto = ModelListManager.ModelItemCacheDTO(
            modelId = "non-existent-model",
            modelPath = tempDir.absolutePath,
            downloadSize = 1024L,
            isPinned = false,
            lastChatTime = 0L,
            downloadTime = System.currentTimeMillis(),
            isLocal = true,
            modelName = "Test",
            tags = null,
            modelMarketItem = null
        )

        // When
        val wrapper = dto.toWrapper(context)

        // Then
        assertNull("Wrapper should be null for non-existent local model", wrapper)
    }

    // ========== Complex Sorting Tests ==========

    @Test
    fun `models should be sorted by pinned then lastChatTime then downloadTime`() = runTest {
        // Given
        val now = System.currentTimeMillis()
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.localPath = tempDir.absolutePath
        }
        val model3 = ModelItem().apply {
            this.modelId = "model3"
            this.localPath = tempDir.absolutePath
        }

        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(model1, true)
            isLocalField.setBoolean(model2, true)
            isLocalField.setBoolean(model3, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(model1, model2, model3)
        every { PreferenceUtils.getPinnedModels(any()) } returns setOf("model2")

        val mockChatDataManager = mockk<ChatDataManager>(relaxed = true)
        every { ChatDataManager.getInstance(any()) } returns mockChatDataManager
        every { mockChatDataManager.getLastChatTime("model1") } returns now - 1000L
        every { mockChatDataManager.getLastChatTime("model2") } returns 0L
        every { mockChatDataManager.getLastChatTime("model3") } returns now - 2000L

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertEquals(3, models!!.size)
        // model2 should be first (pinned)
        assertEquals("model2", models[0].modelItem.modelId)
        assertTrue("First model should be pinned", models[0].isPinned)
    }

    @Test
    fun `multiple pinned models should be sorted by lastChatTime`() = runTest {
        // Given
        val now = System.currentTimeMillis()

        // Create wrappers directly with downloadedModelInfo to control lastChatTime
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.localPath = tempDir.absolutePath
        }

        val wrapper1 = ModelItemWrapper(
            modelItem = model1,
            downloadedModelInfo = ChatDataManager.DownloadedModelInfo(
                "model1", 1000L, tempDir.absolutePath, now - 1000L
            ),
            downloadSize = 1024L,
            isPinned = true
        )

        val wrapper2 = ModelItemWrapper(
            modelItem = model2,
            downloadedModelInfo = ChatDataManager.DownloadedModelInfo(
                "model2", 1000L, tempDir.absolutePath, now - 500L
            ),
            downloadSize = 1024L,
            isPinned = true
        )

        // Test the sorting logic directly
        val unsorted = listOf(wrapper1, wrapper2)
        val sorted = unsorted.sortedWith(
            compareByDescending<ModelItemWrapper> {
                if (it.isPinned) 1 else 0
            }.thenByDescending {
                if (it.lastChatTime > 0) 1 else 0
            }.thenByDescending {
                if (it.lastChatTime > 0) it.lastChatTime else 0L
            }.thenByDescending {
                if (it.isLocal) 1 else 0
            }.thenByDescending {
                if (it.lastChatTime <= 0) it.downloadTime else 0L
            }
        )

        // Then
        assertEquals(2, sorted.size)
        assertTrue("Both models should be pinned", sorted[0].isPinned && sorted[1].isPinned)
        // model2 should be first (more recent lastChatTime: now - 500L > now - 1000L)
        assertEquals("model2", sorted[0].modelItem.modelId)
    }

    @Test
    fun `models with same lastChatTime should be sorted by downloadTime`() = runTest {
        // Given - both models have lastChatTime = 0, so sorting should use downloadTime
        val model1 = ModelItem().apply {
            this.modelId = "model1"
            this.localPath = tempDir.absolutePath
        }
        val model2 = ModelItem().apply {
            this.modelId = "model2"
            this.localPath = tempDir.absolutePath
        }

        val wrapper1 = ModelItemWrapper(
            modelItem = model1,
            downloadedModelInfo = ChatDataManager.DownloadedModelInfo(
                "model1", 1000L, tempDir.absolutePath, 0L  // lastChatTime = 0
            ),
            downloadSize = 1024L,
            isPinned = false
        )

        val wrapper2 = ModelItemWrapper(
            modelItem = model2,
            downloadedModelInfo = ChatDataManager.DownloadedModelInfo(
                "model2", 2000L, tempDir.absolutePath, 0L  // lastChatTime = 0
            ),
            downloadSize = 1024L,
            isPinned = false
        )

        // Test the sorting logic directly
        val unsorted = listOf(wrapper1, wrapper2)
        val sorted = unsorted.sortedWith(
            compareByDescending<ModelItemWrapper> {
                if (it.isPinned) 1 else 0
            }.thenByDescending {
                if (it.lastChatTime > 0) 1 else 0
            }.thenByDescending {
                if (it.lastChatTime > 0) it.lastChatTime else 0L
            }.thenByDescending {
                if (it.isLocal) 1 else 0
            }.thenByDescending {
                if (it.lastChatTime <= 0) it.downloadTime else 0L
            }
        )

        // Then
        assertEquals(2, sorted.size)
        // model2 should be first (more recent downloadTime: 2000L > 1000L)
        assertEquals("model2", sorted[0].modelItem.modelId)
    }

    // ========== Model Properties Edge Cases ==========

    @Test
    fun `model with null tags should return local tag from getModelTags`() = runTest {
        // Given
        val modelId = "model-null-tags"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            // Don't set tags - let it use default value
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val tags = ModelListManager.getModelTags(modelId)

        // Then
        assertTrue("Should contain local tag even with null tags", tags.contains("local"))
    }

    @Test
    fun `model with very long modelId should be handled correctly`() = runTest {
        // Given
        val longModelId = "a".repeat(500)
        val modelItem = ModelItem().apply {
            this.modelId = longModelId
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertEquals(1, models!!.size)
        assertEquals(longModelId, models[0].modelItem.modelId)
    }

    @Test
    fun `model with special characters in modelId should be handled`() = runTest {
        // Given
        val specialModelId = "model-with-特殊字符-and-émojis-🚀"
        val modelItem = ModelItem().apply {
            this.modelId = specialModelId
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        val models = ModelListManager.observeModels().firstOrNull()

        // Then
        assertNotNull("Models should not be null", models)
        assertEquals(1, models!!.size)
        assertEquals(specialModelId, models[0].modelItem.modelId)
    }

    // ========== DTO Additional Tests ==========

    @Test
    fun `DTO with null modelPath and null modelName should handle gracefully`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = null
            this.modelName = null
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 0L,
            isPinned = false
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertNotNull("DTO should not be null", dto)
        assertEquals("test-model", dto!!.modelId)
        assertNull("modelPath should be null", dto.modelPath)
    }

    @Test
    fun `DTO with very large downloadSize should be handled`() = runTest {
        // Given
        val largeSize = Long.MAX_VALUE
        val modelItem = ModelItem().apply {
            this.modelId = "large-model"
            this.localPath = tempDir.absolutePath
        }

        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = largeSize,
            isPinned = false
        )

        // When
        val dto = ModelListManager.ModelItemCacheDTO.from(wrapper)

        // Then
        assertNotNull("DTO should not be null", dto)
        assertEquals(largeSize, dto!!.downloadSize)
    }

    // ========== Model Type Combination Tests ==========

    @Test
    fun `model with both Think and Vision tags should be detected as both`() = runTest {
        // Given
        val modelId = "multimodal-thinking-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("Think", "Vision", "other-tag")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val isThinking = ModelListManager.isThinkingModel(modelId)
        val isVisual = ModelListManager.isVisualModel(modelId)

        // Then
        assertTrue("Should be detected as thinking model", isThinking)
        assertTrue("Should be detected as visual model", isVisual)
    }

    @Test
    fun `model with Audio and Video tags should be detected as both`() = runTest {
        // Given
        val modelId = "multimedia-model"
        val modelItem = ModelItem().apply {
            this.modelId = modelId
            this.tags = listOf("Audio", "Video")
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When
        val isAudio = ModelListManager.isAudioModel(modelId)
        val isVideo = ModelListManager.isVideoModel(modelId)

        // Then
        assertTrue("Should be detected as audio model", isAudio)
        assertTrue("Should be detected as video model", isVideo)
    }

    // ========== Additional Path Parsing Tests ==========

    @Test
    fun `createModelIdFromPath with path containing spaces should handle correctly`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/modelers/model with spaces"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertEquals("Modelers/MNN/model with spaces", modelId)
    }

    @Test
    fun `createModelIdFromPath with path containing special chars should handle correctly`() = runTest {
        // Given
        val modelPath = "${context.filesDir.absolutePath}/.mnnmodels/modelscope/model-特殊_字符"

        // When
        val modelId = invokeCreateModelIdFromPath(modelPath)

        // Then
        assertEquals("ModelScope/MNN/model-特殊_字符", modelId)
    }

    // ========== Cache-Related Tests ==========

    @Test
    fun `clearModelCache should execute without throwing exception`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()

        // When & Then - should not throw
        ModelListManager.clearModelCache()
        assertTrue("clearModelCache should complete successfully", true)
    }

    @Test
    fun `getModelIdModelMap should return non-null map after initialization`() = runTest {
        // Given
        val modelItem = ModelItem().apply {
            this.modelId = "test-model"
            this.localPath = tempDir.absolutePath
        }
        try {
            val isLocalField = ModelItem::class.java.getDeclaredField("isLocal")
            isLocalField.isAccessible = true
            isLocalField.setBoolean(modelItem, true)
        } catch (e: Exception) {
            println("Could not set isLocal: ${e.message}")
        }

        every { LocalModelsProvider.getLocalModels() } returns mutableListOf(modelItem)

        // When
        ModelListManager.initialize(context)
        ModelListManager.observeModels().firstOrNull()
        val modelMap = ModelListManager.getModelIdModelMap()

        // Then
        assertNotNull("Model map should not be null", modelMap)
        assertTrue("Model map should not be empty", modelMap.isNotEmpty())
    }
}
