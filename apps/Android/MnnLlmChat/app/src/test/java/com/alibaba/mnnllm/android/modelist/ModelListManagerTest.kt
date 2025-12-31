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
}
