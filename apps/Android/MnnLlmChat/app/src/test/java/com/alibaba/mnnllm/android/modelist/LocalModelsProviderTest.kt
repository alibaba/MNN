package com.alibaba.mnnllm.android.modelist

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
@Config(sdk = [34], manifest = Config.NONE)
class LocalModelsProviderTest {

    private lateinit var root: File

    @Before
    fun setUp() {
        root = Files.createTempDirectory("local_models").toFile()
    }

    @After
    fun tearDown() {
        root.deleteRecursively()
    }

    private fun writeModelDir(parent: File, name: String, visual: Boolean = false): File {
        val dir = File(parent, name)
        dir.mkdirs()
        File(dir, "config.json").writeText("{}")
        File(dir, "llm.mnn").writeText("weights")
        if (visual) {
            File(dir, "llm_config.json").writeText("""{"is_visual": true}""")
        }
        return dir
    }

    @Test
    fun subFoldersWithConfigJsonBecomeLocalModelItems() {
        val modelDir = writeModelDir(root, "stable-diffusion-v1-5-mnn-opencl")

        val items = LocalModelsProvider.modelsIn(root)

        assertEquals(1, items.size)
        assertEquals(modelDir.absolutePath, items[0].localPath)
        assertEquals("local/${modelDir.absolutePath}", items[0].modelId)
        assertTrue(items[0].isLocal)
    }

    @Test
    fun foldersWithoutConfigJsonAreNotModels() {
        val dir = File(root, "just-files")
        dir.mkdirs()
        File(dir, "llm.mnn").writeText("weights")

        assertTrue(LocalModelsProvider.modelsIn(root).isEmpty())
    }

    @Test
    fun theRootItselfCountsWhenItIsAModelFolder() {
        File(root, "config.json").writeText("{}")

        val items = LocalModelsProvider.modelsIn(root)

        assertEquals(1, items.size)
        assertEquals(root.absolutePath, items[0].localPath)
    }

    @Test
    fun aRootThatIsAModelStillListsItsModelSubFolders() {
        File(root, "config.json").writeText("{}")
        val nested = writeModelDir(root, "MNN-Sana-Edit-V2")

        val items = LocalModelsProvider.modelsIn(root)

        assertEquals(2, items.size)
        assertTrue(items.map { it.localPath }.containsAll(listOf(root.absolutePath, nested.absolutePath)))
    }

    @Test
    fun aMissingRootYieldsNothingInsteadOfThrowing() {
        assertTrue(LocalModelsProvider.modelsIn(File(root, "absent")).isEmpty())
        assertTrue(LocalModelsProvider.modelsIn(File(root, "config.json")).isEmpty())
    }

    @Test
    fun capabilityTagsComeFromTheModelsOwnLlmConfig() {
        writeModelDir(root, "omni-model", visual = true)

        assertEquals(listOf("Vision"), LocalModelsProvider.modelsIn(root)[0].tags)
    }
}
