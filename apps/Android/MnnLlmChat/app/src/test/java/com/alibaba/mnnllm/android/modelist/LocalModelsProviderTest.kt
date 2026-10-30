package com.alibaba.mnnllm.android.modelist

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], manifest = Config.NONE)
class LocalModelsProviderTest {
    private val context: Context get() = ApplicationProvider.getApplicationContext()
    private val imports: File get() = LocalModelsProvider.importedModelsDir(context)

    @After
    fun cleanup() {
        imports.deleteRecursively()
    }

    @Test
    fun `scan discovers published imports but not incomplete staging or missing config`() {
        fun folder(name: String, config: Boolean): File = File(imports, name).apply {
            mkdirs()
            if (config) File(this, "config.json").writeText("{}")
        }
        val complete = folder("Example", true)
        val staging = folder(".import-incomplete", true)
        val missing = folder("MissingConfig", false)
        val ids = LocalModelsProvider.getLocalModels(context).map { it.modelId }
        assertTrue(ids.contains("local/${complete.absolutePath}"))
        assertFalse(ids.contains("local/${staging.absolutePath}"))
        assertFalse(ids.contains("local/${missing.absolutePath}"))
    }

    @Test
    fun `deletion resolves only direct app owned import directories`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val imports = LocalModelsProvider.importedModelsDir(context)
        val owned = File(imports, "Example")
        assertEquals(owned.canonicalFile,
            LocalModelsProvider.importedModelDirectory(context, "local/${owned.absolutePath}"))
        for (id in listOf(
            "local//data/local/tmp/mnn_models/Example",
            "ModelScope/Example",
            "local/${File(imports, ".import-incomplete").absolutePath}",
            "local/${File(imports, "../outside").absolutePath}",
            "local/${File(owned, "nested").absolutePath}"
        )) {
            assertNull(LocalModelsProvider.importedModelDirectory(context, id))
        }
    }

}
