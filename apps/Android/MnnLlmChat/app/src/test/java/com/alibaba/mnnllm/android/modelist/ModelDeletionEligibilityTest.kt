package com.alibaba.mnnllm.android.modelist

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], manifest = Config.NONE)
class ModelDeletionEligibilityTest {
    private val context: Context get() = ApplicationProvider.getApplicationContext()

    @Test
    fun `app owned import can be deleted without download history`() {
        val model = File(LocalModelsProvider.importedModelsDir(context), "Example")
        assertTrue(ModelDeletionHelper.canDeleteModel(context, "local/${model.absolutePath}", false, false))
    }

    @Test
    fun `adb model is not treated as app owned import`() {
        assertFalse(ModelDeletionHelper.canDeleteModel(
            context, "local//data/local/tmp/mnn_models/Example", false, false))
    }

    @Test
    fun `builtin model cannot be deleted even with download history`() {
        assertFalse(ModelDeletionHelper.canDeleteModel(context, "Builtin/Example", true, true))
        assertFalse(ModelDeletionHelper.canDeleteModel(context, "Builtin/Example", true, false))
        val model = File(LocalModelsProvider.importedModelsDir(context), "Example")
        assertFalse(ModelDeletionHelper.canDeleteModel(context, "local/${model.absolutePath}", false, true))
    }

    @Test
    fun `ordinary downloaded model keeps delete action`() {
        assertTrue(ModelDeletionHelper.canDeleteModel(context, "ModelScope/Example", true, false))
        assertFalse(ModelDeletionHelper.canDeleteModel(context, "ModelScope/Example", false, false))
    }

    @Test
    fun `missing model id cannot be deleted even with download history`() {
        assertFalse(ModelDeletionHelper.canDeleteModel(context, null, false, false))
        assertFalse(ModelDeletionHelper.canDeleteModel(context, null, true, false))
    }
}
