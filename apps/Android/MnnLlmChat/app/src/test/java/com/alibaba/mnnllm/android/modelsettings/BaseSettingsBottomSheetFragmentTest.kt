package com.alibaba.mnnllm.android.modelsettings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class BaseSettingsBottomSheetFragmentTest {

    @Test
    fun `resolveConfigFilePath prefers explicit config path`() {
        val resolved = BaseSettingsBottomSheetFragment.resolveConfigFilePath(
            modelId = "Builtin/MNN/test",
            configPath = "/tmp/manual_config.json",
            defaultConfigProvider = { "ignored" }
        )
        assertEquals("/tmp/manual_config.json", resolved)
    }

    @Test
    fun `resolveConfigFilePath returns null when no config file is available`() {
        val resolved = BaseSettingsBottomSheetFragment.resolveConfigFilePath(
            modelId = "",
            configPath = null,
            defaultConfigProvider = { null }
        )
        assertNull(resolved)
    }
}

