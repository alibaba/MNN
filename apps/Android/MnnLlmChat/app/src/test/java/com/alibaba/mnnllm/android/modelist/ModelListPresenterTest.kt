package com.alibaba.mnnllm.android.modelist

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelListPresenterTest {

    @Test
    fun `shouldCheckForUpdate returns true for remote source prefixes`() {
        assertTrue(ModelListPresenter.shouldCheckForUpdate("ModelScope/MNN/Qwen3.5-0.8B-MNN"))
        assertTrue(ModelListPresenter.shouldCheckForUpdate("HuggingFace/taobao-mnn/Qwen3.5-0.8B-MNN"))
        assertTrue(ModelListPresenter.shouldCheckForUpdate("Huggingface/taobao-mnn/Qwen3.5-0.8B-MNN"))
        assertTrue(ModelListPresenter.shouldCheckForUpdate("Modelers/MNN/Qwen3.5-0.8B-MNN"))
    }

    @Test
    fun `shouldCheckForUpdate returns false for local and unknown ids`() {
        assertFalse(ModelListPresenter.shouldCheckForUpdate("local//data/local/tmp/mnn_models/Qwen3.5-0.8B-MNN"))
        assertFalse(ModelListPresenter.shouldCheckForUpdate("Builtin/Qwen3.5-0.8B-MNN"))
        assertFalse(ModelListPresenter.shouldCheckForUpdate("Qwen3.5-0.8B-MNN"))
        assertFalse(ModelListPresenter.shouldCheckForUpdate(null))
        assertFalse(ModelListPresenter.shouldCheckForUpdate(""))
    }
}
