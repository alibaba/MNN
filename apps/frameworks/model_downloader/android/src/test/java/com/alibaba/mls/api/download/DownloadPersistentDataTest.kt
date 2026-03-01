package com.alibaba.mls.api.download

import com.alibaba.mls.api.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], manifest = Config.NONE)
class DownloadPersistentDataTest {

    @Test
    fun saveAndGetTotalSize_shouldShareAcrossSources_withSameRepoName() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val hfModelId = "HuggingFace/taobao-mnn/MiniMind2-MNN"
        val msModelId = "ModelScope/MNN/MiniMind2-MNN"

        DownloadPersistentData.saveDownloadSizeTotal(context, hfModelId, 111L)
        val hfTotal = DownloadPersistentData.getDownloadSizeTotal(context, hfModelId)
        val msTotal = DownloadPersistentData.getDownloadSizeTotal(context, msModelId)

        assertEquals(111L, hfTotal)
        assertEquals(
            "Total size should be shared for same repo tail name across sources",
            hfTotal,
            msTotal
        )
    }
}
