package com.alibaba.mnnllm.android.mainsettings

import com.alibaba.mls.api.source.ModelSources
import org.junit.Assert.assertEquals
import org.junit.Test

class DownloadSourceOrderTest {

    @Test
    fun `download source list keeps ModelScope as second option`() {
        val expectedOrder = listOf(
            ModelSources.sourceHuffingFace,
            ModelSources.sourceModelScope,
            ModelSources.sourceModelers
        )

        assertEquals(expectedOrder, ModelSources.sourceList)
    }
}
