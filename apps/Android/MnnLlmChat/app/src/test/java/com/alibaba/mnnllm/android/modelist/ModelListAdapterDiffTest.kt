package com.alibaba.mnnllm.android.modelist

import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(manifest = Config.NONE)
class ModelListAdapterDiffTest {

    @Test
    fun `updateItems dispatches remove when list shrinks without active filters`() {
        val initial = mutableListOf(
            wrapper("ModelScope/A"),
            wrapper("ModelScope/B"),
            wrapper("ModelScope/C")
        )
        val adapter = ModelListAdapter(initial)
        val observer = RecordingObserver()
        adapter.registerAdapterDataObserver(observer)

        adapter.updateItems(
            listOf(
                wrapper("ModelScope/A"),
                wrapper("ModelScope/B")
            )
        )

        assertEquals(2, adapter.itemCount)
        assertEquals(1, observer.removedCount)
    }

    @Test
    fun `updateItems deduplicates repeated model ids before diffing`() {
        val adapter = ModelListAdapter(mutableListOf())

        adapter.updateItems(
            listOf(
                wrapper("ModelScope/A"),
                wrapper("ModelScope/A"),
                wrapper("ModelScope/B")
            )
        )

        assertEquals(2, adapter.itemCount)
    }

    private fun wrapper(modelId: String): ModelItemWrapper {
        return ModelItemWrapper(
            modelItem = ModelItem().apply {
                this.modelId = modelId
                this.modelName = modelId
            }
        )
    }

    private class RecordingObserver : RecyclerView.AdapterDataObserver() {
        var removedCount: Int = 0

        override fun onItemRangeRemoved(positionStart: Int, itemCount: Int) {
            removedCount += itemCount
        }
    }
}
