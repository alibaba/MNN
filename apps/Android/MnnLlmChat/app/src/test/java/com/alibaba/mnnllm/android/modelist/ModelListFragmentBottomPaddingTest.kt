package com.alibaba.mnnllm.android.modelist

import android.os.Looper
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.R
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ModelListFragmentBottomPaddingTest {

    @Test
    fun `calculateBottomClearancePadding keeps base padding when recycler ends above tab bar`() {
        val padding = ModelListFragment.calculateBottomClearancePadding(
            basePaddingBottom = 8,
            recyclerBottom = 944,
            bottomNavigationTop = 944
        )

        assertEquals(8, padding)
    }

    @Test
    fun `applyBottomClearanceForViews keeps last item above bottom tab bar in eight-item layout`() {
        val activity = Robolectric.buildActivity(AppCompatActivity::class.java).setup().get()
        val root = FrameLayout(activity)
        val recyclerView = RecyclerView(activity).apply {
            layoutManager = LinearLayoutManager(activity)
            clipToPadding = false
            setPadding(0, 0, 0, 8)
            adapter = MockRowUiStateAdapter(buildMockRowStates())
        }
        val bottomNavigation = View(activity).apply {
            id = com.alibaba.mnnllm.android.R.id.bottom_navigation
        }

        root.addView(
            recyclerView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        root.addView(
            bottomNavigation,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                56,
                Gravity.BOTTOM
            )
        )
        activity.setContentView(root)

        layoutRoot(root, width = 1080, height = 1000)
        recyclerView.scrollToPosition(7)
        shadowOf(Looper.getMainLooper()).idle()
        layoutRoot(root, width = 1080, height = 1000)

        val clippedLastItem = recyclerView.layoutManager?.findViewByPosition(7)
        assertNotNull("Expected last item view before applying overlap fix", clippedLastItem)
        assertTrue(
            "Expected last item bottom to extend under bottom tab bar before fix. itemBottom=${clippedLastItem!!.bottom}, tabTop=${bottomNavigation.top}",
            clippedLastItem.bottom > bottomNavigation.top
        )

        ModelListFragment.applyBottomClearanceForViews(
            recyclerView = recyclerView,
            bottomNavigation = bottomNavigation,
            basePaddingBottom = 8
        )
        recyclerView.scrollToPosition(7)
        shadowOf(Looper.getMainLooper()).idle()
        layoutRoot(root, width = 1080, height = 1000)

        val visibleLastItem = recyclerView.layoutManager?.findViewByPosition(7)
        assertNotNull("Expected last item view after applying overlap fix", visibleLastItem)
        assertEquals(64, recyclerView.paddingBottom)
        assertTrue(
            "Expected last item bottom to stay above bottom tab bar after fix. itemBottom=${visibleLastItem!!.bottom}, tabTop=${bottomNavigation.top}",
            visibleLastItem.bottom <= bottomNavigation.top
        )
    }

    private fun layoutRoot(root: FrameLayout, width: Int, height: Int) {
        val widthSpec = View.MeasureSpec.makeMeasureSpec(width, View.MeasureSpec.EXACTLY)
        val heightSpec = View.MeasureSpec.makeMeasureSpec(height, View.MeasureSpec.EXACTLY)
        root.measure(widthSpec, heightSpec)
        root.layout(0, 0, width, height)
    }

    private fun buildMockRowStates(): List<ModelListItemUiState> {
        return listOf(
            ModelListItemUiState(
                title = "MNN-Sana-Edit-V2",
                tags = listOf("魔搭", "文生图"),
                statusText = "1.55 GB",
                timeText = "16:47"
            ),
            ModelListItemUiState(
                title = "Qwen3.5-0.8B-MNN",
                tags = listOf("魔搭", "深度思考", "图像理解"),
                statusText = "522.28 MB"
            ),
            ModelListItemUiState(
                title = "Qwen3.5-2B-MNN",
                tags = listOf("魔搭", "深度思考", "图像理解"),
                statusText = "50 B"
            ),
            ModelListItemUiState(
                title = "Qwen3.5-4B-MNN",
                tags = listOf("魔搭", "深度思考", "图像理解"),
                statusText = "50 B"
            ),
            ModelListItemUiState(
                title = "Qwen3.5-9B-MNN",
                tags = listOf("魔搭", "深度思考", "图像理解"),
                statusText = "50 B"
            ),
            ModelListItemUiState(
                title = "Qwen3.5-35B-A3B-MNN",
                tags = listOf("魔搭", "深度思考", "图像理解"),
                statusText = "50 B"
            ),
            ModelListItemUiState(
                title = "QwQ-32B-Preview-MNN",
                tags = listOf("魔搭", "深度思考"),
                statusText = "50 B"
            ),
            ModelListItemUiState(
                title = "Qwen3-VL-2B-Thinking-MNN",
                tags = listOf("魔搭", "图像理解", "深度思考"),
                statusText = "50 B"
            )
        )
    }

    private class MockRowUiStateAdapter(
        private val items: List<ModelListItemUiState>
    ) : RecyclerView.Adapter<ModelItemHolder>() {

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelItemHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.recycle_item_model, parent, false)
            return ModelItemHolder(view, NoOpModelItemListener, enableLongClick = false)
        }

        override fun onBindViewHolder(holder: ModelItemHolder, position: Int) {
            holder.render(items[position])
        }

        override fun getItemCount(): Int = items.size
    }

    private object NoOpModelItemListener : ModelItemListener {
        override fun onItemClicked(modelItem: ModelItem) = Unit
        override fun onItemLongClicked(modelItem: ModelItem): Boolean = false
        override fun onItemDeleted(modelItem: ModelItem) = Unit
        override fun onItemUpdate(modelItem: ModelItem) = Unit
    }
}
