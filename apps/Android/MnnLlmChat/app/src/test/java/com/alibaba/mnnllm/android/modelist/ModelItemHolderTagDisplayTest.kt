package com.alibaba.mnnllm.android.modelist

import android.view.LayoutInflater
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.modelmarket.ModelMarketConfig
import com.alibaba.mnnllm.android.modelmarket.TagMapper
import io.mockk.mockk
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ModelItemHolderTagDisplayTest {

    @Test
    fun `bind should show think and vision tags in my-model list item`() {
        TagMapper.initializeFromConfig(
            ModelMarketConfig(
                version = "1",
                tagTranslations = mapOf("Think" to "深度思考", "Vision" to "图像理解"),
                quickFilterTags = emptyList(),
                vendorOrder = emptyList(),
                llmModels = emptyList(),
                ttsModels = emptyList(),
                asrModels = emptyList(),
                libs = emptyList()
            )
        )

        val activity = Robolectric.buildActivity(AppCompatActivity::class.java).setup().get()
        activity.setTheme(R.style.AppTheme)
        val itemView = LayoutInflater.from(activity)
            .inflate(R.layout.recycle_item_model, FrameLayout(activity), false)
        val holder = ModelItemHolder(itemView, mockk(relaxed = true), enableLongClick = false)

        val modelItem = ModelItem().apply {
            modelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
            modelName = "Qwen3.5-0.8B-MNN"
            tags = listOf("Think", "Vision")
        }
        val wrapper = ModelItemWrapper(
            modelItem = modelItem,
            downloadedModelInfo = null,
            downloadSize = 1L,
            isPinned = false
        )

        holder.bind(wrapper)

        val tagsLayout = itemView.findViewById<com.alibaba.mnnllm.android.widgets.TagsLayout>(R.id.tagsLayout)
        val renderedTags = (0 until tagsLayout.childCount).mapNotNull {
            (tagsLayout.getChildAt(it) as? TextView)?.text?.toString()
        }

        assertTrue(
            "Expected Think tag, actual=$renderedTags",
            renderedTags.any { it == "Think" || it == "深度思考" }
        )
        assertTrue(
            "Expected Vision tag, actual=$renderedTags",
            renderedTags.any { it == "Vision" || it == "图像理解" }
        )
    }
}
