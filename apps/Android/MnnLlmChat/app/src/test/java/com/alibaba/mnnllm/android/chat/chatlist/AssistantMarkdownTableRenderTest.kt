package com.alibaba.mnnllm.android.chat.chatlist

import android.os.Looper
import android.text.Spanned
import android.view.LayoutInflater
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.Shadows.shadowOf

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class AssistantMarkdownTableRenderTest {

    @Test
    fun `assistant markdown should render gfm tables with table spans`() {
        val activity = Robolectric.buildActivity(AppCompatActivity::class.java).setup().get()
        activity.setTheme(R.style.AppTheme)
        val itemView = LayoutInflater.from(activity)
            .inflate(R.layout.item_holder_assistant, FrameLayout(activity), false)
        val holder = ChatViewHolders.AssistantViewHolder(itemView)
        val data = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            displayText = """
                | Model | Params |
                | --- | --- |
                | Qwen3 30B-A3B | 30B |
                | GPT-OSS 20B | 20B |
            """.trimIndent()
        }

        holder.bind(data, "Qwen", null)
        shadowOf(Looper.getMainLooper()).idle()

        val chatText = itemView.findViewById<TextView>(R.id.tv_chat_text)
        val rendered = chatText.text as Spanned
        val spanNames = rendered.getSpans(0, rendered.length, Any::class.java)
            .map { it.javaClass.simpleName }

        assertTrue(
            "Expected table-related spans for rendered markdown table, actual=$spanNames text=${rendered}",
            spanNames.any { it.contains("Table", ignoreCase = true) }
        )
    }
}
