package com.alibaba.mnnllm.android.chat

import android.content.Intent
import android.os.Looper
import android.view.View
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mnnllm.android.R
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode
import org.robolectric.Shadows.shadowOf
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
@LooperMode(LooperMode.Mode.PAUSED)
class ChatActivityMockStreamAutoScrollTest {

    @Test
    fun `scroll to bottom should resume auto scroll during mock stream`() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val intent = Intent(context, ChatActivity::class.java).apply {
            putExtra("modelName", "Qwen")
            putExtra("modelId", "mock")
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_ENABLE, true)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_TEXT_LENGTH, 200)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_INTERVAL_MS, 100L)
        }

        val activity = Robolectric.buildActivity(ChatActivity::class.java, intent)
            .setup()
            .get()

        activity.chatListComponent.setUserScrollingForTest(true)
        val scrollButton = activity.findViewById<View>(R.id.btn_scroll_to_bottom)
        scrollButton.performClick()

        assertFalse(activity.chatListComponent.isUserScrollingForTest())

        val initialLength = activity.chatListComponent.recentItem?.displayText?.length ?: 0
        shadowOf(Looper.getMainLooper()).idleFor(350, TimeUnit.MILLISECONDS)
        val nextLength = activity.chatListComponent.recentItem?.displayText?.length ?: 0

        assertTrue(
            "Expected mock stream to append more text, initialLength=$initialLength nextLength=$nextLength",
            nextLength > initialLength
        )
    }
}
