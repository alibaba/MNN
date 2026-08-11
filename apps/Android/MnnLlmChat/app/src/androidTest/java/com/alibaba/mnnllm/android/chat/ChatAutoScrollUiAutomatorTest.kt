package com.alibaba.mnnllm.android.chat

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import com.alibaba.mnnllm.android.R
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ChatAutoScrollUiAutomatorTest {

    private data class ChatProbe(
        val atBottom: Boolean,
        val bottomGapPx: Int,
        val userScrolling: Boolean,
        val assistantChars: Int
    )

    private lateinit var device: UiDevice
    private val timeoutMs = 10_000L
    private val settleMs = 2_000L
    private val streamObserveMs = 300L
    private val streamAdvanceTimeoutMs = 2_000L

    @Before
    fun setup() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        device = UiDevice.getInstance(instrumentation)
        device.setOrientationNatural()
        val context = instrumentation.targetContext
        val intent = Intent().apply {
            setClassName(context.packageName, "com.alibaba.mnnllm.android.chat.ChatActivity")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            putExtra("modelName", "Qwen")
            putExtra("modelId", "mock")
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_ENABLE, true)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_TEXT_LENGTH, 8000)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_INTERVAL_MS, 10L)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_LINE_WIDTH, 6)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_DETACH_AT_CHARS, 400)
            putExtra(ChatActivity.EXTRA_MOCK_STREAM_PAUSE_ON_DETACH, true)
        }
        context.startActivity(intent)
        device.wait(Until.hasObject(By.pkg(context.packageName).depth(0)), timeoutMs)
        assertNotNull(
            "Chat recyclerView not found after launching mock ChatActivity",
            device.wait(Until.findObject(By.res(context.packageName, "recyclerView")), timeoutMs)
        )
    }

    @After
    fun tearDown() {
        device.unfreezeRotation()
    }

    @Test
    fun scrollToBottomRestoresAutoScrollDuringStreaming() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val recycler = device.wait(Until.findObject(By.res(packageName, "recyclerView")), timeoutMs)
        assertNotNull("Chat recyclerView not found", recycler)

        Thread.sleep(1_500L)

        val scrollButton = device.wait(
            Until.findObject(By.desc(context.getString(R.string.scroll_to_bottom))),
            timeoutMs
        )
        assertNotNull("Scroll-to-bottom button did not appear after user scroll", scrollButton)
        val beforeClick = readProbe(packageName)
        assertTrue("Expected chat list to leave bottom after manual swipe: $beforeClick", !beforeClick.atBottom)
        assertTrue("Expected user scrolling state to be true after manual swipe: $beforeClick", beforeClick.userScrolling)
        scrollButton!!.click()

        Thread.sleep(300L)

        var previous = readProbe(packageName)
        assertTrue("Scroll-to-bottom tap should clear user scrolling state: $previous", !previous.userScrolling)
        assertTrue("Scroll-to-bottom tap should bring chat back to bottom: $previous", previous.atBottom)

        repeat(3) { sampleIndex ->
            val current = waitForAssistantCharsToIncrease(packageName, previous.assistantChars)
            assertTrue(
                "Mock stream did not continue after tapping scroll-to-bottom at sample $sampleIndex",
                current.assistantChars > previous.assistantChars
            )
            assertTrue(
                "Chat stopped following new content after tapping scroll-to-bottom at sample $sampleIndex: $current",
                current.atBottom
            )
            assertTrue(
                "User scrolling state unexpectedly stayed true after tapping scroll-to-bottom at sample $sampleIndex: $current",
                !current.userScrolling
            )
            previous = current
        }
    }

    private fun waitForAssistantCharsToIncrease(packageName: String, previousChars: Int): ChatProbe {
        val deadline = System.currentTimeMillis() + streamAdvanceTimeoutMs
        var latest = readProbe(packageName)
        while (System.currentTimeMillis() < deadline) {
            if (latest.assistantChars > previousChars) {
                return latest
            }
            Thread.sleep(streamObserveMs)
            latest = readProbe(packageName)
        }
        return latest
    }

    private fun readProbe(packageName: String): ChatProbe {
        val recycler = device.findObject(By.res(packageName, "recyclerView"))
        assertNotNull("Chat recyclerView not found while reading probe", recycler)
        return parseProbe(recycler!!)
    }

    private fun parseProbe(recycler: UiObject2): ChatProbe {
        val description = recycler.contentDescription?.toString()
        assertNotNull("RecyclerView debug probe missing", description)
        val values = description!!
            .split(';')
            .mapNotNull { entry ->
                val parts = entry.split('=', limit = 2)
                if (parts.size == 2) parts[0] to parts[1] else null
            }
            .toMap()

        return ChatProbe(
            atBottom = values["atBottom"].toBoolean(),
            bottomGapPx = values["bottomGapPx"]?.toIntOrNull() ?: Int.MAX_VALUE,
            userScrolling = values["userScrolling"].toBoolean(),
            assistantChars = values["assistantChars"]?.toIntOrNull() ?: -1
        )
    }
}
