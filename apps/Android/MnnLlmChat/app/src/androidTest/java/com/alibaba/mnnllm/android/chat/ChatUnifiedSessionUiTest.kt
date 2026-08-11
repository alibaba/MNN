package com.alibaba.mnnllm.android.chat

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import androidx.test.uiautomator.UiDevice
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * UI test for Chat flow with unified session management (LlmRuntimeController).
 * Verifies that typing and sending a message works; optionally waits for assistant response.
 */
@RunWith(AndroidJUnit4::class)
class ChatUnifiedSessionUiTest {

    private lateinit var device: UiDevice
    private val timeoutMs = 10_000L
    private val responseTimeoutMs = 60_000L

    @Before
    fun setup() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        device = UiDevice.getInstance(instrumentation)
        val context = instrumentation.targetContext
        val intent = context.packageManager.getLaunchIntentForPackage(context.packageName)
            ?: throw IllegalStateException("Launch intent not found for ${context.packageName}")
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        context.startActivity(intent)
        device.wait(Until.hasObject(By.pkg(context.packageName).depth(0)), timeoutMs)
        ensureChatScreen(context.packageName)
    }

    @Test
    fun sendMessageAndVerifyUserBubbleAppears() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val chatInput = device.wait(
            Until.findObject(By.res(packageName, "et_message")),
            timeoutMs
        )
        assertNotNull("Chat input et_message not found", chatInput)
        chatInput!!.click()
        chatInput.setText("hi")

        val sendButton = device.wait(
            Until.findObject(By.res(packageName, "btn_send")),
            timeoutMs
        )
        assertNotNull("Send button btn_send not found", sendButton)
        sendButton!!.click()

        val userMessage = device.wait(
            Until.findObject(By.text("hi")),
            timeoutMs
        )
        assertNotNull(
            "User message 'hi' did not appear in chat after send; unified session send flow may be broken",
            userMessage
        )
    }

    @Test
    fun sendMessageAndWaitForAssistantResponse() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val chatInput = device.wait(
            Until.findObject(By.res(packageName, "et_message")),
            timeoutMs
        )
        assertNotNull("Chat input et_message not found", chatInput)
        chatInput!!.click()
        chatInput.setText("say hello")

        val sendButton = device.wait(
            Until.findObject(By.res(packageName, "btn_send")),
            timeoutMs
        )
        assertNotNull("Send button btn_send not found", sendButton)
        sendButton!!.click()

        val userMessage = device.wait(
            Until.findObject(By.text("say hello")),
            timeoutMs
        )
        assertNotNull("User message did not appear", userMessage)

        val loadingAppeared = device.wait(
            Until.findObject(By.res(packageName, "view_assistant_loading")),
            responseTimeoutMs
        ) != null
        if (loadingAppeared) {
            device.wait(
                Until.gone(By.res(packageName, "view_assistant_loading")),
                responseTimeoutMs
            )
        }
        val recycler = device.findObject(By.res(packageName, "recyclerView"))
        assertTrue(
            "Expected loading indicator or at least 2 chat items (user + assistant); model may be missing",
            recycler?.childCount?.let { it >= 2 } == true || loadingAppeared
        )
    }

    private fun ensureChatScreen(packageName: String) {
        val chatInput = device.wait(Until.findObject(By.res(packageName, "et_message")), 1_500L)
        if (chatInput != null) {
            return
        }

        val preferredModel = waitForAnyText("Qwen3.5-0.8B-MNN", "Qwen3.5-2B-MNN", "Qwen3.5-4B-MNN")
        val modelEntry = preferredModel
            ?: device.wait(Until.findObject(By.res(packageName, "tvModelTitle")), timeoutMs)
        assertNotNull("Model entry not found to enter chat screen", modelEntry)
        modelEntry!!.click()

        val entered = device.wait(Until.findObject(By.res(packageName, "et_message")), timeoutMs)
        assertNotNull("Failed to enter chat screen from model list", entered)
    }

    private fun waitForAnyText(vararg candidates: String): UiObject2? {
        candidates.forEach { text ->
            val found = device.wait(Until.findObject(By.text(text)), 1_500L)
            if (found != null) {
                return found
            }
        }
        return null
    }
}
