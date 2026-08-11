package com.alibaba.mnnllm.android.modelsettings

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
 * UI test guarding against issue #4259: model settings from BOTH home and ChatActivity
 * must not corrupt config. After settings save, merged config (llm_model, llm_weight)
 * must remain valid.
 *
 * Flows:
 * - Home: long-press model -> Settings -> edit System Prompt -> save -> enter chat -> send
 * - Chat: enter chat -> menu Model Settings -> edit System Prompt -> save -> send
 *
 * Caller (smoke script) must run `dumpapp config dump <modelId>` after each flow
 * to verify merged config is correct.
 */
@RunWith(AndroidJUnit4::class)
class ModelSettingsConfigUiAutomatorTest {

    private lateinit var device: UiDevice
    private val timeoutMs = 15_000L
    private val modelReadyTimeoutMs = 60_000L

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
        ensureHomeScreen(context.packageName)
    }

    @Test
    fun homeSettingsSaveThenChatSend_noCrash() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val modelEntry = findModelEntry(packageName)
        assertNotNull("Model entry not found on home screen", modelEntry)

        openSettingsFromHome(modelEntry!!, packageName)
        editSystemPromptAndSave(packageName, "home smoke #4259")
        enterChatAndSend(packageName, "hi")
    }

    @Test
    fun homeSettingsSaveReopen_showsSavedSystemPrompt() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val modelEntry = findModelEntry(packageName)
        assertNotNull("Model entry not found on home screen", modelEntry)

        val savedPrompt = "persist check #4259"
        openSettingsFromHome(modelEntry!!, packageName)
        editSystemPromptAndSave(packageName, savedPrompt)

        openSettingsFromHome(findModelEntry(packageName)!!, packageName)
        scrollSettingsToSystemPrompt(packageName)
        val systemPromptEdit = device.wait(
            Until.findObject(By.res(packageName, "editTextSystemPrompt")),
            timeoutMs
        )
        assertNotNull("System Prompt edit not found when reopening settings", systemPromptEdit)
        val displayedText = systemPromptEdit!!.text ?: ""
        assertTrue(
            "Saved system prompt not displayed after reopen (expected containing '$savedPrompt', got '$displayedText')",
            displayedText.contains(savedPrompt)
        )
    }

    @Test
    fun chatSettingsSaveThenSend_noCrash() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val modelEntry = findModelEntry(packageName)
        assertNotNull("Model entry not found on home screen", modelEntry)
        modelEntry!!.click()
        Thread.sleep(300)

        val chatInput = device.wait(
            Until.findObject(By.res(packageName, "et_message")),
            modelReadyTimeoutMs
        )
        assertNotNull("Chat input not found (model may not be ready)", chatInput)

        openSettingsFromChat(packageName)
        editSystemPromptAndSave(packageName, "chat smoke #4259")
        sendMessage(packageName, "hi")
    }

    private fun openSettingsFromHome(modelEntry: UiObject2, packageName: String) {
        modelEntry.click(1500)
        Thread.sleep(500)
        val settingsItem = waitForAnyText("Settings", "设置")
        assertNotNull("Settings menu item not found after long-press", settingsItem)
        settingsItem!!.click()
        Thread.sleep(800)
    }

    private fun openSettingsFromChat(packageName: String) {
        val overflow = device.findObject(By.descContains("More options"))
            ?: device.findObject(By.descContains("更多选项"))
            ?: device.findObject(By.descContains("更多"))
            ?: device.findObject(By.descContains("Menu"))
            ?: device.findObject(By.descContains("菜单"))
            ?: device.findObject(By.descContains("Options"))
        if (overflow != null) {
            overflow.click()
        } else {
            device.pressMenu()
        }
        Thread.sleep(1200)

        var settingsItem = device.wait(Until.findObject(By.res(packageName, "menu_item_model_settings")), 4_000L)
            ?: findSettingsExcludingApi(packageName)
        if (settingsItem == null) {
            device.swipe(device.displayWidth / 2, device.displayHeight * 3 / 4, device.displayWidth / 2, device.displayHeight / 4, 10)
            Thread.sleep(500)
            settingsItem = device.wait(Until.findObject(By.res(packageName, "menu_item_model_settings")), 3_000L)
                ?: findSettingsExcludingApi(packageName)
        }
        if (settingsItem == null) {
            settingsItem = device.wait(Until.findObject(By.text("Settings")), 2_000L)
                ?: device.wait(Until.findObject(By.text("设置")), 2_000L)
            if (settingsItem != null) {
                val t = settingsItem.text ?: ""
                if (t.contains("API", ignoreCase = true)) settingsItem = null
            }
        }
        assertNotNull("Model Settings menu item not found (menu may need scroll)", settingsItem)
        settingsItem!!.click()
        Thread.sleep(800)
    }

    private fun findSettingsExcludingApi(packageName: String): UiObject2? {
        val byText = waitForAnyText("Settings", "设置")
        if (byText != null) {
            val text = byText.text ?: ""
            if (!text.contains("API", ignoreCase = true)) return byText
        }
        return null
    }

    /** Scroll settings sheet down to reveal System Prompt (may be below fold on small screens). */
    private fun scrollSettingsToSystemPrompt(packageName: String) {
        val scrollView = device.wait(
            Until.findObject(By.res(packageName, "settings_scroll_view")),
            3_000L
        ) ?: return
        val bounds = scrollView.visibleBounds
        val centerX = bounds.centerX()
        val centerY = bounds.centerY()
        val topY = bounds.top + 100
        val bottomY = bounds.bottom - 100
        for (i in 0..2) {
            if (device.findObject(By.res(packageName, "editTextSystemPrompt")) != null) return
            device.swipe(centerX, bottomY, centerX, topY, 8)
            Thread.sleep(400)
        }
    }

    private fun editSystemPromptAndSave(packageName: String, promptSuffix: String) {
        scrollSettingsToSystemPrompt(packageName)
        val systemPromptEdit = device.wait(
            Until.findObject(By.res(packageName, "editTextSystemPrompt")),
            timeoutMs
        )
        assertNotNull("System Prompt edit not found in settings sheet", systemPromptEdit)
        systemPromptEdit!!.click()
        systemPromptEdit.clear()
        systemPromptEdit.setText("smoke test prompt $promptSuffix")

        val doneButton = waitForAnyText("Done", "完成")
        assertNotNull("Done button not found in settings sheet", doneButton)
        doneButton!!.click()
        Thread.sleep(500)
    }

    private fun enterChatAndSend(packageName: String, message: String) {
        val modelEntryAgain = findModelEntry(packageName)
        assertNotNull("Model entry not found after closing settings", modelEntryAgain)
        modelEntryAgain!!.click()
        Thread.sleep(300)

        val chatInput = device.wait(
            Until.findObject(By.res(packageName, "et_message")),
            modelReadyTimeoutMs
        )
        assertNotNull("Chat input not found after entering chat", chatInput)
        sendMessage(packageName, message)
    }

    private fun sendMessage(packageName: String, message: String) {
        val chatInput = device.wait(
            Until.findObject(By.res(packageName, "et_message")),
            timeoutMs
        )
        assertNotNull("Chat input not found", chatInput)
        chatInput!!.click()
        chatInput.setText(message)

        val sendButton = device.wait(
            Until.findObject(By.res(packageName, "btn_send")),
            timeoutMs
        )
        assertNotNull("Send button not found", sendButton)
        sendButton!!.click()

        val userMessage = device.wait(Until.findObject(By.text(message)), timeoutMs)
        assertNotNull("User message did not appear; app may have crashed (issue #4259)", userMessage)

        val chatInputStill = device.wait(
            Until.findObject(By.res(packageName, "et_message")),
            5_000L
        )
        assertNotNull("Chat input disappeared after send; app may have crashed (issue #4259)", chatInputStill)
    }

    // --- Prompt cache toggle helpers ---

    private fun findPromptCacheToggle(packageName: String): UiObject2? {
        return device.wait(Until.findObject(By.res(packageName, "promptCacheToggle")), timeoutMs)
    }

    private fun setPromptCache(packageName: String, enabled: Boolean) {
        val toggle = findPromptCacheToggle(packageName)
        assertNotNull("Prompt cache toggle not found", toggle)
        if (toggle!!.isChecked != enabled) {
            toggle.click()
            Thread.sleep(300)
        }
    }

    private fun saveSettingsById(packageName: String) {
        val saveButton = device.wait(Until.findObject(By.res(packageName, "button_save")), timeoutMs)
        assertNotNull("Save button not found", saveButton)
        saveButton!!.click()
        Thread.sleep(500)
    }

    private fun resetSettingsById(packageName: String) {
        val resetButton = device.wait(Until.findObject(By.res(packageName, "button_reset")), timeoutMs)
        assertNotNull("Reset button not found", resetButton)
        resetButton!!.click()
        Thread.sleep(800)
    }

    @Test
    fun homeSettingsTogglePromptCache_reopenShowsSavedState() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val modelEntry = findModelEntry(packageName)
        assertNotNull("Model entry not found on home screen", modelEntry)

        openSettingsFromHome(modelEntry!!, packageName)
        setPromptCache(packageName, true)
        saveSettingsById(packageName)

        openSettingsFromHome(findModelEntry(packageName)!!, packageName)
        val toggle = findPromptCacheToggle(packageName)
        assertNotNull("Prompt cache toggle not found on reopen", toggle)
        assertTrue("Prompt cache toggle should be checked after save", toggle!!.isChecked)

        // Clean up: restore to default (false)
        setPromptCache(packageName, false)
        saveSettingsById(packageName)
    }

    @Test
    fun homeSettingsReset_restoresPromptCacheDefault() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName

        val modelEntry = findModelEntry(packageName)
        assertNotNull("Model entry not found on home screen", modelEntry)

        // Set to non-default (true)
        openSettingsFromHome(modelEntry!!, packageName)
        setPromptCache(packageName, true)
        saveSettingsById(packageName)

        // Reopen, reset to defaults, and wait for the toggle to reflect them.
        openSettingsFromHome(findModelEntry(packageName)!!, packageName)
        resetSettingsById(packageName)
        val resetToggle = device.wait(
            Until.findObject(By.res(packageName, "promptCacheToggle").checked(false)),
            timeoutMs
        )
        assertNotNull(
            "Prompt cache toggle should be unchecked after reset (default is false)",
            resetToggle
        )
    }

    private fun ensureHomeScreen(packageName: String) {
        val chatInput = device.wait(Until.findObject(By.res(packageName, "et_message")), 1_500L)
        if (chatInput != null) {
            device.pressBack()
            Thread.sleep(500)
        }
    }

    private fun findModelEntry(packageName: String): UiObject2? {
        val preferred = waitForAnyText("Qwen3.5-0.8B-MNN", "Qwen3.5-2B-MNN", "Qwen3.5-4B-MNN")
        if (preferred != null) return preferred
        return device.wait(Until.findObject(By.res(packageName, "tvModelTitle")), timeoutMs)
    }

    private fun waitForAnyText(vararg candidates: String): UiObject2? {
        candidates.forEach { text ->
            val found = device.wait(Until.findObject(By.text(text)), 2_000L)
            if (found != null) return found
        }
        return null
    }
}
