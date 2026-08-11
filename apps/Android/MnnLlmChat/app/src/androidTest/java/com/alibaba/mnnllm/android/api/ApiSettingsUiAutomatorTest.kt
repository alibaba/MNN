package com.alibaba.mnnllm.android.api

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import androidx.test.uiautomator.UiDevice
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNotEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ApiSettingsUiAutomatorTest {
    private lateinit var device: UiDevice
    private val timeoutMs = 10_000L

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
        dismissBlockingSystemDialogs()
        ensureChatScreen(context.packageName)
        dismissBlockingSystemDialogs()
    }

    @Test
    fun openApiSettingsAndToggleHttpsSwitch() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val apiSettingsItem = findApiSettingsMenuItem(context.packageName)
        assertNotNull("API Settings menu item not found", apiSettingsItem)
        apiSettingsItem!!.click()

        val httpsSwitch = device.wait(
            Until.findObject(By.res(context.packageName, "switch_use_https_url")),
            timeoutMs
        )
        assertNotNull("HTTPS URL switch not found in API settings sheet", httpsSwitch)

        val initialState = httpsSwitch!!.isChecked
        httpsSwitch.click()

        val switched = device.wait(
            Until.findObject(By.res(context.packageName, "switch_use_https_url")),
            timeoutMs
        )
        assertNotNull("HTTPS URL switch disappeared after click", switched)
        assertNotEquals("HTTPS URL switch state did not change", initialState, switched!!.isChecked)

        // Restore previous state to avoid polluting subsequent tests.
        switched.click()
    }

    private fun findApiSettingsMenuItem(packageName: String): UiObject2? {
        dismissBlockingSystemDialogs()
        val direct = waitForAnyText("API Settings", "API设置", "API 設置")
        if (direct != null) {
            return direct
        }

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

        val byText = waitForAnyText("API Settings", "API设置", "API 設置")
        if (byText != null) {
            return byText
        }

        return device.wait(Until.findObject(By.res(packageName, "menu_item_api_settings")), 1_500L)
    }

    private fun ensureChatScreen(packageName: String) {
        val chatInput = device.wait(Until.findObject(By.res(packageName, "et_message")), 1_500L)
        if (chatInput != null) {
            return
        }

        val preferredModel = waitForAnyText("Qwen3.5-0.8B-MNN", "Qwen3.5-2B-MNN", "Qwen3.5-4B-MNN")
        val modelEntry = preferredModel ?: device.wait(Until.findObject(By.res(packageName, "tvModelTitle")), timeoutMs)
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

    private fun dismissBlockingSystemDialogs() {
        repeat(3) {
            val dismissButton = waitForAnyText(
                "始终允许",
                "仅在使用中允许",
                "仅在使用期间允许",
                "允许",
                "Allow",
                "While using the app",
                "仅此一次",
                "Only this time",
                "拒绝",
                "Don’t allow",
                "Don't allow"
            ) ?: return
            dismissButton.click()
            device.waitForIdle()
            Thread.sleep(300)
        }
    }
}
