package com.alibaba.mnnllm.android.api

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Until
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
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
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK)
        context.startActivity(intent)
        device.wait(Until.hasObject(By.pkg(context.packageName).depth(0)), timeoutMs)
    }

    @Test
    fun openApiSettingsAndToggleHttpsSwitch() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        openOverflowMenuIfNeeded()

        val apiSettingsItem = waitForAnyText("API Settings", "API设置", "API 設置")
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

    private fun openOverflowMenuIfNeeded() {
        val alreadyOpen = waitForAnyText("API Settings", "API设置", "API 設置") != null
        if (alreadyOpen) return

        val overflow = device.findObject(
            By.descContains("More options")
        ) ?: device.findObject(By.descContains("更多选项"))

        if (overflow != null) {
            overflow.click()
            return
        }
        device.pressMenu()
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
