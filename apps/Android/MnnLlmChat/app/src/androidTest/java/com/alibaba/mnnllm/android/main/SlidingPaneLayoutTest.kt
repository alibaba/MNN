package com.alibaba.mnnllm.android.main

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SlidingPaneLayoutTest {

    private lateinit var device: UiDevice
    private val appPackage = "com.alibaba.mnnllm.android"
    private val timeout = 5000L

    @Before
    fun setUp() {
        device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        device.pressHome()

        val context = ApplicationProvider.getApplicationContext<Context>()
        val intent = context.packageManager.getLaunchIntentForPackage(appPackage)?.apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        context.startActivity(intent)

        device.wait(Until.hasObject(By.pkg(appPackage).depth(0)), timeout)
    }

    @Test
    fun testDrawerSwipeBehaviors() {
        // Wait for main screen to be ready
        device.wait(Until.findObject(By.res(appPackage, "drawer_layout")), timeout)
        Thread.sleep(3000)

        // Swipe from left to right (20% to 80% to mimic a user gesture from middle of screen)
        val width = device.displayWidth
        val height = device.displayHeight
        
        device.swipe((width * 0.2).toInt(), height / 2, (width * 0.8).toInt(), height / 2, 100)

        // Wait for nav_view to appear
        val navView = device.wait(Until.findObject(By.res(appPackage, "nav_view")), timeout)
        assertTrue("Navigation Drawer should be found", navView != null)
        
        // Ensure it's actually visible on screen by checking its visible bounds
        assertTrue("Navigation Drawer should be visible after swiping right", navView.visibleCenter.x > 0)

        // Press back to close the drawer
        device.pressBack()
        Thread.sleep(1500) // Wait for animation
        
        // Wait for it to disappear or move off-screen
        val closedNavView = device.findObject(By.res(appPackage, "nav_view"))
        val isOffScreen = closedNavView == null || closedNavView.visibleBounds.width() <= 0
        assertTrue("Navigation Drawer should close after pressing back", isOffScreen)
    }
}
