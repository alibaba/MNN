package com.alibaba.mnnllm.android.mainsettings

import android.app.Application
import android.widget.TextView
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.ApplicationProvider as MlsApplicationProvider
import com.alibaba.mnnllm.android.R
import io.mockk.every
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Unit tests for StorageManagementActivity: launches activity and verifies summary is loaded.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class StorageManagementActivityTest {

    @Before
    fun setUp() {
        mockkStatic(MlsApplicationProvider::class)
        every { MlsApplicationProvider.get() } returns ApplicationProvider.getApplicationContext<Application>()
    }

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun activity_launches_and_displays_storage_summary() {
        val controller = Robolectric.buildActivity(StorageManagementActivity::class.java)
        val activity = controller.setup().get()

        val tvTotal = activity.findViewById<TextView>(R.id.tv_internal_total)
        val tvUsed = activity.findViewById<TextView>(R.id.tv_internal_used)
        val tvModelFiles = activity.findViewById<TextView>(R.id.tv_model_files)
        val tvMmapCache = activity.findViewById<TextView>(R.id.tv_mmap_cache)

        assertNotNull(tvTotal)
        assertTrue(tvTotal.text.toString().isNotEmpty())
        assertNotNull(tvUsed)
        assertNotNull(tvModelFiles)
        assertNotNull(tvMmapCache)
    }
}
