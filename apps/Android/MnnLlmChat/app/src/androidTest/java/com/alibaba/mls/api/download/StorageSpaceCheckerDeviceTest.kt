package com.alibaba.mls.api.download

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Exercises the real StorageManager/StatFs path on the device. */
@RunWith(AndroidJUnit4::class)
class StorageSpaceCheckerDeviceTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun reportsFreeSpaceForTheModelDirectory() {
        val available = StorageSpaceChecker.availableBytes(context, File(context.filesDir, ".mnnmodels/not-created-yet"))
        assertNotNull(available)
        assertTrue(available!! > 0)
    }

    @Test
    fun smallDownloadFitsAndAnImpossibleOneIsRefusedWithAMessage() {
        val dir = File(context.filesDir, ".mnnmodels/not-created-yet")
        assertNull(StorageSpaceChecker.check(context, dir, 1024L))
        val message = StorageSpaceChecker.check(context, dir, 1L shl 50)  // 1 PiB
        assertNotNull(message)
        assertTrue(message!!.isNotBlank())
    }
}
