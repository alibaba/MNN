package com.alibaba.mls.api.download

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.File

class StorageSpaceCheckerTest {
    private val mb = 1024L * 1024

    @Test
    fun `fits when free space covers the rest plus the margin`() {
        assertEquals(0L, StorageSpaceChecker.shortfall(availableBytes = 1024 * mb, remainingBytes = 500 * mb))
        assertEquals(0L, StorageSpaceChecker.shortfall(availableBytes = 700 * mb, remainingBytes = 500 * mb))
    }

    @Test
    fun `reports the missing bytes including the margin`() {
        assertEquals(100 * mb, StorageSpaceChecker.shortfall(availableBytes = 600 * mb, remainingBytes = 500 * mb))
        assertEquals(150 * mb, StorageSpaceChecker.shortfall(availableBytes = 50 * mb, remainingBytes = 0))
    }

    @Test
    fun `negative remainder counts as nothing left to download`() {
        assertEquals(0L, StorageSpaceChecker.shortfall(availableBytes = 300 * mb, remainingBytes = -10))
    }

    @Test
    fun `no context or nothing left to download skips the check`() {
        assertNull(StorageSpaceChecker.check(null, File("."), 10 * mb))
    }
}
