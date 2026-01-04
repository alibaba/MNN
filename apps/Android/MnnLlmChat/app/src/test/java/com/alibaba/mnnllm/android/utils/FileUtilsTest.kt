package com.alibaba.mnnllm.android.utils

import android.net.Uri
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import org.junit.After
import org.junit.Before
import java.nio.file.Files

@RunWith(RobolectricTestRunner::class)
class FileUtilsTest {

    private lateinit var tempDir: File

    @Before
    fun setup() {
        // Create a temporary directory for testing
        tempDir = Files.createTempDirectory("fileutils_test").toFile()
    }

    @After
    fun tearDown() {
        // Clean up temporary directory
        if (::tempDir.isInitialized && tempDir.exists()) {
            tempDir.deleteRecursively()
        }
    }

    // ========== formatFileSize Tests ==========

    @Test
    fun `formatFileSize with 0 bytes should return 0 B`() {
        // When
        val result = FileUtils.formatFileSize(0L)

        // Then
        assertEquals("0 B", result)
    }

    @Test
    fun `formatFileSize with bytes less than KB should return bytes`() {
        // When
        val result = FileUtils.formatFileSize(512L)

        // Then
        assertEquals("512 B", result)
    }

    @Test
    fun `formatFileSize with exactly 1 KB should return KB`() {
        // When
        val result = FileUtils.formatFileSize(1024L)

        // Then
        assertEquals("1.00 KB", result)
    }

    @Test
    fun `formatFileSize with KB range should return KB`() {
        // When
        val result = FileUtils.formatFileSize(2048L)

        // Then
        assertEquals("2.00 KB", result)
    }

    @Test
    fun `formatFileSize with exactly 1 MB should return MB`() {
        // When
        val result = FileUtils.formatFileSize(1024L * 1024L)

        // Then
        assertEquals("1.00 MB", result)
    }

    @Test
    fun `formatFileSize with MB range should return MB`() {
        // When
        val result = FileUtils.formatFileSize(5 * 1024L * 1024L)

        // Then
        assertEquals("5.00 MB", result)
    }

    @Test
    fun `formatFileSize with exactly 1 GB should return GB`() {
        // When
        val result = FileUtils.formatFileSize(1024L * 1024L * 1024L)

        // Then
        assertEquals("1.00 GB", result)
    }

    @Test
    fun `formatFileSize with GB range should return GB`() {
        // When
        val result = FileUtils.formatFileSize(3 * 1024L * 1024L * 1024L)

        // Then
        assertEquals("3.00 GB", result)
    }

    @Test
    fun `formatFileSize with fractional KB should format correctly`() {
        // When
        val result = FileUtils.formatFileSize(1536L) // 1.5 KB

        // Then
        assertEquals("1.50 KB", result)
    }

    @Test
    fun `formatFileSize with fractional MB should format correctly`() {
        // When
        val result = FileUtils.formatFileSize((2.5 * 1024 * 1024).toLong())

        // Then
        assertEquals("2.50 MB", result)
    }

    @Test
    fun `formatFileSize with boundary 1023 bytes should return bytes`() {
        // When
        val result = FileUtils.formatFileSize(1023L)

        // Then
        assertEquals("1023 B", result)
    }

    @Test
    fun `formatFileSize with large value should return GB`() {
        // When
        val result = FileUtils.formatFileSize(100L * 1024L * 1024L * 1024L)

        // Then
        assertEquals("100.00 GB", result)
    }

    // ========== getPathForUri Tests ==========

    @Test
    fun `getPathForUri with file scheme should return path`() {
        // Given
        val uri = Uri.parse("file:///storage/emulated/0/test.txt")

        // When
        val result = FileUtils.getPathForUri(uri)

        // Then
        assertEquals("/storage/emulated/0/test.txt", result)
    }

    @Test
    fun `getPathForUri with content scheme should return null`() {
        // Given
        val uri = Uri.parse("content://media/external/images/1")

        // When
        val result = FileUtils.getPathForUri(uri)

        // Then
        assertNull(result)
    }

    @Test
    fun `getPathForUri with http scheme should return null`() {
        // Given
        val uri = Uri.parse("http://example.com/file.txt")

        // When
        val result = FileUtils.getPathForUri(uri)

        // Then
        assertNull(result)
    }

    // ========== getFileSize Tests ==========

    @Test
    fun `getFileSize with null file should return 0`() {
        // When
        val result = FileUtils.getFileSize(null)

        // Then
        assertEquals(0L, result)
    }

    @Test
    fun `getFileSize with non-existent file should return 0`() {
        // Given
        val nonExistentFile = File(tempDir, "non_existent.txt")

        // When
        val result = FileUtils.getFileSize(nonExistentFile)

        // Then
        assertEquals(0L, result)
    }

    @Test
    fun `getFileSize with empty file should return 0`() {
        // Given
        val emptyFile = File(tempDir, "empty.txt")
        emptyFile.createNewFile()

        // When
        val result = FileUtils.getFileSize(emptyFile)

        // Then
        assertEquals(0L, result)
    }

    @Test
    fun `getFileSize with single file should return correct size`() {
        // Given
        val testFile = File(tempDir, "test.txt")
        testFile.writeText("Hello World") // 11 bytes

        // When
        val result = FileUtils.getFileSize(testFile)

        // Then
        assertEquals(11L, result)
    }

    @Test
    fun `getFileSize with directory containing files should return total size`() {
        // Given
        val subDir = File(tempDir, "subdir")
        subDir.mkdir()

        val file1 = File(subDir, "file1.txt")
        file1.writeText("12345") // 5 bytes

        val file2 = File(subDir, "file2.txt")
        file2.writeText("67890") // 5 bytes

        // When
        val result = FileUtils.getFileSize(subDir)

        // Then
        assertEquals(10L, result)
    }

    @Test
    fun `getFileSize with nested directories should calculate recursively`() {
        // Given
        val subDir1 = File(tempDir, "subdir1")
        subDir1.mkdir()

        val subDir2 = File(subDir1, "subdir2")
        subDir2.mkdir()

        val file1 = File(subDir1, "file1.txt")
        file1.writeText("abc") // 3 bytes

        val file2 = File(subDir2, "file2.txt")
        file2.writeText("defgh") // 5 bytes

        // When
        val result = FileUtils.getFileSize(subDir1)

        // Then
        assertEquals(8L, result)
    }
}
