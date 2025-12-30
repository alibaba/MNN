package com.alibaba.mls.api.download

import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import java.io.File
import java.nio.file.Files

class FileUtilsTest {

    private lateinit var tempDir: File

    @Before
    fun setUp() {
        tempDir = Files.createTempDirectory("fileUtilsTest").toFile()
    }

    @After
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    @Test
    fun testGetFileSize() {
        // Case 1: Null or non-existent
        assertEquals(0L, FileUtils.getFileSize(null))
        assertEquals(0L, FileUtils.getFileSize(File(tempDir, "non_existent")))

        // Case 2: Single file
        val file1 = File(tempDir, "test1.txt")
        file1.writeText("12345") // 5 bytes
        assertEquals(5L, FileUtils.getFileSize(file1))

        // Case 3: Directory with files
        val subDir = File(tempDir, "subdir")
        subDir.mkdir()
        val file2 = File(subDir, "test2.txt")
        file2.writeText("1234567890") // 10 bytes
        
        // Total: 5 + 10 = 15
        assertEquals(15L, FileUtils.getFileSize(tempDir))
        
        // Test empty directory
        val emptyDir = File(tempDir, "empty")
        emptyDir.mkdir()
        assertEquals(0L, FileUtils.getFileSize(emptyDir))
    }
}
