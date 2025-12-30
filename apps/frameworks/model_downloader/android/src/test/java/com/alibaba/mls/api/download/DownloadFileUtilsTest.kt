package com.alibaba.mls.api.download

import org.junit.Test
import kotlin.test.assertEquals
import java.io.File

class DownloadFileUtilsTest {

    @Test
    fun testRepoFolderName() {
        // repoFolderName(repoId: String?, repoType: String?): String
        
        assertEquals("models--taobao-mnn--Qwen-1.5B", DownloadFileUtils.repoFolderName("taobao-mnn/Qwen-1.5B", "model"))
        assertEquals("models--simple", DownloadFileUtils.repoFolderName("simple", "model"))
        assertEquals("datasetss--mnn--data", DownloadFileUtils.repoFolderName("mnn/data", "datasets")) // typpo in expected? Let's check logic: adds "s" to repoType.
        // Code: parts.add(repoType + "s")
        // So "datasets" -> "datasetss". Implementation detail, we match it.
        
        assertEquals("", DownloadFileUtils.repoFolderName(null, "model"))
        assertEquals("", DownloadFileUtils.repoFolderName("id", null))
        
        // Test ignoring empty parts
        assertEquals("models--a--b", DownloadFileUtils.repoFolderName("a//b", "model"))
    }

    @Test
    fun testGetLastFileName() {
        assertEquals("file.txt", DownloadFileUtils.getLastFileName("path/to/file.txt"))
        assertEquals("file.txt", DownloadFileUtils.getLastFileName("file.txt"))
        assertEquals("", DownloadFileUtils.getLastFileName(""))
        assertEquals("foo", DownloadFileUtils.getLastFileName("/foo"))
    }
    
    @Test
    fun testGetPointerPath() {
        // getPointerPath(storageFolder: File?, commitHash: String, relativePath: String): File
        val root = File("/tmp/models")
        val commit = "abc12345"
        val rel = "config.json"
        
        val file = DownloadFileUtils.getPointerPath(root, commit, rel)
        // Expected: /tmp/models/snapshots/abc12345/config.json
        val expected = File(root, "snapshots/$commit/$rel")
        assertEquals(expected.absolutePath, file.absolutePath)
    }

    @Test
    fun testGetPointerPathParent() {
        val root = File("/tmp/models")
        val sha = "sha123"
        val dir = DownloadFileUtils.getPointerPathParent(root, sha)
        val expected = File(root, "snapshots/$sha")
        assertEquals(expected.absolutePath, dir.absolutePath)
    }
}
