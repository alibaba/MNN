package com.alibaba.mnnllm.android.chat

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ChatRouterTest {

    @Test
    fun resolveDiffusionDir_keepsDirectoryPath() {
        val dir = Files.createTempDirectory("chat-router-dir").toFile()
        try {
            val resolved = ChatRouter.resolveDiffusionDir(dir.absolutePath)
            assertEquals(dir.absolutePath, resolved)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun resolveDiffusionDir_usesParentWhenGivenConfigFile() {
        val dir = Files.createTempDirectory("chat-router-file").toFile()
        val configFile = File(dir, "config.json")
        try {
            configFile.writeText("{}")
            val resolved = ChatRouter.resolveDiffusionDir(configFile.absolutePath)
            assertEquals(dir.absolutePath, resolved)
        } finally {
            dir.deleteRecursively()
        }
    }
}
