package com.alibaba.mnnllm.android.modelist

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], manifest = Config.NONE)
class LocalModelImporterTest {
    @get:Rule val temporaryFolder = TemporaryFolder()

    private val root: File get() = File(temporaryFolder.root, "imported_models")
    private fun importer() = LocalModelImporter(root) { Long.MAX_VALUE }

    @Test
    fun `embedded weight model does not need a separate weight file`() = runTest {
        val result = importer().import(directory("Example", file("config.json", "{}"),
            file("llm.mnn", "model with embedded weights")))
        assertTrue(File(result, "llm.mnn").isFile)
    }

    @Test
    fun `oversized config is rejected`() = runTest {
        expectIOException { importer().import(model(config = " ".repeat(1024 * 1024 + 1) + "{}")) }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `copies complete directory and publishes only after copying`() = runTest {
        val model = model(extra = listOf(directory("assets", file("vocab.txt", "tokens"))))
        model.entries.forEach { entry ->
            entry.onOpen = { assertFalse(File(root, model.name).exists()) }
        }

        val result = importer().import(model)

        assertEquals(File(root, "Example"), result)
        assertEquals("tokens", File(result, "assets/vocab.txt").readText())
        assertEquals("{}", File(result, "config.json").readText())
        assertEquals(listOf("Example"), root.listFiles()!!.map { it.name })
    }

    @Test
    fun `duplicate directory never overwrites an existing import`() = runTest {
        val existing = importer().import(model())
        File(existing, "sentinel").writeText("keep")

        expectIOException { importer().import(model()) }

        assertEquals("keep", File(existing, "sentinel").readText())
        assertEquals(listOf("Example"), root.listFiles()!!.map { it.name })
    }

    @Test
    fun `malformed config is rejected without exposing a model`() = runTest {
        expectIOException { importer().import(model(config = "not json")) }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `missing config is rejected without exposing a model`() = runTest {
        expectIOException { importer().import(directory("Example", file("llm.mnn", "model"))) }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `missing required model file is rejected`() = runTest {
        expectIOException { importer().import(directory("Example", file("config.json", "{}"))) }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `insufficient space leaves no partial import`() = runTest {
        expectIOException { LocalModelImporter(root) { 0L }.import(model()) }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `provider read failure removes staging directory`() = runTest {
        val source = model()
        source.entries.last().onOpen = { throw IOException("provider disconnected") }

        expectIOException { importer().import(source) }

        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `cancellation propagates and removes staging directory`() = runTest {
        val source = model()
        source.entries.last().onOpen = { throw CancellationException("cancel import") }
        try {
            importer().import(source)
            throw AssertionError("Expected cancellation")
        } catch (_: CancellationException) {
            assertNoPublishedOrTemporaryFiles()
        }
    }

    @Test
    fun `cancelled coroutine stops copy and cleans staging`() = runTest {
        val source = model()
        lateinit var job: Job
        source.entries.last().onOpen = { job.cancel() }
        job = launch { importer().import(source) }
        job.join()
        assertTrue(job.isCancelled)
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `failed copy can be retried without changing the source`() = runTest {
        val source = model()
        val last = source.entries.last()
        last.onOpen = { throw IOException("temporary failure") }
        expectIOException { importer().import(source) }
        last.onOpen = {}

        val result = importer().import(source)

        assertEquals("{}", File(result, "config.json").readText())
        assertEquals("{}", source.entries.first().open().bufferedReader().use { it.readText() })
    }

    @Test
    fun `model reference must stay inside the imported directory`() = runTest {
        for (path in listOf("../outside.mnn", "/outside.mnn", "missing.mnn")) {
            expectIOException { importer().import(model(config = """{"llm_model":"$path"}""")) }
            assertNoPublishedOrTemporaryFiles()
        }
    }

    @Test
    fun `non object config and empty model file are rejected`() = runTest {
        expectIOException { importer().import(model(config = "[]")) }
        assertNoPublishedOrTemporaryFiles()
        expectIOException {
            importer().import(directory("Example", file("config.json", "{}"), file("llm.mnn", "")))
        }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `explicit weight path must exist and stay inside the model`() = runTest {
        for (path in listOf("../outside.weight", "missing.weight")) {
            expectIOException { importer().import(model(config = """{"llm_weight":"$path"}""")) }
            assertNoPublishedOrTemporaryFiles()
        }
    }

    @Test
    fun `custom relative model path is preserved`() = runTest {
        val source = directory("Example", file("config.json", """{"llm_model":"nested/model.mnn"}"""),
            directory("nested", file("model.mnn", "model")))
        val result = importer().import(source)
        assertEquals("model", File(result, "nested/model.mnn").readText())
    }

    @Test
    fun `interrupted staging is cleaned before the next import`() = runTest {
        val interrupted = File(root, ".import-interrupted")
        interrupted.mkdirs()
        File(interrupted, "config.json").writeText("{}")
        val result = importer().import(model())
        assertFalse(interrupted.exists())
        assertEquals(listOf(result), root.listFiles()!!.toList())
    }

    @Test
    fun `hidden child assets are preserved but hidden root is rejected`() = runTest {
        val result = importer().import(model(extra = listOf(file(".metadata", "metadata"))))
        assertEquals("metadata", File(result, ".metadata").readText())
        expectIOException {
            importer().import(directory(".hidden", file("config.json", "{}"), file("llm.mnn", "model")))
        }
        assertFalse(File(root, ".hidden").exists())
    }

    @Test
    fun `duplicate provider child names are rejected rather than overwritten`() = runTest {
        expectIOException { importer().import(model(extra = listOf(file("llm.mnn", "replacement")))) }
        assertNoPublishedOrTemporaryFiles()
    }

    @Test
    fun `unsafe provider names cannot escape import directory`() = runTest {
        for (name in listOf("../outside", "/outside", ".", "..")) {
            expectIOException { importer().import(model(extra = listOf(file(name, "bad")))) }
            assertNoPublishedOrTemporaryFiles()
            assertFalse(File(temporaryFolder.root, "outside").exists())
        }
    }

    private fun assertNoPublishedOrTemporaryFiles() {
        assertTrue(root.listFiles().isNullOrEmpty())
    }

    private suspend fun expectIOException(block: suspend () -> Unit) {
        try {
            block()
            throw AssertionError("Expected IOException")
        } catch (_: IOException) {
            // Expected: failed imports must leave no visible model.
        }
    }

    private fun model(config: String = "{}", extra: List<Node> = emptyList()): Node =
        directory("Example", file("config.json", config), file("llm.mnn", "model"),
            file("llm.mnn.weight", "weights"), file("tokenizer.txt", "tokens"),
            file("llm_config.json", "{}"), *extra.toTypedArray())

    private fun directory(name: String, vararg entries: Node) = Node(name, true, entries.toList())
    private fun file(name: String, contents: String) = Node(name, false, contents = contents)

    private class Node(
        override val name: String,
        override val isDirectory: Boolean,
        val entries: List<Node> = emptyList(),
        private val contents: String = ""
    ) : LocalModelImporter.Source {
        var onOpen: () -> Unit = {}
        override fun children(): List<LocalModelImporter.Source> = entries
        override fun open(): InputStream {
            onOpen()
            return ByteArrayInputStream(contents.toByteArray())
        }
    }
}
