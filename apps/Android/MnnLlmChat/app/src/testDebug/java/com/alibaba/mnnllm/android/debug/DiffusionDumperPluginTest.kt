package com.alibaba.mnnllm.android.debug

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class DiffusionDumperPluginTest {

    private class FakeController : DiffusionDebugController {
        var models = listOf(
            DiffusionModelInfo(
                modelId = "ModelScope/MNN/stable-diffusion-v1-5",
                displayName = "Stable Diffusion 1.5",
                isLocal = true,
                localPath = "/tmp/sd15"
            )
        )
        var runResult = DiffusionRunResult(
            success = true,
            elapsedMs = 1234L,
            outputPath = "/sdcard/diffusion_output.jpg",
            outputFileSize = 1024L,
            nativeResult = hashMapOf("total_timeus" to 123456L)
        )
        var lastRequest: DiffusionRunRequest? = null

        override fun listModels(): List<DiffusionModelInfo> = models

        override fun runGeneration(
            request: DiffusionRunRequest,
            progressCallback: (String) -> Unit
        ): DiffusionRunResult {
            lastRequest = request
            progressCallback("10")
            progressCallback("100")
            return runResult
        }
    }

    @Test
    fun `plugin name should be diffusion`() {
        val plugin = DiffusionDumperPlugin(FakeController())
        assertEquals("diffusion", plugin.name)
    }

    @Test
    fun `no args should print usage`() {
        val out = ByteArrayOutputStream()
        val plugin = DiffusionDumperPlugin(FakeController())

        plugin.execute(emptyList(), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Usage: dumpapp diffusion"))
        assertTrue(output.contains("list"))
        assertTrue(output.contains("run"))
    }

    @Test
    fun `list should print available diffusion models`() {
        val out = ByteArrayOutputStream()
        val plugin = DiffusionDumperPlugin(FakeController())

        plugin.execute(listOf("list"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Available diffusion models"))
        assertTrue(output.contains("ModelScope/MNN/stable-diffusion-v1-5"))
        assertTrue(output.contains("downloaded"))
    }

    @Test
    fun `list should show empty message`() {
        val out = ByteArrayOutputStream()
        val controller = FakeController().apply { models = emptyList() }
        val plugin = DiffusionDumperPlugin(controller)

        plugin.execute(listOf("list"), PrintStream(out))

        assertTrue(out.toString().contains("No diffusion models available"))
    }

    @Test
    fun `run should fail when missing required arguments`() {
        val out = ByteArrayOutputStream()
        val plugin = DiffusionDumperPlugin(FakeController())

        plugin.execute(listOf("run"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Usage: dumpapp diffusion run"))
    }

    @Test
    fun `run should parse options and call controller`() {
        val out = ByteArrayOutputStream()
        val controller = FakeController()
        val plugin = DiffusionDumperPlugin(controller)

        plugin.execute(
            listOf(
                "run",
                "ModelScope/MNN/stable-diffusion-v1-5",
                "a cute cat",
                "--output", "/sdcard/out.jpg",
                "--steps", "30",
                "--seed", "7"
            ),
            PrintStream(out)
        )

        val request = controller.lastRequest
        assertEquals("ModelScope/MNN/stable-diffusion-v1-5", request?.modelRef)
        assertEquals("a cute cat", request?.prompt)
        assertEquals("/sdcard/out.jpg", request?.outputPath)
        assertEquals(30, request?.steps)
        assertEquals(7, request?.seed)
    }

    @Test
    fun `run should print success result`() {
        val out = ByteArrayOutputStream()
        val plugin = DiffusionDumperPlugin(FakeController())

        plugin.execute(
            listOf("run", "ModelScope/MNN/stable-diffusion-v1-5", "a cute cat"),
            PrintStream(out)
        )

        val output = out.toString()
        assertTrue(output.contains("=== Diffusion Generation ==="))
        assertTrue(output.contains("Progress: 10%"))
        assertTrue(output.contains("Success: true"))
        assertTrue(output.contains("Output saved to: /sdcard/diffusion_output.jpg"))
    }

    @Test
    fun `run should print error message on failure`() {
        val out = ByteArrayOutputStream()
        val controller = FakeController().apply {
            runResult = DiffusionRunResult(
                success = false,
                errorMessage = "Model path not found",
                elapsedMs = 10L
            )
        }
        val plugin = DiffusionDumperPlugin(controller)

        plugin.execute(
            listOf("run", "bad-model", "a cute cat"),
            PrintStream(out)
        )

        val output = out.toString()
        assertTrue(output.contains("Success: false"))
        assertTrue(output.contains("Error: Model path not found"))
    }
}
