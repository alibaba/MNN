package com.alibaba.mnnllm.android.chat

import android.content.Intent
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class SingleTokenGenerationUiTest {

    private lateinit var device: UiDevice

    @Before
    fun setup() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        device = UiDevice.getInstance(instrumentation)
    }

    @Test
    fun assistantResponseShouldDecodeMoreThanOneToken() {
        val packageName = launchChatActivity()
        sendMessage(packageName, "please output numbers 1 2 3 4 5 only")
        val benchmark = waitForLatestBenchmark(packageName, expectedBenchmarkCount = 1)
        val decodeTokens = parseDecodeTokens(benchmark)
        assertTrue(
            "Expected decode token count > 1 for a normal chat reply, actual=$decodeTokens benchmark=$benchmark",
            decodeTokens > 1
        )
    }

    @Test
    fun assistantResponseShouldContinueOnSecondTurn() {
        val packageName = launchChatActivity()

        sendMessage(packageName, "please output numbers 1 2 3 4 5 only")
        val firstBenchmark = waitForLatestBenchmark(packageName, expectedBenchmarkCount = 1)
        assertTrue(
            "First turn should decode more than one token, benchmark=$firstBenchmark",
            parseDecodeTokens(firstBenchmark) > 1
        )

        sendMessage(packageName, "say hi only")
        val secondBenchmark = waitForLatestBenchmark(packageName, expectedBenchmarkCount = 2)
        assertTrue(
            "Second turn should decode more than one token, benchmark=$secondBenchmark",
            parseDecodeTokens(secondBenchmark) > 1
        )
        assertEquals(
            "Expected a new benchmark entry for the second turn",
            2,
            findBenchmarkViews(packageName).size
        )
    }

    private fun launchChatActivity(): String {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageName = context.packageName
        val configFile = File(
            context.filesDir,
            ".mnnmodels/modelscope/models--MNN--Qwen3.5-0.8B-MNN/snapshots/_no_sha_/config.json"
        )
        assertTrue("Config file missing: ${configFile.absolutePath}", configFile.exists())

        val intent = Intent().apply {
            setClassName(packageName, "com.alibaba.mnnllm.android.chat.ChatActivity")
            putExtra("modelId", "ModelScope/MNN/Qwen3.5-0.8B-MNN")
            putExtra("modelName", "Qwen3.5-0.8B-MNN")
            putExtra("configFilePath", configFile.absolutePath)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        context.startActivity(intent)

        val chatInput = device.wait(Until.findObject(By.res(packageName, "et_message")), 90_000L)
        assertNotNull("Chat input et_message not found after launching ChatActivity", chatInput)
        return packageName
    }

    private fun sendMessage(packageName: String, message: String) {
        val chatInput = device.wait(Until.findObject(By.res(packageName, "et_message")), 10_000L)
        assertNotNull("Chat input et_message not found before send", chatInput)
        chatInput!!.click()
        chatInput.setText(message)

        val sendButton = device.wait(Until.findObject(By.res(packageName, "btn_send")), 10_000L)
        assertNotNull("Send button btn_send not found", sendButton)
        sendButton!!.click()
    }

    private fun waitForLatestBenchmark(packageName: String, expectedBenchmarkCount: Int): String {
        val deadline = SystemClock.uptimeMillis() + 120_000L
        var latestBenchmark = ""
        while (SystemClock.uptimeMillis() < deadline) {
            val benchmarkViews = findBenchmarkViews(packageName)
            if (benchmarkViews.size >= expectedBenchmarkCount) {
                latestBenchmark = benchmarkViews.last().text.orEmpty()
                if (latestBenchmark.contains("Decode:")) {
                    return latestBenchmark
                }
            }
            SystemClock.sleep(500L)
        }
        throw AssertionError(
            "Benchmark text did not appear for turn $expectedBenchmarkCount. Latest benchmark=$latestBenchmark count=${findBenchmarkViews(packageName).size}"
        )
    }

    private fun findBenchmarkViews(packageName: String): List<UiObject2> {
        return device.findObjects(By.res(packageName, "tv_chat_benchmark"))
    }

    private fun parseDecodeTokens(benchmark: String): Int {
        val decodeMatch = Regex("""Decode:\s*[^,]+,\s*(\d+)\s+tokens""").find(benchmark)
        assertNotNull("Decode token count missing from benchmark text: $benchmark", decodeMatch)
        return decodeMatch!!.groupValues[1].toInt()
    }
}
