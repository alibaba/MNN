package com.alibaba.mnnllm.android.chat

import android.net.Uri
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import org.junit.Assert.*
import java.text.SimpleDateFormat
import java.util.*

/**
 * Test utility class providing common methods and test data for Chat module testing*/
object ChatTestUtils {

    private var sessionCounter = 0
    private var modelCounter = 0

    /**
     * Create test ChatDataItem*/
    fun createTestChatDataItem(
        type: Int = ChatViewHolders.USER,
        text: String = "Test message",
        time: String = "2024-01-01 10:00:00"
    ): ChatDataItem {
        return ChatDataItem(time, type, text)
    }

    /**
     * Create user message*/
    fun createUserMessage(text: String): ChatDataItem {
        return ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.USER, text)
    }

    /**
     * Create assistant message*/
    fun createAssistantMessage(text: String): ChatDataItem {
        return ChatDataItem("2024-01-01 10:01:00", ChatViewHolders.ASSISTANT, text)
    }

    /**
     * Create message with image*/
    fun createImageMessage(
        text: String = "Check out this image",
        imageUri: Uri = Uri.parse("file:///test/image.jpg")
    ): ChatDataItem {
        return ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.USER, text).apply {
            this.imageUri = imageUri
        }
    }

    /**
     * Create message with audio*/
    fun createAudioMessage(
        text: String = "Voice message",
        audioPath: String = "/test/audio.mp3",
        duration: Float = 5.5f
    ): ChatDataItem {
        return ChatDataItem.createAudioInputData("2024-01-01 10:00:00", text, audioPath, duration)
    }

    /**
     * Create test chat history*/
    fun createTestChatHistory(): List<ChatDataItem> {
        return listOf(
            createUserMessage("Hello"),
            createAssistantMessage("Hi there! How can I help you?"),
            createUserMessage("Can you explain machine learning?"),
            createAssistantMessage("Machine learning is a subset of artificial intelligence...")
        )
    }

    /**
     * Create test generation result*/
    fun createTestGenerateResult(response: String, benchmark: String): HashMap<String, Any> {
        return HashMap<String, Any>().apply {
            put("response", response)
            put("benchmark", benchmark)
            put("tokens", 150)
            put("time", 2.5)
        }
    }

    /**
     * Create test session ID*/
    fun createTestSessionId(): String {
        sessionCounter++
        return "test-session-$sessionCounter"
    }

    /**
     * Create test model ID*/
    fun createTestModelId(prefix: String): String {
        modelCounter++
        return "$prefix/test-model-$modelCounter"
    }

    /**
     * Get test date format*/
    fun getTestDateFormat(): SimpleDateFormat {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    }

    /**
     * Create test timestamp*/
    fun createTestTimestamp(): Long {
        return System.currentTimeMillis()
    }

    /**
     * Validate basic properties of ChatDataItem*/
    fun assertChatDataItemValid(
        item: ChatDataItem,
        expectedType: Int,
        expectedText: String,
        expectedTime: String
    ) {
        assertEquals(expectedType, item.type)
        assertEquals(expectedText, item.text)
        assertEquals(expectedTime, item.time)
        assertFalse(item.loading)
        assertTrue(item.showThinking)
    }

    /**
     * Validate basic properties of generation result*/
    fun assertGenerateResultValid(result: HashMap<String, Any>, expectedResponse: String) {
        assertEquals(expectedResponse, result["response"])
        assertNotNull(result["benchmark"])
        assertNotNull(result["tokens"])
        assertNotNull(result["time"])
    }

    /**
     * Create test URI*/
    fun createTestUri(scheme: String = "file", path: String = "/test/file.txt"): Uri {
        return Uri.parse("$scheme://$path")
    }

    /**
     * Create test audio file path*/
    fun createTestAudioPath(): String {
        return "/test/audio_${System.currentTimeMillis()}.mp3"
    }

    /**
     * Create test image file path*/
    fun createTestImagePath(): String {
        return "/test/image_${System.currentTimeMillis()}.jpg"
    }

    /**
     * Create test error message*/
    fun createTestErrorMessage(message: String = "Test error"): Exception {
        return RuntimeException(message)
    }

    /**
     * Validate error handling*/
    fun assertErrorHandled(
        exception: Exception,
        expectedMessage: String
    ) {
        assertTrue(exception.message?.contains(expectedMessage) == true)
    }

    /**
     * Create test progress info*/
    fun createTestProgressInfo(
        progress: String = "Generating...",
        percentage: Double = 0.5
    ): HashMap<String, Any> {
        return HashMap<String, Any>().apply {
            put("progress", progress)
            put("percentage", percentage)
            put("tokens", 100)
        }
    }

    /**
     * Validate progress info*/
    fun assertProgressInfoValid(
        progressInfo: HashMap<String, Any>,
        expectedProgress: String
    ) {
        assertTrue(progressInfo.containsKey("progress"))
        assertEquals(expectedProgress, progressInfo["progress"])
    }

    /**
     * Create test config path*/
    fun createTestConfigPath(): String {
        return "/test/config_${System.currentTimeMillis()}"
    }

    /**
     * Create test diffusion model path*/
    fun createTestDiffusionPath(): String {
        return "/test/diffusion_${System.currentTimeMillis()}"
    }

    /**
     * Validate session state*/
    fun assertSessionStateValid(
        sessionId: String?,
        expectedSessionId: String?
    ) {
        assertEquals(expectedSessionId, sessionId)
    }

    /**
     * Create test benchmark info*/
    fun createTestBenchmarkInfo(
        generationTime: Double = 2.5,
        tokenCount: Int = 150,
        modelName: String = "test-model"
    ): HashMap<String, Any> {
        return HashMap<String, Any>().apply {
            put("generation_time", generationTime)
            put("token_count", tokenCount)
            put("model_name", modelName)
            put("timestamp", System.currentTimeMillis())
        }
    }

    /**
     * Validate benchmark info*/
    fun assertBenchmarkInfoValid(
        benchmarkInfo: HashMap<String, Any>,
        expectedGenerationTime: Double
    ) {
        assertTrue(benchmarkInfo.containsKey("generation_time"))
        assertEquals(expectedGenerationTime, benchmarkInfo["generation_time"])
    }
} 