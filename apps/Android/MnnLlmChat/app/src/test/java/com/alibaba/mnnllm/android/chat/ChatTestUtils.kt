package com.alibaba.mnnllm.android.chat

import android.net.Uri
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import org.junit.Assert.*
import java.text.SimpleDateFormat
import java.util.*

/**
 * 测试工具类，提供Chat模块测试中常用的工具方法和测试数据
 */
object ChatTestUtils {

    private var sessionCounter = 0
    private var modelCounter = 0

    /**
     * 创建测试用的ChatDataItem
     */
    fun createTestChatDataItem(
        type: Int = ChatViewHolders.USER,
        text: String = "Test message",
        time: String = "2024-01-01 10:00:00"
    ): ChatDataItem {
        return ChatDataItem(time, type, text)
    }

    /**
     * 创建用户消息
     */
    fun createUserMessage(text: String): ChatDataItem {
        return ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.USER, text)
    }

    /**
     * 创建助手消息
     */
    fun createAssistantMessage(text: String): ChatDataItem {
        return ChatDataItem("2024-01-01 10:01:00", ChatViewHolders.ASSISTANT, text)
    }

    /**
     * 创建带图片的消息
     */
    fun createImageMessage(
        text: String = "Check out this image",
        imageUri: Uri = Uri.parse("file:///test/image.jpg")
    ): ChatDataItem {
        return ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.USER, text).apply {
            this.imageUri = imageUri
        }
    }

    /**
     * 创建带音频的消息
     */
    fun createAudioMessage(
        text: String = "Voice message",
        audioPath: String = "/test/audio.mp3",
        duration: Float = 5.5f
    ): ChatDataItem {
        return ChatDataItem.createAudioInputData("2024-01-01 10:00:00", text, audioPath, duration)
    }

    /**
     * 创建测试用的聊天历史
     */
    fun createTestChatHistory(): List<ChatDataItem> {
        return listOf(
            createUserMessage("Hello"),
            createAssistantMessage("Hi there! How can I help you?"),
            createUserMessage("Can you explain machine learning?"),
            createAssistantMessage("Machine learning is a subset of artificial intelligence...")
        )
    }

    /**
     * 创建测试用的生成结果
     */
    fun createTestGenerateResult(response: String, benchmark: String): HashMap<String, Any> {
        return HashMap<String, Any>().apply {
            put("response", response)
            put("benchmark", benchmark)
            put("tokens", 150)
            put("time", 2.5)
        }
    }

    /**
     * 创建测试用的会话ID
     */
    fun createTestSessionId(): String {
        sessionCounter++
        return "test-session-$sessionCounter"
    }

    /**
     * 创建测试用的模型ID
     */
    fun createTestModelId(prefix: String): String {
        modelCounter++
        return "$prefix/test-model-$modelCounter"
    }

    /**
     * 获取测试用的日期格式
     */
    fun getTestDateFormat(): SimpleDateFormat {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    }

    /**
     * 创建测试用的时间戳
     */
    fun createTestTimestamp(): Long {
        return System.currentTimeMillis()
    }

    /**
     * 验证ChatDataItem的基本属性
     */
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
     * 验证生成结果的基本属性
     */
    fun assertGenerateResultValid(result: HashMap<String, Any>, expectedResponse: String) {
        assertEquals(expectedResponse, result["response"])
        assertNotNull(result["benchmark"])
        assertNotNull(result["tokens"])
        assertNotNull(result["time"])
    }

    /**
     * 创建测试用的URI
     */
    fun createTestUri(scheme: String = "file", path: String = "/test/file.txt"): Uri {
        return Uri.parse("$scheme://$path")
    }

    /**
     * 创建测试用的音频文件路径
     */
    fun createTestAudioPath(): String {
        return "/test/audio_${System.currentTimeMillis()}.mp3"
    }

    /**
     * 创建测试用的图片文件路径
     */
    fun createTestImagePath(): String {
        return "/test/image_${System.currentTimeMillis()}.jpg"
    }

    /**
     * 创建测试用的错误消息
     */
    fun createTestErrorMessage(message: String = "Test error"): Exception {
        return RuntimeException(message)
    }

    /**
     * 验证错误处理
     */
    fun assertErrorHandled(
        exception: Exception,
        expectedMessage: String
    ) {
        assertTrue(exception.message?.contains(expectedMessage) == true)
    }

    /**
     * 创建测试用的进度信息
     */
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
     * 验证进度信息
     */
    fun assertProgressInfoValid(
        progressInfo: HashMap<String, Any>,
        expectedProgress: String
    ) {
        assertTrue(progressInfo.containsKey("progress"))
        assertEquals(expectedProgress, progressInfo["progress"])
    }

    /**
     * 创建测试用的配置路径
     */
    fun createTestConfigPath(): String {
        return "/test/config_${System.currentTimeMillis()}"
    }

    /**
     * 创建测试用的扩散模型路径
     */
    fun createTestDiffusionPath(): String {
        return "/test/diffusion_${System.currentTimeMillis()}"
    }

    /**
     * 验证会话状态
     */
    fun assertSessionStateValid(
        sessionId: String?,
        expectedSessionId: String?
    ) {
        assertEquals(expectedSessionId, sessionId)
    }

    /**
     * 创建测试用的基准信息
     */
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
     * 验证基准信息
     */
    fun assertBenchmarkInfoValid(
        benchmarkInfo: HashMap<String, Any>,
        expectedGenerationTime: Double
    ) {
        assertTrue(benchmarkInfo.containsKey("generation_time"))
        assertEquals(expectedGenerationTime, benchmarkInfo["generation_time"])
    }
} 