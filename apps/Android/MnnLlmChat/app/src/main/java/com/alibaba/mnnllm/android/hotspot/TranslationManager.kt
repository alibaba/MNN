package com.alibaba.mnnllm.android.hotspot

import android.util.Log
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.concurrent.PriorityBlockingQueue

private const val TAG = "TranslationManager"

/**
 * Manages a priority queue of translation tasks processed sequentially by one LLM session.
 *
 * Priority (lowest number = highest priority):
 *   0 → UI text translation for a new language
 *   1 → newly sent messages (FIFO)
 *   2 → retranslation with more context
 *   3 → historical messages for a newcomer (newest→oldest)
 */
class TranslationManager(
    private val llmSession: LlmSession,
    private val onTranslationReady: suspend (messageId: String, language: String, text: String, retranslationCount: Int, previousVersions: List<String>) -> Unit,
    private val onUiTranslationReady: suspend (requestId: String, language: String, translations: Map<String, String>) -> Unit,
) {
    private val queue = PriorityBlockingQueue<TranslationTask>()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val enqueuedKeys = mutableSetOf<String>()

    init {
        scope.launch { processQueue() }
    }

    fun enqueue(task: TranslationTask) {
        synchronized(enqueuedKeys) {
            if (enqueuedKeys.add(task.key)) {
                queue.offer(task)
            }
        }
    }

    fun stop() {
        scope.cancel()
        queue.clear()
        synchronized(enqueuedKeys) { enqueuedKeys.clear() }
    }

    private suspend fun processQueue() {
        while (true) {
            val task = try {
                // Blocking take - waits until a task is available
                kotlinx.coroutines.withContext(Dispatchers.IO) { queue.take() }
            } catch (e: InterruptedException) {
                break
            }
            synchronized(enqueuedKeys) { enqueuedKeys.remove(task.key) }
            try {
                processTask(task)
            } catch (e: Exception) {
                Log.e(TAG, "Translation task failed: ${task.key}", e)
            }
        }
    }

    private suspend fun processTask(task: TranslationTask) {
        when (task) {
            is TranslationTask.MessageTranslationTask -> translateMessage(task)
            is TranslationTask.UiTranslationTask -> translateUi(task)
            is TranslationTask.HistoryTranslationTask -> {
                // Reuse MessageTranslationTask logic
                translateMessage(
                    TranslationTask.MessageTranslationTask(task.messageId, task.language)
                )
            }
        }
    }

    /**
     * Translate a single chat message. If contextCount > 0, the previous
     * [contextCount] messages in the conversation history are included in the
     * prompt to aid contextual accuracy.
     */
    private suspend fun translateMessage(task: TranslationTask.MessageTranslationTask) {
        val session = llmSession
        session.setKeepHistory(false)

        val languageName = languageNameFor(task.language)
        val prompt = buildTranslationPrompt(task, languageName)

        val result = StringBuilder()
        try {
            session.generate(prompt, emptyMap(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress != null) result.append(progress)
                    return false  // false = continue generating; true = stop
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "LLM generate failed for ${task.messageId}", e)
            return
        }

        val translated = result.toString().trim()
        if (translated.isNotEmpty()) {
            val previous = if (task.previousTranslation != null) listOf(task.previousTranslation) else emptyList()
            onTranslationReady(task.messageId, task.language, translated, task.contextCount, previous)
        }
    }

    private fun buildTranslationPrompt(
        task: TranslationTask.MessageTranslationTask,
        languageName: String,
    ): String {
        val messageText = ChatServerManager.instance?.getMessageText(task.messageId) ?: ""
        return if (task.contextCount > 0) {
            val contextMessages = ChatServerManager.instance
                ?.getContextMessages(task.messageId, task.contextCount)
                ?.joinToString("\n") { "${it.username}: ${it.text}" }
                ?: ""
            "Translate the last message to $languageName. Reply with ONLY the translation.\n\nConversation context:\n$contextMessages\n\nLast message to translate:\n$messageText"
        } else {
            "Translate the following text to $languageName. Reply with ONLY the translation, nothing else.\n$messageText"
        }
    }

    /** Translate a batch of UI strings to the given language. */
    private suspend fun translateUi(task: TranslationTask.UiTranslationTask) {
        val session = llmSession
        session.setKeepHistory(false)

        val languageName = languageNameFor(task.language)
        val uiStrings = UI_STRINGS_EN.entries.joinToString("\n") { (k, v) -> "$k: $v" }
        val prompt = """Translate each UI string to $languageName for a chat application. 
Output ONLY a JSON object with the same keys and translated values. No other text.
Input:
$uiStrings"""

        val result = StringBuilder()
        try {
            session.generate(prompt, emptyMap(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress != null) result.append(progress)
                    return false  // false = continue generating; true = stop
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "UI translation failed for ${task.language}", e)
            return
        }

        val raw = result.toString().trim()
        // Extract the JSON object from the response
        val jsonStart = raw.indexOf('{')
        val jsonEnd = raw.lastIndexOf('}')
        if (jsonStart < 0 || jsonEnd < jsonStart) return

        try {
            val jsonStr = raw.substring(jsonStart, jsonEnd + 1)
            val map = com.google.gson.Gson()
                .fromJson<Map<String, String>>(jsonStr, object : com.google.gson.reflect.TypeToken<Map<String, String>>() {}.type)
            onUiTranslationReady(task.requestId, task.language, map)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse UI translations for ${task.language}", e)
        }
    }

    companion object {
        /** Full English name for a BCP-47 / ISO 639-1 language code. */
        fun languageNameFor(code: String): String = when (code.lowercase().take(2)) {
            "en" -> "English"
            "ko" -> "Korean"
            "ja" -> "Japanese"
            "zh" -> "Chinese"
            "es" -> "Spanish"
            "fr" -> "French"
            "de" -> "German"
            "it" -> "Italian"
            "pt" -> "Portuguese"
            "ru" -> "Russian"
            "ar" -> "Arabic"
            "hi" -> "Hindi"
            "th" -> "Thai"
            "vi" -> "Vietnamese"
            "id" -> "Indonesian"
            "nl" -> "Dutch"
            "pl" -> "Polish"
            "tr" -> "Turkish"
            "sv" -> "Swedish"
            "da" -> "Danish"
            "fi" -> "Finnish"
            "no" -> "Norwegian"
            "cs" -> "Czech"
            "sk" -> "Slovak"
            "hu" -> "Hungarian"
            "ro" -> "Romanian"
            "bg" -> "Bulgarian"
            "hr" -> "Croatian"
            "uk" -> "Ukrainian"
            "he" -> "Hebrew"
            "fa" -> "Persian"
            "ms" -> "Malay"
            "tl" -> "Filipino"
            "bn" -> "Bengali"
            else -> code
        }

        /** English UI strings used as the source for translation. */
        val UI_STRINGS_EN = mapOf(
            "lang_title" to "Select your language",
            "setup_username" to "Choose a username",
            "setup_username_hint" to "Enter your name",
            "setup_avatar" to "Profile picture (optional)",
            "btn_choose_photo" to "Choose photo",
            "btn_skip" to "Skip",
            "btn_join" to "Join chat",
            "chat_placeholder" to "Type a message…",
            "btn_send" to "Send",
            "btn_export" to "Export chat",
            "translating" to "Translating…",
            "retranslating" to "Retranslating…",
            "ctx_view_original" to "View original",
            "ctx_view_translation" to "View translation",
            "ctx_retranslate" to "Retranslate with more context",
            "ctx_view_previous" to "View previous translation",
            "ctx_reply" to "Reply",
            "ctx_cancel_reply" to "Cancel reply",
            "connected_users" to "Connected users",
            "connection_lost" to "Connection lost. Reconnecting…",
            "connection_restored" to "Connected",
            "you" to "You",
            "server_host" to "Host",
            "welcome" to "Welcome to the chat!",
            "no_messages" to "No messages yet. Say hello!",
        )
    }
}
