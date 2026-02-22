package com.alibaba.mnnllm.android.hotspot

import android.util.Log
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.PriorityBlockingQueue
import java.util.Locale

private const val TAG = "TranslationManager"

/** Snapshot of what the LLM is currently doing, for debug display. */
data class InferenceDebugState(
    val prompt: String = "",
    val partialOutput: String = "",
    val idle: Boolean = true,
)

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

    // ── Debug state ────────────────────────────────────────────────────────────
    private val _debugFlow = MutableStateFlow(InferenceDebugState())
    val debugFlow: StateFlow<InferenceDebugState> = _debugFlow.asStateFlow()

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
        _debugFlow.value = InferenceDebugState() // reset to idle
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
                    TranslationTask.MessageTranslationTask(task.messageId, task.oldLanguage, task.language)
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
        val oldLanguageName = languageNameFor(task.oldLanguage)
        val prompt = buildTranslationPrompt(task, oldLanguageName, languageName)

        val result = StringBuilder()

        // Repetition tracking
        val recentTokens = ArrayDeque<String>(24)
        val linesSeenThisInference = mutableMapOf<String, Int>()
        var currentLine = StringBuilder()
        var shouldStop = false

        // Start a new debug cycle -> overwrite prior debug state
        _debugFlow.value = InferenceDebugState(prompt = prompt, partialOutput = "", idle = false)

        try {
            session.generate(prompt, emptyMap(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress == null) return true

                    result.append(progress)
                    currentLine.append(progress)

                    // ── Token repetition check (last 24 tokens) ──────────────
                    val token = progress
                    recentTokens.addLast(token)
                    if (recentTokens.size > 24) recentTokens.removeFirst()
                    val tokenRepeatCount = recentTokens.count { it == token }
                    if (tokenRepeatCount >= 8) {
                        Log.w(TAG, "Stopping inference: token '$token' repeated $tokenRepeatCount times in last 24 tokens")
                        shouldStop = true
                    }

                    // ── Line repetition check ────────────────────────────────
                    if (progress.contains('\n')) {
                        val parts = progress.split('\n')
                        // Complete the current line with text before the first newline
                        currentLine.append(parts[0])
                        val completedLine = currentLine.toString().trim()
                        if (completedLine.isNotEmpty()) {
                            val lineCount = (linesSeenThisInference[completedLine] ?: 0) + 1
                            linesSeenThisInference[completedLine] = lineCount
                            if (lineCount >= 5) {
                                Log.w(TAG, "Stopping inference: line '$completedLine' repeated $lineCount times")
                                shouldStop = true
                            }
                        }
                        // Start fresh for whatever came after the last newline
                        currentLine = StringBuilder(parts.last())
                    }

                    // Update debug display (still in-progress)
                    _debugFlow.value = InferenceDebugState(
                        prompt = prompt,
                        partialOutput = result.toString(),
                        idle = false,
                    )

                    return shouldStop
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "LLM generate failed for ${task.messageId}", e)
            // Preserve the most-recent prompt and partial output, but mark idle.
            _debugFlow.value = InferenceDebugState(
                prompt = prompt,
                partialOutput = result.toString(),
                idle = true
            )
            return
        }

        // Completed: mark idle but keep the completed partial output visible
        _debugFlow.value = InferenceDebugState(
            prompt = prompt,
            partialOutput = result.toString(),
            idle = true
        )

        var translated = result.toString().trim()

        // Strip <think>...</think> prefix if the output starts with one
        if (translated.startsWith("<think>")) {
            val closeTag = "</think>"
            val closeIdx = translated.indexOf(closeTag)
            if (closeIdx >= 0) {
                translated = translated.substring(closeIdx + closeTag.length).trim()
            }
        }

        if (translated.isNotEmpty()) {
            val previous = if (task.previousTranslation != null) listOf(task.previousTranslation) else emptyList()
            onTranslationReady(task.messageId, task.language, translated, task.contextCount, previous)
        }
    }

    private fun buildTranslationPrompt(
        task: TranslationTask.MessageTranslationTask,
        oldLanguageName: String,
        languageName: String,
    ): String {
        val messageText = ChatServerManager.instance?.getMessageText(task.messageId) ?: ""
        return if (task.contextCount > 0) {
            val contextMessages = ChatServerManager.instance
                ?.getContextMessages(task.messageId, task.contextCount)
                ?.joinToString("\n") { "${it.username}: ${it.text}" }
                ?: ""
            "Translate the last message from $oldLanguageName to $languageName. Reply with ONLY the translation.\n\nConversation context:\n$contextMessages\n\nLast message to translate:\n$messageText"
        } else {
            "Translate the following text from $oldLanguageName to $languageName. Reply with ONLY the translation, nothing else.\n$messageText"
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

        // Repetition tracking
        val recentTokens = ArrayDeque<String>(24)
        val linesSeenThisInference = mutableMapOf<String, Int>()
        var currentLine = StringBuilder()
        var shouldStop = false

        // Start a new debug cycle -> overwrite prior debug state
        _debugFlow.value = InferenceDebugState(prompt = prompt, partialOutput = "", idle = false)

        try {
            session.generate(prompt, emptyMap(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress == null) return true

                    result.append(progress)
                    currentLine.append(progress)

                    // ── Token repetition check ───────────────────────────────
                    val token = progress
                    recentTokens.addLast(token)
                    if (recentTokens.size > 24) recentTokens.removeFirst()
                    val tokenRepeatCount = recentTokens.count { it == token }
                    if (tokenRepeatCount >= 8) {
                        Log.w(TAG, "Stopping UI inference: token '$token' repeated $tokenRepeatCount times")
                        shouldStop = true
                    }

                    // ── Line repetition check ────────────────────────────────
                    if (progress.contains('\n')) {
                        val parts = progress.split('\n')
                        currentLine.append(parts[0])
                        val completedLine = currentLine.toString().trim()
                        if (completedLine.isNotEmpty()) {
                            val lineCount = (linesSeenThisInference[completedLine] ?: 0) + 1
                            linesSeenThisInference[completedLine] = lineCount
                            if (lineCount >= 5) {
                                Log.w(TAG, "Stopping UI inference: line '$completedLine' repeated $lineCount times")
                                shouldStop = true
                            }
                        }
                        currentLine = StringBuilder(parts.last())
                    }

                    _debugFlow.value = InferenceDebugState(
                        prompt = prompt,
                        partialOutput = result.toString(),
                        idle = false,
                    )

                    return shouldStop
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "UI translation failed for ${task.language}", e)
            // Preserve prompt + partial output but mark idle
            _debugFlow.value = InferenceDebugState(
                prompt = prompt,
                partialOutput = result.toString(),
                idle = true
            )
            return
        }

        // Completed: mark idle but keep the completed partial output visible
        _debugFlow.value = InferenceDebugState(
            prompt = prompt,
            partialOutput = result.toString(),
            idle = true
        )

        var raw = result.toString().trim()

        // Strip <think>...</think> prefix if present
        if (raw.startsWith("<think>")) {
            val closeTag = "</think>"
            val closeIdx = raw.indexOf(closeTag)
            if (closeIdx >= 0) {
                raw = raw.substring(closeIdx + closeTag.length).trim()
            }
        }

        // Extract the JSON object from the response //TODO: would possibly be more efficient to translate in smaller chunks, but certainly harder to code.
        val jsonStart = raw.indexOf('{')
        val jsonEnd = raw.lastIndexOf('}')
        if (jsonStart < 0 || jsonEnd < jsonStart) return
        
        try {
            val jsonStr = raw.substring(jsonStart, jsonEnd + 1)
            Log.d(TAG, "UI translation JSON to parse: $jsonStr")

            // Clean up any potentially malformed JSON
            val cleanedJsonStr = jsonStr
                // poorly-quoted keys:  'key': " or `key": ' or "key': ` or anything like that -> "key": "...
                .replace(Regex("""(?<=[{,\s])['"`]([a-z_]+)['"`]\s*:\s*['"`]"""), "\"$1\": \"")
                // unquoted keys:  key:  -> "key":
                .replace(Regex("""(?<=[{,\s])([a-z_]+)\s*:\s*"""), "\"$1\": ")
                // bad ending quote on value: "can't',\n or "you`\n} or "me',\n} or similar - MUST have a \n to be captured, but comma or not
                .replace(Regex("""['"`](,?\s*\n)"""), "\"$1")
                // trailing comma before closing curly bracket (any amount of whitespace in between)
                .replace(Regex(""",\s*\}"""), "}")

            Log.d(TAG, "Cleaned JSON: $cleanedJsonStr")

            val map = com.google.gson.Gson()
                .fromJson<Map<String, String>>(cleanedJsonStr, object : com.google.gson.reflect.TypeToken<Map<String, String>>() {}.type)
            Log.d(TAG, "UI translation parsed successfully: ${map.size} keys")
            onUiTranslationReady(task.requestId, task.language, map)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse UI translations for ${task.language}", e)
            //TODO: Handle by using English until the app restarts, to avoid wasting power retrying repeatedly.
        }
    }

    companion object {
        /** Full English name for a BCP-47 / ISO 639-1 language code. */
        fun languageNameFor(code: String): String {
            val trimmed = code.lowercase().take(2)
            val name = Locale(trimmed).getDisplayLanguage(Locale.ENGLISH)
            // getDisplayLanguage returns the code itself if it can't resolve it
            return name.ifBlank { code }
        }

        /** English UI strings used as the source for translation. */
        val UI_STRINGS_EN = mapOf(
            "chat_title" to "Chat",
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
            "ctx_prev_translation" to "View previous translation",
            "ctx_reply" to "Reply",
            "ctx_cancel_reply" to "Cancel reply",
            "connected_users" to "Connected users",
            "connection_lost" to "Connection lost. Reconnecting…",
            "connection_restored" to "Connected",
            "you" to "You",
            "server_host" to "Host",
            "welcome" to "Welcome to the chat!",
            "no_messages" to "No messages yet. Say hello!",
            "ctx_view_latest" to "View latest translation",
            "ctx_copy_message" to "Copy message",
        )

        val UI_STRINGS_KO = mapOf(
            "chat_title" to "채팅",
            "lang_title" to "언어를 선택하세요",
            "setup_username" to "사용자 이름 선택",
            "setup_username_hint" to "이름을 입력하세요",
            "setup_avatar" to "프로필 사진 (선택)",
            "btn_choose_photo" to "사진 선택",
            "btn_skip" to "건너뛰기",
            "btn_join" to "채팅 참여",
            "chat_placeholder" to "메시지를 입력하세요…",
            "btn_send" to "전송",
            "btn_export" to "채팅 내보내기",
            "translating" to "번역 중…",
            "retranslating" to "재번역 중…",
            "ctx_view_original" to "원문 보기",
            "ctx_view_translation" to "번역 보기",
            "ctx_retranslate" to "더 많은 맥락으로 재번역",
            "ctx_prev_translation" to "이전 번역 보기",
            "ctx_reply" to "답장",
            "ctx_cancel_reply" to "답장 취소",
            "connected_users" to "접속자",
            "connection_lost" to "연결 끊김. 재연결 중…",
            "connection_restored" to "연결됨",
            "you" to "나",
            "server_host" to "호스트",
            "welcome" to "채팅에 오신 것을 환영합니다!",
            "no_messages" to "아직 메시지가 없습니다.",
            "ctx_view_latest" to "최신 번역 보기",
            "ctx_copy_message" to "메시지 복사",
        )

        val UI_STRINGS_JA = mapOf(
            "chat_title" to "チャット",
            "lang_title" to "言語を選択してください",
            "setup_username" to "ユーザー名を入力",
            "setup_username_hint" to "名前を入力してください",
            "setup_avatar" to "プロフィール画像（任意）",
            "btn_choose_photo" to "写真を選ぶ",
            "btn_skip" to "スキップ",
            "btn_join" to "チャットに参加",
            "chat_placeholder" to "メッセージを入力…",
            "btn_send" to "送信",
            "btn_export" to "チャットをエクスポート",
            "translating" to "翻訳中…",
            "retranslating" to "再翻訳中…",
            "ctx_view_original" to "原文を見る",
            "ctx_view_translation" to "翻訳を見る",
            "ctx_retranslate" to "より多くの文脈で再翻訳",
            "ctx_prev_translation" to "前の翻訳を見る",
            "ctx_reply" to "返信",
            "ctx_cancel_reply" to "返信をキャンセル",
            "connected_users" to "接続ユーザー",
            "connection_lost" to "接続が切れました。再接続中…",
            "connection_restored" to "接続済み",
            "you" to "あなた",
            "server_host" to "ホスト",
            "welcome" to "チャットへようこそ！",
            "no_messages" to "まだメッセージはありません。",
            "ctx_view_latest" to "最新の翻訳を表示",
            "ctx_copy_message" to "メッセージをコピー",
        )
    }
}