package com.alibaba.mnnllm.android.hotspot

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.LlmSession
import com.google.gson.Gson
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.request.receiveText
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.server.sse.SSE
import io.ktor.server.sse.sse
import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "ChatServerManager"
const val CHAT_SERVER_PORT = 8765

/**
 * Manages the Ktor-based interpreted chat server.
 *
 * Lifecycle:
 *   1. Call [start] with a model config path.
 *   2. The server listens on [CHAT_SERVER_PORT] from all interfaces (0.0.0.0).
 *   3. Clients connect via the hotspot IP printed in [HotspotConnectionInfo.urlQrContent].
 *   4. Call [stop] to shut everything down.
 */
class ChatServerManager private constructor(private val context: Context) {

    // ── State ──────────────────────────────────────────────────────────────────
    private val gson = Gson()
    private val users = ConcurrentHashMap<String, ChatUser>()          // userId → user
    private val messages = mutableListOf<ChatMessage>()                // in arrival order
    private val translations = ConcurrentHashMap<String, MessageTranslation>() // "msgId:lang" → translation
    private val uiTranslations = ConcurrentHashMap<String, Map<String, String>>() // lang → key→value
    private val connectedIds = ConcurrentHashMap.newKeySet<String>()   // currently connected SSE users
    private val _connectedCountFlow = MutableStateFlow(0)
    val connectedCountFlow: StateFlow<Int> = _connectedCountFlow.asStateFlow()

    // ── Broadcast ──────────────────────────────────────────────────────────────
    // DROP_OLDEST ensures a single slow/stuck subscriber (e.g. an in-app WebView with
    // a stalled connection) can never block broadcasts to all other healthy SSE clients.
    private val eventFlow = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 256,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    // ── Coroutine scope ────────────────────────────────────────────────────────
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // ── Ktor server ────────────────────────────────────────────────────────────
    private var server: io.ktor.server.engine.EmbeddedServer<*, *>? = null
    private var chatService: ChatService? = null
    private var llmSession: LlmSession? = null
    private var translationManager: TranslationManager? = null

    // ── HTML asset ────────────────────────────────────────────────────────────
    private val chatHtml: String by lazy {
        try {
            context.assets.open("chat_app.html").bufferedReader().readText()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load chat_app.html from assets", e)
            "<h1>Chat app not found</h1>"
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    fun start(modelId: String, configPath: String) {
        if (server != null) return
        initLlmSession(modelId, configPath)
        server = embeddedServer(Netty, host = "0.0.0.0", port = CHAT_SERVER_PORT) {
            configureKtor()
        }
        server!!.start(wait = false)
        Log.i(TAG, "Chat server started on port $CHAT_SERVER_PORT")
        instance = this
    }

    fun stop() {
        translationManager?.stop()
        translationManager = null
        server?.stop(gracePeriodMillis = 500, timeoutMillis = 2000)
        server = null
        llmSession?.release()
        llmSession = null
        chatService = null
        scope.cancel()
        users.clear()
        synchronized(messages) { messages.clear() }
        translations.clear()
        uiTranslations.clear()
        connectedIds.clear()
        _connectedCountFlow.value = 0
        instance = null
        Log.i(TAG, "Chat server stopped")
    }

    fun isRunning() = server != null

    /** Returns the plain text of a message (used by TranslationManager). */
    fun getMessageText(messageId: String): String? =
        synchronized(messages) { messages.find { it.id == messageId }?.text }

    /** Returns the N messages before [messageId] (for context-aware retranslation). */
    fun getContextMessages(messageId: String, contextCount: Int): List<ChatMessage> {
        synchronized(messages) {
            val idx = messages.indexOfFirst { it.id == messageId }
            if (idx <= 0) return emptyList()
            val from = maxOf(0, idx - contextCount)
            return messages.subList(from, idx)
        }
    }

    fun getConnectedUserCount(): Int = connectedIds.size

    // ── Private helpers ────────────────────────────────────────────────────────

    private fun initLlmSession(modelId: String, configPath: String) {
        chatService = ChatService()
        llmSession = chatService!!.createLlmSession(modelId, configPath, null, null, false)
        scope.launch(Dispatchers.IO) {
            try {
                llmSession!!.load()
                Log.i(TAG, "LLM session loaded for translation")
                translationManager = TranslationManager(
                    llmSession = llmSession!!,
                    onTranslationReady = { msgId, lang, text, retranslationCount, previousVersions ->
                        val key = "$msgId:$lang"
                        val existing = translations[key]
                        val allPrevious = (existing?.previousVersions ?: emptyList()) + previousVersions
                        val translation = MessageTranslation(
                            messageId = msgId,
                            language = lang,
                            text = text,
                            retranslationCount = retranslationCount,
                            previousVersions = allPrevious.distinct(),
                            isPending = false,
                        )
                        translations[key] = translation
                        broadcast("translation", translation)
                    },
                    onUiTranslationReady = { requestId, lang, map ->
                        uiTranslations[lang] = map
                        broadcast("ui_translations", mapOf("requestId" to requestId, "language" to lang, "translations" to map))
                    },
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load LLM session", e)
            }
        }
    }

    private fun broadcast(type: String, data: Any) {
        val json = gson.toJson(mapOf("type" to type, "data" to data))
        scope.launch { eventFlow.emit(json) }
    }

    private fun markPendingTranslation(messageId: String, language: String) {
        val key = "$messageId:$language"
        translations[key] = MessageTranslation(
            messageId = messageId,
            language = language,
            text = "",
            isPending = true,
        )
        broadcast("translation_pending", mapOf("messageId" to messageId, "language" to language))
    }

    private fun queueTranslationForAllLanguages(message: ChatMessage) {
        val sourceLanguage = users[message.userId]?.language ?: "en"
        val targetLanguages = users.values.map { it.language }.toSet() - sourceLanguage
        for (lang in targetLanguages) {
            markPendingTranslation(message.id, lang)
            translationManager?.enqueue(
                TranslationTask.MessageTranslationTask(
                    messageId = message.id,
                    language = lang,
                    sequenceNumber = message.timestamp,
                )
            )
        }
    }

    // ── Ktor configuration ─────────────────────────────────────────────────────

    private fun Application.configureKtor() {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true; isLenient = true })
        }
        install(CORS) {
            allowMethod(HttpMethod.Get)
            allowMethod(HttpMethod.Post)
            allowHeader(HttpHeaders.ContentType)
            anyHost()
        }
        install(SSE)
        routing { routes() }
    }

    private fun io.ktor.server.routing.Routing.routes() {

        // ── Serve HTML app ──────────────────────────────────────────────────
        get("/") {
            call.respondText(chatHtml, ContentType.Text.Html)
        }

        // ── SSE endpoint ────────────────────────────────────────────────────
        sse("/api/events") {
            val userId = call.request.queryParameters["userId"] ?: run {
                send(ServerSentEvent(data = gson.toJson(mapOf("type" to "error", "message" to "userId required"))))
                return@sse
            }
            connectedIds.add(userId)
            _connectedCountFlow.value = connectedIds.size
            try {
                // Send current state to this new client
                val initPayload = gson.toJson(mapOf(
                    "type" to "init",
                    "userId" to userId,
                    "users" to users.values.toList(),
                    "messages" to synchronized(messages) { messages.toList() },
                    "translations" to translations.values.toList(),
                ))
                send(ServerSentEvent(data = initPayload))

                // Broadcast subsequent events; swallow individual send errors so
                // one broken connection cannot cancel another client's SSE stream.
                eventFlow.collect { json ->
                    try {
                        send(ServerSentEvent(data = json))
                    } catch (e: Exception) {
                        Log.w(TAG, "SSE send failed for $userId, dropping event", e)
                    }
                }
            } finally {
                connectedIds.remove(userId)
                _connectedCountFlow.value = connectedIds.size
                // Only broadcast user_left if this client actually completed /api/join
                if (users.containsKey(userId)) {
                    broadcast("user_left", mapOf("userId" to userId))
                }
            }
        }

        // ── Join / update user ──────────────────────────────────────────────
        post("/api/join") {
            try {
                val body = gson.fromJson(call.receiveText(), Map::class.java)
                val userId = body["userId"] as? String ?: return@post call.respond(HttpStatusCode.BadRequest)
                val username = (body["username"] as? String)?.take(32) ?: return@post call.respond(HttpStatusCode.BadRequest)
                val language = (body["language"] as? String)?.take(10) ?: "en"
                val avatar = body["avatar"] as? String   // base64 data-URI, optional

                val isNew = !users.containsKey(userId)
                val user = ChatUser(
                    id = userId,
                    username = username,
                    language = language,
                    avatarBase64 = avatar,
                )
                users[userId] = user

                if (isNew) {
                    broadcast("user_joined", user)
                    // Queue history translations for this user's language
                    val needsTranslation = synchronized(messages) { messages.toList() }
                    for ((idx, msg) in needsTranslation.withIndex()) {
                        val sourceLanguage = users[msg.userId]?.language ?: "en"
                        if (sourceLanguage != language && !translations.containsKey("${msg.id}:$language")) {
                            markPendingTranslation(msg.id, language)
                            translationManager?.enqueue(
                                TranslationTask.HistoryTranslationTask(
                                    messageId = msg.id,
                                    language = language,
                                    reverseOrder = idx.toLong(),
                                )
                            )
                        }
                    }
                } else {
                    broadcast("user_updated", user)
                }

                // Queue UI translation if language is not hard-coded and not cached
                if (!setOf("en", "ko", "ja").contains(language) && !uiTranslations.containsKey(language)) {
                    val requestId = java.util.UUID.randomUUID().toString()
                    translationManager?.enqueue(TranslationTask.UiTranslationTask(language, requestId))
                }

                call.respond(HttpStatusCode.OK, mapOf("ok" to true))
            } catch (e: Exception) {
                Log.e(TAG, "Join error", e)
                call.respond(HttpStatusCode.InternalServerError)
            }
        }

        // ── Send message ────────────────────────────────────────────────────
        post("/api/message") {
            try {
                val body = gson.fromJson(call.receiveText(), Map::class.java)
                val userId = body["userId"] as? String ?: return@post call.respond(HttpStatusCode.BadRequest)
                val text = (body["text"] as? String)?.trim() ?: return@post call.respond(HttpStatusCode.BadRequest)
                val replyToId = body["replyToId"] as? String
                if (text.isEmpty()) return@post call.respond(HttpStatusCode.BadRequest)
                val user = users[userId] ?: return@post call.respond(HttpStatusCode.Unauthorized)

                val replyToMsg = replyToId?.let { id -> synchronized(messages) { messages.find { it.id == id } } }
                val msg = ChatMessage(
                    userId = userId,
                    username = user.username,
                    text = text,
                    replyToId = replyToId,
                    replyToText = replyToMsg?.text?.take(100),
                    replyToUsername = replyToMsg?.username,
                )
                synchronized(messages) { messages.add(msg) }
                broadcast("message", msg)
                queueTranslationForAllLanguages(msg)

                call.respond(HttpStatusCode.OK, mapOf("id" to msg.id))
            } catch (e: Exception) {
                Log.e(TAG, "Message error", e)
                call.respond(HttpStatusCode.InternalServerError)
            }
        }

        // ── Get history ─────────────────────────────────────────────────────
        get("/api/history") {
            val msgs = synchronized(messages) { messages.toList() }
            call.respondText(gson.toJson(mapOf("messages" to msgs, "translations" to translations.values.toList())), ContentType.Application.Json)
        }

        // ── Get connected users ─────────────────────────────────────────────
        get("/api/users") {
            call.respondText(gson.toJson(users.values.toList()), ContentType.Application.Json)
        }

        // ── Get UI translations ─────────────────────────────────────────────
        get("/api/ui-translations") {
            val lang = call.request.queryParameters["lang"] ?: "en"
            val cached = uiTranslations[lang]
            if (cached != null) {
                call.respondText(gson.toJson(mapOf("language" to lang, "translations" to cached)), ContentType.Application.Json)
            } else {
                call.respond(HttpStatusCode.NotFound, mapOf("pending" to true))
            }
        }

        // ── Request retranslation ────────────────────────────────────────────
        post("/api/retranslate") {
            try {
                val body = gson.fromJson(call.receiveText(), Map::class.java)
                val messageId = body["messageId"] as? String ?: return@post call.respond(HttpStatusCode.BadRequest)
                val language = body["language"] as? String ?: return@post call.respond(HttpStatusCode.BadRequest)
                val contextCount = (body["contextCount"] as? Double)?.toInt() ?: 1

                val existing = translations["$messageId:$language"]
                val previousTranslation = existing?.text?.takeIf { it.isNotEmpty() }

                markPendingTranslation(messageId, language)
                translationManager?.enqueue(
                    TranslationTask.MessageTranslationTask(
                        messageId = messageId,
                        language = language,
                        contextCount = contextCount,
                        previousTranslation = previousTranslation,
                    )
                )
                call.respond(HttpStatusCode.OK, mapOf("ok" to true))
            } catch (e: Exception) {
                Log.e(TAG, "Retranslate error", e)
                call.respond(HttpStatusCode.InternalServerError)
            }
        }

        // ── Serve avatar data-URI ────────────────────────────────────────────
        get("/api/avatar/{userId}") {
            val userId = call.parameters["userId"] ?: return@get call.respond(HttpStatusCode.BadRequest)
            val avatarData = users[userId]?.avatarBase64
            if (!avatarData.isNullOrEmpty()) {
                call.respondText(avatarData, ContentType.Text.Plain)
            } else {
                call.respond(HttpStatusCode.NotFound)
            }
        }
    }

    companion object {
        @Volatile
        var instance: ChatServerManager? = null
            private set

        fun create(context: Context): ChatServerManager {
            val mgr = ChatServerManager(context.applicationContext)
            instance = mgr
            return mgr
        }
    }
}
