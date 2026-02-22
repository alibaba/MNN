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
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

private const val TAG = "ChatServerManager"
const val CHAT_SERVER_PORT = 8765

/**
 * The fixed userId used for every client connecting from the loopback address
 * (127.0.0.1 / localhost).  Because the Android WebView and the default browser
 * have completely separate localStorage, we cannot rely on a stored userId to
 * identify the host across different browser contexts.  Instead, the client
 * always uses this well-known constant when it detects it is on 127.0.0.1, and
 * the server returns this value from [GET /api/host-user-id] so the client never
 * needs it hard-coded.
 */
const val HOST_USER_ID = "host"

/**
 * Manages the Ktor-based interpreted chat server.
 *
 * Lifecycle:
 *   1. Call [start] with a model config path.
 *   2. The server listens on [CHAT_SERVER_PORT] from all interfaces (0.0.0.0).
 *   3. Clients connect via the hotspot IP printed in [HotspotConnectionInfo.urlQrContent].
 *   4. Call [stop] to shut everything down.
 *
 * Multi-tab / multi-browser support for the host device:
 *   Any client reaching the server via 127.0.0.1 is the device owner.  Rather
 *   than relying on localStorage (which is siloed per browser), the client asks
 *   GET /api/host-user-id to learn the fixed userId it should use, and the server
 *   always treats that userId as a single logical user regardless of how many
 *   tabs or browsers are open.  We track a per-userId SSE connection reference
 *   count so that closing one tab does NOT broadcast user_left until ALL
 *   connections for that userId drop.
 */
class ChatServerManager private constructor(private val context: Context) {

    // ── State ──────────────────────────────────────────────────────────────────
    private val gson = Gson()
    private val users = ConcurrentHashMap<String, ChatUser>()          // userId → user
    private val messages = mutableListOf<ChatMessage>()                // in arrival order
    private val translations = ConcurrentHashMap<String, MessageTranslation>() // "msgId:lang" → translation
    private val uiTranslations = ConcurrentHashMap<String, Map<String, String>>() // lang → key→value

    /**
     * Tracks how many active SSE connections exist per userId.
     * A user is considered "connected" as long as this count is > 0.
     * We use AtomicInteger so increment/decrement are thread-safe without locking the map.
     */
    private val connectedTabCounts = ConcurrentHashMap<String, AtomicInteger>()

    private val _connectedCountFlow = MutableStateFlow(0)
    val connectedCountFlow: StateFlow<Int> = _connectedCountFlow.asStateFlow()

    // ── Exposed inference debug flow (single flow instance the UI can reliably collect) ──
    private val _inferenceDebugFlow = MutableStateFlow(InferenceDebugState())
    val inferenceDebugFlow: StateFlow<InferenceDebugState> = _inferenceDebugFlow.asStateFlow()

    // debug collector job that mirrors TranslationManager.debugFlow -> _inferenceDebugFlow
    private var debugCollectorJob: kotlinx.coroutines.Job? = null

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
            var html = context.assets.open("chat_app.html").bufferedReader().readText()
            // Inject the Kotlin-side hard-coded UI strings so the client doesn't have to duplicate them.
            // The HTML contains a marker <!--__INJECT_BUILTIN_UI__--> which we replace with an object assignment.
            try {
                val enJson = gson.toJson(TranslationManager.UI_STRINGS_EN)
                val koJson = gson.toJson(TranslationManager.UI_STRINGS_KO)
                val jaJson = gson.toJson(TranslationManager.UI_STRINGS_JA)
                val injection = "<script>window.__BUILTIN_UI_STRINGS__ = { en: $enJson, ko: $koJson, ja: $jaJson };</script>"
                html = html.replace("<!--__INJECT_BUILTIN_UI__-->", injection)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to inject builtin UI strings into HTML", e)
            }
            html
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load chat_app.html from assets", e)
            "<h1>Chat app not found</h1>"
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    fun start(modelId: String, configPath: String) {
        if (server != null) return
        loadUiTranslationsCache()
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

        // stop mirroring and reset debug state
        debugCollectorJob?.cancel()
        debugCollectorJob = null
        _inferenceDebugFlow.value = InferenceDebugState()

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
        connectedTabCounts.clear()
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

    /** Number of distinct users with at least one active SSE connection. */
    fun getConnectedUserCount(): Int = connectedTabCounts.count { it.value.get() > 0 }

    // ── Private helpers ────────────────────────────────────────────────────────

    /**
     * Increments the tab count for [userId] and returns the new count.
     * Also refreshes [_connectedCountFlow] to reflect the distinct-user count.
     */
    private fun onTabConnected(userId: String): Int {
        val count = connectedTabCounts.getOrPut(userId) { AtomicInteger(0) }.incrementAndGet()
        _connectedCountFlow.value = getConnectedUserCount()
        Log.d(TAG, "Tab connected for $userId → $count active tabs")
        return count
    }

    /**
     * Decrements the tab count for [userId] and returns the new count.
     * Removes the entry entirely when it reaches zero.
     * Also refreshes [_connectedCountFlow].
     */
    private fun onTabDisconnected(userId: String): Int {
        val counter = connectedTabCounts[userId] ?: return 0
        val count = counter.decrementAndGet()
        if (count <= 0) {
            connectedTabCounts.remove(userId)
        }
        _connectedCountFlow.value = getConnectedUserCount()
        Log.d(TAG, "Tab disconnected for $userId → $count active tabs remaining")
        return maxOf(count, 0)
    }

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
                        saveUiTranslationsCache()   // persist immediately
                        broadcast("ui_translations", mapOf("requestId" to requestId, "language" to lang, "translations" to map))
                    },
                )

                // Mirror the TranslationManager's debug flow into the manager's single _inferenceDebugFlow
                debugCollectorJob?.cancel()
                debugCollectorJob = scope.launch {
                    translationManager!!.debugFlow.collect { state ->
                        _inferenceDebugFlow.value = state
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load LLM session", e)
            }
        }
    }

    private val uiTranslationsCacheFile: File by lazy {
        File(context.filesDir, "ui_translations_cache.json")
    }

    private fun loadUiTranslationsCache() {
        try {
            if (uiTranslationsCacheFile.exists()) {
                val type = object : com.google.gson.reflect.TypeToken<Map<String, Map<String, String>>>() {}.type
                val cached: Map<String, Map<String, String>> = gson.fromJson(uiTranslationsCacheFile.readText(), type)
                uiTranslations.putAll(cached)
                Log.i(TAG, "Loaded UI translation cache: ${cached.keys}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load UI translations cache", e)
        }
    }

    private fun saveUiTranslationsCache() {
        try {
            uiTranslationsCacheFile.writeText(gson.toJson(uiTranslations))
        } catch (e: Exception) {
            Log.w(TAG, "Failed to save UI translations cache", e)
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
                    oldLanguage = sourceLanguage,
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

        // ── Host identity ────────────────────────────────────────────────────
        // Returns the fixed host userId and the current host ChatUser (if they
        // have already joined), so that any client on 127.0.0.1 — regardless of
        // which browser or WebView it is running in — can adopt the same identity.
        // Response shape: { "hostUserId": "host", "user": <ChatUser|null> }
        get("/api/host-user-id") {
            val user = users[HOST_USER_ID]
            call.respondText(
                gson.toJson(mapOf("hostUserId" to HOST_USER_ID, "user" to user)),
                ContentType.Application.Json,
            )
        }

        // ── Session check ────────────────────────────────────────────────────
        // Used by remote clients opening a second tab in the same browser.
        // localStorage is shared across same-browser tabs, so myUserId will
        // already match a known user if they joined in an earlier tab.
        // Returns the stored ChatUser (200) or 404 if userId is unknown.
        get("/api/session/{userId}") {
            val userId = call.parameters["userId"]
            val user = userId?.let { users[it] }
            if (user != null) {
                call.respondText(gson.toJson(user), ContentType.Application.Json)
            } else {
                call.respond(HttpStatusCode.NotFound)
            }
        }

        // ── SSE endpoint ────────────────────────────────────────────────────
        sse("/api/events") {
            val userId = call.request.queryParameters["userId"] ?: run {
                send(ServerSentEvent(data = gson.toJson(mapOf("type" to "error", "message" to "userId required"))))
                return@sse
            }

            // Increment tab count; first tab for this user marks them as "online".
            val tabCount = onTabConnected(userId)
            Log.d(TAG, "SSE opened for $userId (tab #$tabCount)")

            try {
                // Send current state to this new tab.
                val initPayload = gson.toJson(mapOf(
                    "type" to "init",
                    "userId" to userId,
                    "users" to users.values.toList(),
                    "messages" to synchronized(messages) { messages.toList() },
                    "translations" to translations.values.toList(),
                    "uiTranslations" to uiTranslations,
                ))
                send(ServerSentEvent(data = initPayload))

                // Broadcast subsequent events; swallow individual send errors so
                // one broken connection cannot cancel another client's SSE stream.
                // Don't catch InterruptedException, which indicates that user isn't connected anymore.
                eventFlow.collect { json ->
                    send(ServerSentEvent(data = json))
                }
            } finally {
                // Only broadcast user_left when the LAST tab for this user closes.
                val remaining = onTabDisconnected(userId)
                if (remaining == 0 && users.containsKey(userId)) {
                    broadcast("user_left", mapOf("userId" to userId))
                    Log.d(TAG, "All tabs closed for $userId → broadcasting user_left")
                }
            }
        }

        // ── Get UI translation early for joining user who already picked their language ──────────────────────────────────────────────
        post("/api/request-ui-translation") {
            try {
                val body = gson.fromJson(call.receiveText(), Map::class.java)
                val language = body["language"] as? String ?: return@post call.respond(HttpStatusCode.BadRequest)
                if (!setOf("en", "ko", "ja").contains(language)) {
                    val cached = uiTranslations[language]
                    if (cached != null) {
                        // Already cached — will be sent via SSE init when user connects
                        call.respondText(
                            gson.toJson(mapOf("status" to "cached", "translations" to cached)),
                            ContentType.Application.Json,
                        )
                    } else {
                        val requestId = java.util.UUID.randomUUID().toString()
                        translationManager?.enqueue(TranslationTask.UiTranslationTask(language, requestId))
                        call.respond(HttpStatusCode.OK, mapOf("status" to "queued"))
                    }
                } else {
                    call.respond(HttpStatusCode.OK, mapOf("status" to "builtin"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "request-ui-translation error", e)
                call.respond(HttpStatusCode.InternalServerError)
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
                                    oldLanguage = sourceLanguage,
                                    language = language,
                                    reverseOrder = idx.toLong(),
                                )
                            )
                        }
                    }
                } else {
                    broadcast("user_updated", user)
                }

                // Belt-and-suspenders: if UI translation is already cached for this language,
                // broadcast it directly to the joining user so they don't miss it due to
                // SSE/join timing races.
                if (!setOf("en", "ko", "ja").contains(language)) {
                    val cached = uiTranslations[language]
                    if (cached != null) {
                        broadcast("ui_translations", mapOf(
                            "requestId" to "cached",
                            "language" to language,
                            "translations" to cached,
                        ))
                    } else {
                        val requestId = java.util.UUID.randomUUID().toString()
                        translationManager?.enqueue(TranslationTask.UiTranslationTask(language, requestId))
                    }
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

                // Determine original/source language for the message to set oldLanguage
                val msgObj = synchronized(messages) { messages.find { it.id == messageId } }
                val sourceLanguage = msgObj?.let { users[it.userId]?.language } ?: "en"

                markPendingTranslation(messageId, language)
                translationManager?.enqueue(
                    TranslationTask.MessageTranslationTask(
                        messageId = messageId,
                        oldLanguage = sourceLanguage,
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

        // ── Serve avatar data-URI ───────────────────────────────────────────
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

        private val _idleDebugFlow = MutableStateFlow(InferenceDebugState())

        fun create(context: Context): ChatServerManager {
            val mgr = ChatServerManager(context.applicationContext)
            instance = mgr
            return mgr
        }
    }
}