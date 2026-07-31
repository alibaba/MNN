package com.alibaba.mnntalk.engine

import android.content.Context
import android.os.SystemClock
import com.alibaba.mnntalk.model.VoiceModelPaths
import com.taobao.meta.avatar.tts.TtsService
import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class LocalVoiceChatEngine(private val context: Context) : Closeable {
    companion object {
        private const val BARGE_IN_GUARD_MILLIS = 450L
        private const val SYSTEM_PROMPT =
            "你是一个运行在手机本地的语音助手。请使用简洁、自然、适合朗读的口语回答。" +
                "不要使用 Markdown、表格或代码块。除非用户要求，否则回答控制在三句话以内。"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val activeTurn = AtomicLong(0L)
    private val prepared = AtomicBoolean(false)
    private val conversationActive = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)

    private val stateMutable = MutableStateFlow<VoiceChatState>(VoiceChatState.Idle)
    val state: StateFlow<VoiceChatState> = stateMutable.asStateFlow()

    private val transcriptsMutable = MutableStateFlow<List<TranscriptLine>>(emptyList())
    val transcripts: StateFlow<List<TranscriptLine>> = transcriptsMutable.asStateFlow()

    private val metricsMutable = MutableStateFlow<GenerationMetrics?>(null)
    val metrics: StateFlow<GenerationMetrics?> = metricsMutable.asStateFlow()

    private var asr: StreamingAsr? = null
    private var llm: LocalLlm? = null
    private var tts: TtsService? = null
    private var audioPlayer: PcmAudioPlayer? = null
    private var generationJob: Job? = null
    private var lastPlaybackStartedAt = 0L

    suspend fun prepare(paths: VoiceModelPaths) {
        check(!closed.get()) { "Voice engine is closed" }
        if (prepared.get()) {
            stateMutable.value = VoiceChatState.Ready
            return
        }

        try {
            stateMutable.value = VoiceChatState.Preparing("加载本地对话模型")
            val loadedLlm = withContext(Dispatchers.IO) {
                LocalLlm.load(
                    configFile = paths.llmConfig,
                    cacheDirectory = File(context.cacheDir, "llm_mmap"),
                    systemPrompt = SYSTEM_PROMPT
                )
            }
            llm = loadedLlm

            stateMutable.value = VoiceChatState.Preparing("加载本地语音识别")
            val loadedAsr = StreamingAsr(context, paths.asrDirectory)
            asr = loadedAsr
            loadedAsr.initialize()

            stateMutable.value = VoiceChatState.Preparing("加载本地语音合成")
            val loadedTts = TtsService().apply { setLanguage("zh") }
            tts = loadedTts
            check(withContext(Dispatchers.IO) { loadedTts.init(paths.ttsDirectory.absolutePath) }) {
                "TTS model failed to load"
            }

            val sampleRate = readTtsSampleRate(paths.ttsDirectory)
            audioPlayer = PcmAudioPlayer(sampleRate)
            prepared.set(true)
            stateMutable.value = VoiceChatState.Ready
        } catch (error: Throwable) {
            releaseEngines()
            stateMutable.value = VoiceChatState.Error(error.message ?: "本地模型加载失败")
            throw error
        }
    }

    fun startConversation() {
        if (!prepared.get()) {
            stateMutable.value = VoiceChatState.Error("本地模型尚未加载")
            return
        }
        if (!conversationActive.compareAndSet(false, true)) return

        try {
            asr?.start(object : StreamingAsr.Listener {
                override fun onPartial(text: String) {
                    handlePartialRecognition(text)
                }

                override fun onFinal(text: String) {
                    handleFinalRecognition(text)
                }

                override fun onError(error: Throwable) {
                    stateMutable.value = VoiceChatState.Error(
                        error.message ?: "本地语音识别失败"
                    )
                }
            })
            stateMutable.value = VoiceChatState.Listening()
        } catch (error: Throwable) {
            conversationActive.set(false)
            stateMutable.value = VoiceChatState.Error(error.message ?: "无法启动麦克风")
        }
    }

    fun stopConversation() {
        if (!conversationActive.compareAndSet(true, false)) return
        interruptActiveTurn()
        asr?.stop()
        stateMutable.value = VoiceChatState.Ready
    }

    fun newConversation() {
        interruptActiveTurn()
        transcriptsMutable.value = emptyList()
        metricsMutable.value = null
        scope.launch(Dispatchers.IO) {
            llm?.reset()
            if (conversationActive.get()) {
                stateMutable.value = VoiceChatState.Listening()
            } else {
                stateMutable.value = VoiceChatState.Ready
            }
        }
    }

    private fun handlePartialRecognition(text: String) {
        if (!conversationActive.get()) return
        val currentState = stateMutable.value
        val canInterrupt =
            currentState is VoiceChatState.Thinking || currentState is VoiceChatState.Speaking
        val outsidePlaybackGuard =
            SystemClock.elapsedRealtime() - lastPlaybackStartedAt > BARGE_IN_GUARD_MILLIS
        if (canInterrupt && outsidePlaybackGuard && text.length >= 2) {
            interruptActiveTurn()
        }
        if (stateMutable.value !is VoiceChatState.Thinking &&
            stateMutable.value !is VoiceChatState.Speaking
        ) {
            stateMutable.value = VoiceChatState.Listening(text)
        }
    }

    private fun handleFinalRecognition(text: String) {
        val normalized = text.trim()
        if (!conversationActive.get() || normalized.isEmpty()) return
        startTurn(normalized)
    }

    private fun startTurn(userText: String) {
        interruptActiveTurn()
        val turnId = activeTurn.incrementAndGet()
        val userLineId = turnId * 2
        val assistantLineId = userLineId + 1
        appendTranscript(TranscriptLine(userLineId, Speaker.USER, userText, true))
        appendTranscript(TranscriptLine(assistantLineId, Speaker.ASSISTANT, "", false))
        stateMutable.value = VoiceChatState.Thinking(userText)

        generationJob = scope.launch(Dispatchers.IO) {
            coroutineScope {
                val speechSegments = Channel<String>(Channel.UNLIMITED)
                val chunker = SentenceChunker()
                val player = checkNotNull(audioPlayer)
                val playbackId = player.begin()
                val speechJob = launch(Dispatchers.IO) {
                    var started = false
                    for (segment in speechSegments) {
                        if (turnId != activeTurn.get()) break
                        if (!started) {
                            started = true
                            lastPlaybackStartedAt = SystemClock.elapsedRealtime()
                            stateMutable.value = VoiceChatState.Speaking(
                                transcriptText(assistantLineId)
                            )
                        }
                        val samples = tts?.process(segment, 0) ?: ShortArray(0)
                        if (turnId != activeTurn.get()) break
                        player.write(playbackId, samples)
                    }
                }

                val turnMetrics = llm?.generate(userText) { token ->
                    if (turnId != activeTurn.get()) {
                        true
                    } else {
                        appendAssistantToken(assistantLineId, token)
                        chunker.append(token).forEach { segment ->
                            speechSegments.trySend(segment)
                        }
                        false
                    }
                }

                if (turnId == activeTurn.get()) {
                    chunker.flush()?.let { speechSegments.send(it) }
                }
                speechSegments.close()
                speechJob.join()

                if (turnId == activeTurn.get()) {
                    markTranscriptFinal(assistantLineId)
                    if (turnMetrics != null) {
                        metricsMutable.value = turnMetrics
                    }
                    player.awaitDrain(playbackId)
                    if (turnId == activeTurn.get() && conversationActive.get()) {
                        stateMutable.value = VoiceChatState.Listening()
                    }
                }
            }
        }
    }

    private fun interruptActiveTurn() {
        activeTurn.incrementAndGet()
        llm?.stop()
        audioPlayer?.interrupt()
        transcriptsMutable.value = transcriptsMutable.value.map {
            if (it.speaker == Speaker.ASSISTANT && !it.isFinal) {
                it.copy(isFinal = true)
            } else {
                it
            }
        }
        if (conversationActive.get()) {
            stateMutable.value = VoiceChatState.Listening()
        }
    }

    private fun appendTranscript(line: TranscriptLine) {
        transcriptsMutable.value = transcriptsMutable.value + line
    }

    private fun appendAssistantToken(id: Long, token: String) {
        transcriptsMutable.value = transcriptsMutable.value.map {
            if (it.id == id) {
                it.copy(text = it.text + token)
            } else {
                it
            }
        }
        if (stateMutable.value is VoiceChatState.Speaking) {
            stateMutable.value = VoiceChatState.Speaking(transcriptText(id))
        }
    }

    private fun markTranscriptFinal(id: Long) {
        transcriptsMutable.value = transcriptsMutable.value.map {
            if (it.id == id) it.copy(isFinal = true) else it
        }
    }

    private fun transcriptText(id: Long): String {
        return transcriptsMutable.value.firstOrNull { it.id == id }?.text.orEmpty()
    }

    private fun readTtsSampleRate(directory: File): Int {
        val fallback = 44_100
        return runCatching {
            val config = JSONObject(File(directory, "config.json").readText())
            config.optInt("sample_rate", fallback).takeIf { it > 0 } ?: fallback
        }.getOrDefault(fallback)
    }

    private fun releaseEngines() {
        audioPlayer?.close()
        audioPlayer = null
        asr?.close()
        asr = null
        llm?.close()
        llm = null
        tts?.destroy()
        tts = null
        prepared.set(false)
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        conversationActive.set(false)
        activeTurn.incrementAndGet()
        llm?.stop()
        asr?.stop()
        audioPlayer?.interrupt()
        scope.cancel()
        releaseEngines()
    }
}
