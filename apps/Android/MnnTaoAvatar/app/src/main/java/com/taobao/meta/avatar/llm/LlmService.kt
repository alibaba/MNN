package com.taobao.meta.avatar.llm

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.ChatService
import com.alibaba.mnnllm.android.ChatSession
import com.taobao.meta.avatar.settings.MainSettings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.cancellable
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class LlmService {

    private var chatSession: ChatSession? = null
    private var stopRequested = false

    suspend fun init(modelDir: String?): Boolean = withContext(Dispatchers.IO) {
        Log.d(TAG, "createSession begin")
        chatSession = ChatService.provide().createSession(
            "test_modelId_sessionId",
            "$modelDir/config.json",
            false, "llm_session", null
        )
        Log.d(TAG, "createSession create success")
        chatSession!!.load()
        Log.d(TAG, "createSession  load success")
        true
    }

    fun startNewSession() {
        chatSession?.reset()
        chatSession?.updatePrompt(MainSettings.getLlmPrompt(ApplicationProvider.get()))
    }

    fun generate(text: String): Flow<Pair<String?, String>> = channelFlow {
        stopRequested = false
        val result = StringBuilder()
        withContext(Dispatchers.Default) {
            chatSession?.generate(text, object : ChatSession.GenerateProgressListener {
                override fun onProgress(progress: String?):Boolean {
                    CoroutineScope(Dispatchers.Default).launch {
                        if (progress != null) {
                            result.append(progress)
                        }
                        send(Pair(progress, result.toString()))
                    }
                    return stopRequested
                }
            })
        }
    }.cancellable()

    fun requestStop() {
        stopRequested = true
    }

    fun unload() {
        chatSession?.release()
    }

    companion object {
        private const val TAG = "LLMService"
    }
}

