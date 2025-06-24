// Created by ruoyi.sjd on 2025/4/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.tts

import com.k2fsa.sherpa.mnn.GeneratedAudio
import com.k2fsa.sherpa.mnn.OfflineTts
import com.k2fsa.sherpa.mnn.OfflineTtsConfig
import com.k2fsa.sherpa.mnn.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.mnn.OfflineTtsModelConfig
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SherpaTts {

    private lateinit var tts: OfflineTts
    @Volatile
    private var hasLoading = false
    private val initComplete = CompletableDeferred<Boolean>()

    suspend fun init(modelDir: String?) {
        if (initComplete.isCompleted) {
            return
        }
        if (hasLoading) {
            initComplete.await()
        }
        withContext(Dispatchers.IO) {
            val tts_path = "/data/local/tmp/kokoro-multi-lang-v1_0"
            val config = OfflineTtsConfig(
                model= OfflineTtsModelConfig(
                    kokoro= OfflineTtsKokoroModelConfig(
                        model="${tts_path}/model.mnn",
                        voices="${tts_path}/voices.bin",
                        tokens="${tts_path}/tokens.txt",
                        dataDir="${tts_path}/espeak-ng-data",
                        dictDir="${tts_path}/dict",
                        lexicon="${tts_path}/lexicon-us-en.txt,${tts_path}/lexicon-zh.txt",
                    ),
                    numThreads=2,
                    debug=true,
                ),
            )
            tts = OfflineTts(config=config)
        }
        initComplete.complete(true)
    }

    fun process(text: String): GeneratedAudio? {
        if (!initComplete.isCompleted) {
            return null
        }
        val audio = tts.generate(text=text, sid = 47, speed = 1.0f)
        return audio
    }

    fun release() {
        tts.release()
    }
}