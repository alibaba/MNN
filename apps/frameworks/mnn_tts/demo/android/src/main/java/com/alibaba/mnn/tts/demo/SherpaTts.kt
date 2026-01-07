package com.alibaba.mnn.tts.demo

import android.util.Log
import com.k2fsa.sherpa.mnn.GeneratedAudio
import com.k2fsa.sherpa.mnn.OfflineTts
import com.k2fsa.sherpa.mnn.OfflineTtsConfig
import com.k2fsa.sherpa.mnn.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.mnn.OfflineTtsModelConfig
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class SherpaTts {

    private var tts: OfflineTts? = null
    @Volatile
    private var hasLoading = false
    private val initComplete = CompletableDeferred<Boolean>()

    companion object {
        private const val TAG = "SherpaTts"
    }

    suspend fun init(modelDir: String?) {
        if (initComplete.isCompleted) {
            return
        }
        if (hasLoading) {
            initComplete.await()
            return // Added return to match logic
        }
        hasLoading = true

        withContext(Dispatchers.IO) {
            try {
                val tts_path = modelDir
                if (tts_path == null) {
                    Log.e(TAG, "Model dir is null")
                    initComplete.complete(false)
                    return@withContext
                }

                Log.d(TAG, "Initializing SherpaTts with path: $tts_path")

                // Check for required files to avoid native crash if possible, though config handles some defaults
                if (!File(tts_path, "model.mnn").exists()) {
                     Log.e(TAG, "model.mnn not found in $tts_path")
                }

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
                Log.d(TAG, "SherpaTts initialized successfully")
                initComplete.complete(true)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize SherpaTts", e)
                initComplete.complete(false)
            }
        }
    }

    fun process(text: String): GeneratedAudio? {
        if (!initComplete.isCompleted) {
            Log.w(TAG, "process called before init complete")
            return null
        }
        try {
            // Check if init was successful
            if (initComplete.getCompleted() == false) return null

            val ttsInstance = tts ?: return null
            val audio = ttsInstance.generate(text=text, sid = 0, speed = 1.0f) // sid=0 default, can be parameter
            return audio
        } catch (e: Exception) {
            Log.e(TAG, "Error processing text", e)
            return null
        }
    }

    fun release() {
        try {
            tts?.release()
            tts = null
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing SherpaTts", e)
        }
    }
}
