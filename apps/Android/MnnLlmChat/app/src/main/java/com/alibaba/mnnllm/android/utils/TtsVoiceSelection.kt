package com.alibaba.mnnllm.android.utils

import android.content.Context
import androidx.preference.PreferenceManager
import org.json.JSONObject
import java.io.File

/** Supertonic's native loader supports these IDs and both asset directory layouts. */
object TtsVoiceSelection {
    private val voiceIds = listOf("M1", "M2", "F1", "F2")

    fun availableVoices(modelDir: String): List<String> {
        return try {
            val root = File(modelDir)
            if (JSONObject(File(root, "config.json").readText()).optString("model_type") != "supertonic") {
                return emptyList()
            }
            voiceIds.filter { id ->
                val primary = File(root, "mnn_models/voice_styles/$id.json")
                val file = if (primary.isFile && primary.canRead()) primary else File(root, "voice_styles/$id.json")
                file.isFile && file.canRead()
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun savedVoice(context: Context, modelId: String): String? =
        PreferenceManager.getDefaultSharedPreferences(context).getString("tts_voice:$modelId", null)

    fun saveVoice(context: Context, modelId: String, voiceId: String?) {
        val editor = PreferenceManager.getDefaultSharedPreferences(context).edit()
        if (voiceId == null) editor.remove("tts_voice:$modelId")
        else editor.putString("tts_voice:$modelId", voiceId)
        editor.apply()
    }

    fun selectedVoice(context: Context, modelId: String?, modelDir: String): String? {
        if (modelId.isNullOrEmpty()) return null
        return savedVoice(context, modelId)?.takeIf { it in availableVoices(modelDir) }
    }
}
