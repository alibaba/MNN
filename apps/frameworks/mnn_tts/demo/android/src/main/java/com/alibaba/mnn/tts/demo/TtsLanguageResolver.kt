package com.alibaba.mnn.tts.demo

object TtsLanguageResolver {
    fun resolve(text: String, selectedLanguage: String): String {
        if (text.any { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.HAN }) {
            return "zh"
        }
        return selectedLanguage.ifBlank { "zh" }
    }
}
