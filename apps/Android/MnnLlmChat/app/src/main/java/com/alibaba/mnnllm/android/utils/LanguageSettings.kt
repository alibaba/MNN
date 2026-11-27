package com.alibaba.mnnllm.android.utils

import android.content.Context
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat

/**
 * App language selection shared by the in-app setting and Android 13's per-app language screen.
 *
 * The choice is stored by AppCompat in `androidx.appcompat.app.AppCompatDelegate.application_locales_record_file`
 * and applied through [AppCompatDelegate.setApplicationLocales], so both entry points read and write
 * the same state: picking a language here is reflected in Settings > Apps > MNN Chat > Language and
 * the other way round. An empty locale list means "follow the system", which is also AppCompat's
 * default when the user has never chosen.
 */
object LanguageSettings {

    /** Selectable languages, in the order shown in the settings row. */
    enum class Language(val tag: String?, val labelRes: Int) {
        FOLLOW_SYSTEM(null, com.alibaba.mnnllm.android.R.string.language_follow_system),
        ENGLISH("en", com.alibaba.mnnllm.android.R.string.language_english),
        SIMPLIFIED_CHINESE("zh-CN", com.alibaba.mnnllm.android.R.string.language_simplified_chinese),
        TRADITIONAL_CHINESE("zh-TW", com.alibaba.mnnllm.android.R.string.language_traditional_chinese),
        RUSSIAN("ru", com.alibaba.mnnllm.android.R.string.language_russian);

        companion object {
            fun fromTag(tag: String?): Language =
                entries.firstOrNull { it.tag == tag } ?: FOLLOW_SYSTEM

            /** Maps the applied locale list back to an entry; `null` spread means follow system. */
            fun fromLocales(locales: LocaleListCompat): Language {
                val first = if (locales.isEmpty) null else locales.get(0)
                if (first == null) return FOLLOW_SYSTEM
                when (first.language) {
                    "ru" -> return RUSSIAN
                    "zh" -> {
                        // Compare the full tag so zh-HK / zh-MO resolve the way their resources do
                        // instead of being reported as a variant the user never picked.
                        val tag = first.toLanguageTag()
                        return if (tag.startsWith("zh-TW") || tag.startsWith("zh-HK") || tag.startsWith("zh-MO")) {
                            TRADITIONAL_CHINESE
                        } else {
                            SIMPLIFIED_CHINESE
                        }
                    }
                    else -> return ENGLISH
                }
            }
        }
    }

    fun current(): Language = Language.fromLocales(AppCompatDelegate.getApplicationLocales())

    /** Applies [language] app-wide; an empty locale list restores the system language. */
    fun apply(language: Language) {
        val locales = if (language.tag == null) {
            LocaleListCompat.getEmptyLocaleList()
        } else {
            LocaleListCompat.forLanguageTags(language.tag)
        }
        AppCompatDelegate.setApplicationLocales(locales)
    }

    fun label(context: Context, language: Language): String = context.getString(language.labelRes)
}