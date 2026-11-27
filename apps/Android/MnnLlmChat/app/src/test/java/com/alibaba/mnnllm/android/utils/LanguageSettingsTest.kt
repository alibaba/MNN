package com.alibaba.mnnllm.android.utils

import androidx.core.os.LocaleListCompat
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The language row and Android 13's per-app language screen share one AppCompat state, so the
 * mapping between a stored locale and the row's selection has to be exact in both directions.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LanguageSettingsTest {

    private fun locales(vararg tags: String) = LocaleListCompat.forLanguageTags(tags.joinToString(","))

    @Test
    fun `empty locale list means follow system`() {
        assertEquals(
            LanguageSettings.Language.FOLLOW_SYSTEM,
            LanguageSettings.Language.fromLocales(LocaleListCompat.getEmptyLocaleList())
        )
    }

    @Test
    fun `english locale maps to english`() {
        assertEquals(
            LanguageSettings.Language.ENGLISH,
            LanguageSettings.Language.fromLocales(locales("en"))
        )
    }

    @Test
    fun `english regional locale maps to english`() {
        assertEquals(
            LanguageSettings.Language.ENGLISH,
            LanguageSettings.Language.fromLocales(locales("en-US"))
        )
    }

    @Test
    fun `zh-CN maps to simplified chinese`() {
        assertEquals(
            LanguageSettings.Language.SIMPLIFIED_CHINESE,
            LanguageSettings.Language.fromLocales(locales("zh-CN"))
        )
    }

    @Test
    fun `zh-TW maps to traditional chinese`() {
        assertEquals(
            LanguageSettings.Language.TRADITIONAL_CHINESE,
            LanguageSettings.Language.fromLocales(locales("zh-TW"))
        )
    }

    @Test
    fun `zh-HK maps to traditional chinese`() {
        assertEquals(
            LanguageSettings.Language.TRADITIONAL_CHINESE,
            LanguageSettings.Language.fromLocales(locales("zh-HK"))
        )
    }

    @Test
    fun `zh-MO maps to traditional chinese`() {
        assertEquals(
            LanguageSettings.Language.TRADITIONAL_CHINESE,
            LanguageSettings.Language.fromLocales(locales("zh-MO"))
        )
    }

    @Test
    fun `zh without region maps to simplified chinese as the fallback`() {
        assertEquals(
            LanguageSettings.Language.SIMPLIFIED_CHINESE,
            LanguageSettings.Language.fromLocales(locales("zh"))
        )
    }

    @Test
    fun `zh-SG maps to simplified chinese`() {
        assertEquals(
            LanguageSettings.Language.SIMPLIFIED_CHINESE,
            LanguageSettings.Language.fromLocales(locales("zh-SG"))
        )
    }

    @Test
    fun `ru maps to russian`() {
        assertEquals(
            LanguageSettings.Language.RUSSIAN,
            LanguageSettings.Language.fromLocales(locales("ru"))
        )
    }

    @Test
    fun `ru-RU maps to russian`() {
        assertEquals(
            LanguageSettings.Language.RUSSIAN,
            LanguageSettings.Language.fromLocales(locales("ru-RU"))
        )
    }

    @Test
    fun `fromTag round trips every selectable language`() {
        LanguageSettings.Language.entries.forEach { language ->
            assertEquals(language, LanguageSettings.Language.fromTag(language.tag))
        }
    }

    @Test
    fun `unknown tag falls back to follow system`() {
        assertEquals(LanguageSettings.Language.FOLLOW_SYSTEM, LanguageSettings.Language.fromTag("fr"))
        assertEquals(LanguageSettings.Language.FOLLOW_SYSTEM, LanguageSettings.Language.fromTag(null))
    }

    // Applying a language goes through AppCompatDelegate, which has no delegate to write to in a
    // plain JVM test, so the setApplicationLocales round trip is verified on device instead.
}