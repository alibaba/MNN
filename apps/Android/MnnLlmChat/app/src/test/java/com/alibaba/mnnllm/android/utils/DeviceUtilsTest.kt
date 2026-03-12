package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.content.res.Configuration
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.Locale

/**
 * TDD tests for DeviceUtils.isChinese locale detection.
 * 
 * Current behavior: only zh_CN returns true
 * Expected behavior: all Chinese locales (zh_CN, zh_TW, zh_HK, zh_SG, zh) should return true
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class DeviceUtilsTest {

    private fun createContextWithLocale(locale: Locale): Context {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        return context.createConfigurationContext(config)
    }

    // ==================== 当前行为测试（应该通过） ====================

    @Test
    fun `isChinese should return true for zh_CN locale`() {
        val context = createContextWithLocale(Locale.SIMPLIFIED_CHINESE) // zh_CN
        assertTrue("zh_CN should be Chinese", DeviceUtils.isChinese(context))
    }

    @Test
    fun `isChinese should return false for en_US locale`() {
        val context = createContextWithLocale(Locale.US)
        assertFalse("en_US should not be Chinese", DeviceUtils.isChinese(context))
    }

    // ==================== 预期行为测试（当前会失败，修复后应通过） ====================

    @Test
    fun `isChinese should return true for zh_TW locale`() {
        val context = createContextWithLocale(Locale.TRADITIONAL_CHINESE) // zh_TW
        assertTrue(
            "zh_TW (Traditional Chinese - Taiwan) should be recognized as Chinese",
            DeviceUtils.isChinese(context)
        )
    }

    @Test
    fun `isChinese should return true for zh_HK locale`() {
        val context = createContextWithLocale(Locale("zh", "HK"))
        assertTrue(
            "zh_HK (Chinese - Hong Kong) should be recognized as Chinese",
            DeviceUtils.isChinese(context)
        )
    }

    @Test
    fun `isChinese should return true for zh_SG locale`() {
        val context = createContextWithLocale(Locale("zh", "SG"))
        assertTrue(
            "zh_SG (Chinese - Singapore) should be recognized as Chinese",
            DeviceUtils.isChinese(context)
        )
    }

    @Test
    fun `isChinese should return true for zh locale without country`() {
        val context = createContextWithLocale(Locale.CHINESE) // zh (no country)
        assertTrue(
            "zh (Chinese without country) should be recognized as Chinese",
            DeviceUtils.isChinese(context)
        )
    }

    @Test
    fun `isChinese should return false for ja_JP locale`() {
        val context = createContextWithLocale(Locale.JAPAN)
        assertFalse("ja_JP should not be Chinese", DeviceUtils.isChinese(context))
    }

    @Test
    fun `isChinese should return false for ko_KR locale`() {
        val context = createContextWithLocale(Locale.KOREA)
        assertFalse("ko_KR should not be Chinese", DeviceUtils.isChinese(context))
    }
}
