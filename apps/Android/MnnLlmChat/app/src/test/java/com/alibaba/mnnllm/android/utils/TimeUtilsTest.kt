package com.alibaba.mnnllm.android.utils

import org.junit.Assert.*
import org.junit.Test

class TimeUtilsTest {

    @Test
    fun `convertIsoToTimestamp with valid ISO string should return correct timestamp`() {
        // Given
        val isoString = "2024-01-01T10:00:00Z"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNotNull(result)
        assertEquals(1704103200L, result)
    }

    @Test
    fun `convertIsoToTimestamp with null input should return null`() {
        // Given
        val isoString: String? = null

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }

    @Test
    fun `convertIsoToTimestamp with empty string should return null`() {
        // Given
        val isoString = ""

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }

    @Test
    fun `convertIsoToTimestamp with blank string should return null`() {
        // Given
        val isoString = "   "

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }

    @Test
    fun `convertIsoToTimestamp with invalid format should return null`() {
        // Given
        val isoString = "not-a-valid-iso-string"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }

    @Test
    fun `convertIsoToTimestamp with partial ISO format should return null`() {
        // Given
        val isoString = "2024-01-01"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }

    @Test
    fun `convertIsoToTimestamp with ISO string with milliseconds should work`() {
        // Given
        val isoString = "2024-01-01T10:00:00.123Z"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNotNull(result)
        assertEquals(1704103200L, result)
    }

    @Test
    fun `convertIsoToTimestamp with ISO string with timezone offset should work`() {
        // Given
        val isoString = "2024-01-01T10:00:00+08:00"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNotNull(result)
        // 10:00 +08:00 = 02:00 UTC
        assertEquals(1704074400L, result)
    }

    @Test
    fun `convertIsoToTimestamp with epoch start time should work`() {
        // Given
        val isoString = "1970-01-01T00:00:00Z"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNotNull(result)
        assertEquals(0L, result)
    }

    @Test
    fun `convertIsoToTimestamp with far future date should work`() {
        // Given
        val isoString = "2099-12-31T23:59:59Z"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNotNull(result)
        assertEquals(4102444799L, result)
    }

    @Test
    fun `convertIsoToTimestamp with malformed date should return null`() {
        // Given
        val isoString = "2024-13-01T10:00:00Z" // Invalid month

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }

    @Test
    fun `convertIsoToTimestamp with special characters should return null`() {
        // Given
        val isoString = "2024-01-01T10:00:00Z\n\r"

        // When
        val result = TimeUtils.convertIsoToTimestamp(isoString)

        // Then
        assertNull(result)
    }
}
