package com.alibaba.mls.api.download

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import java.util.TimeZone

class TimeUtilsTest {

    @Test
    fun testConvertIsoToTimestamp() {
        // 2024-12-25T10:30:45.123Z
        // We need to compare with expected epoch millis. 
        // 2024-12-25 10:30:45.123 UTC
        
        // Note: Creating a fixed timestamp to verify against might be tricky due to pure calculation, 
        // but we can trust the SimpleDateFormat in the verification if we use the same construction.
        // Or simply checks if it parses correctly.
        
        val isoWithMillis = "2024-12-25T10:30:45.123Z"
        val timestamp1 = TimeUtils.convertIsoToTimestamp(isoWithMillis)
        // 1735122645123
        assertEquals(1735122645123L, timestamp1)

        val isoVal = "2024-12-25T10:30:45Z"
        val timestamp2 = TimeUtils.convertIsoToTimestamp(isoVal)
        // 1735122645000
        assertEquals(1735122645000L, timestamp2)

        assertNull(TimeUtils.convertIsoToTimestamp(null))
        assertNull(TimeUtils.convertIsoToTimestamp("invalid-date"))
    }
}
