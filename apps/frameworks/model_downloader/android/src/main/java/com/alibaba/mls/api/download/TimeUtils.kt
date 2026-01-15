package com.alibaba.mls.api.download

import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone

object TimeUtils {
    fun convertIsoToTimestamp(isoString: String?): Long? {
        if (isoString == null) return null
        return try {
            val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
            format.timeZone = TimeZone.getTimeZone("UTC")
            format.parse(isoString)?.time
        } catch (e: Exception) {
            try {
                // Fallback for formats without milliseconds
                val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
                format.timeZone = TimeZone.getTimeZone("UTC")
                format.parse(isoString)?.time
            } catch (e2: Exception) {
                null
            }
        }
    }
}
