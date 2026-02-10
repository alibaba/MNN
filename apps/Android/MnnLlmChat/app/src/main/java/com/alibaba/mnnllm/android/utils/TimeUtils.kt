// Created by ruoyi.sjd on 2025/7/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils

import java.time.Instant
import java.time.format.DateTimeParseException

object TimeUtils {

    fun convertIsoToTimestamp(isoString: String?): Long? {
        if (isoString.isNullOrEmpty()) {
            return null
        }
        return try {
            val instant = Instant.parse(isoString)
            instant.epochSecond
        } catch (e: DateTimeParseException) {
            null
        }
    }
}