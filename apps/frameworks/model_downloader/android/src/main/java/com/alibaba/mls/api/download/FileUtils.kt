package com.alibaba.mls.api.download

import java.io.File

object FileUtils {
    fun getFileSize(file: File?): Long {
        if (file == null || !file.exists()) {
            return 0
        }
        if (file.isFile) {
            return file.length()
        }
        var size: Long = 0
        val files = file.listFiles()
        if (files != null) {
            for (f in files) {
                size += getFileSize(f)
            }
        }
        return size
    }
}
