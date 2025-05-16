// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download.hf

import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.hf.HfFileMetadata
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException

object HfFileMetadataUtils {
    private const val HUGGINGFACE_HEADER_X_REPO_COMMIT = "x-repo-commit"
    private const val HUGGINGFACE_HEADER_X_LINKED_ETAG = "x-linked-etag"
    private const val HUGGINGFACE_HEADER_X_LINKED_SIZE = "x-linked-size"
    private fun parseContentLength(contentLength: String?): Long {
        if (!contentLength.isNullOrEmpty()) {
            return try {
                contentLength.toLong()
            } catch (e: NumberFormatException) {
                0
            }
        }
        return 0
    }

    private fun normalizeETag(etag: String?): String? {
        if (etag != null && etag.startsWith("\"") && etag.endsWith("\"")) {
            return etag.substring(1, etag.length - 1)
        }
        return etag
    }

    @JvmStatic
    @Throws(FileDownloadException::class)
    fun getFileMetadata(client: OkHttpClient, url: String): HfFileMetadata {
        val request: Request = Request.Builder()
            .url(url)
            .head()
            .header("Accept-Encoding", "identity")
            .build()

        val metadata = HfFileMetadata()

        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful && response.code != 302) {
                    throw FileDownloadException("Failed to fetch metadata status " + response.code)
                }
                metadata.location = url
                if (response.code == 302) {
                    val location = response.header("Location")
                    if (location != null) {
                        metadata.location = location
                    }
                }

                val linkedEtag = response.header(HUGGINGFACE_HEADER_X_LINKED_ETAG)
                val etag = response.header("ETag")
                metadata.etag =
                    normalizeETag(if (!linkedEtag.isNullOrEmpty()) linkedEtag else etag)

                val linkedSize = response.header(HUGGINGFACE_HEADER_X_LINKED_SIZE)
                val contentLength = response.header("Content-Length")
                metadata.size =
                    parseContentLength(if (!linkedSize.isNullOrEmpty()) linkedSize else contentLength)
                metadata.commitHash = response.header(HUGGINGFACE_HEADER_X_REPO_COMMIT)
            }
        } catch (e: IOException) {
            throw FileDownloadException("GetFileMetadata IOException: " + e.message)
        }

        return metadata
    }
}
