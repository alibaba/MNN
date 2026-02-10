// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download.hf

import android.util.Log
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.security.DigestInputStream
import java.security.MessageDigest
import java.security.NoSuchAlgorithmException
import java.util.Locale
import kotlin.io.path.exists

object HfShaVerifier {
    private const val TAG = "HfShaVerifier"
    fun verify(etag: String, file: Path): Boolean {
        Log.d(TAG, "Verifying $file exists: ${file.exists()}")
        val expected = etag.trim('"').lowercase(Locale.ROOT)
        val actual = when (expected.length) {
            40 -> gitSha1Hex(file)
            64 -> digestHex(file, "SHA-256")
            else -> throw IllegalArgumentException("Unexpected ETag length: ${expected.length}")
        }
        Log.d(TAG, "Verifying $file: expected=$expected actual=$actual")
        return expected == actual
    }

    @Throws(IOException::class, NoSuchAlgorithmException::class)
    private fun gitSha1Hex(file: Path): String {
        val md = MessageDigest.getInstance("SHA-1")
        val size = Files.size(file)
        md.update("blob $size\u0000".toByteArray(StandardCharsets.UTF_8))
        Files.newInputStream(file).use { input ->
            val buf = ByteArray(8192)
            var n: Int
            while (input.read(buf).also { n = it } > 0) {
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    @Throws(IOException::class, NoSuchAlgorithmException::class)
    private fun digestHex(file: Path, algo: String): String {
        val md = MessageDigest.getInstance(algo)
        Files.newInputStream(file).use { `in` ->
            DigestInputStream(`in`, md).use { dis ->
                val buf = ByteArray(8192)
                while (dis.read(buf) != -1) { }
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}