// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.file.FileVisitResult
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.SimpleFileVisitor
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.BasicFileAttributes

object DownloadFileUtils {
    const val TAG: String = "FileUtils"

    @JvmStatic
    fun repoFolderName(repoId: String?, repoType: String?): String {
        if (repoId == null || repoType == null) {
            return ""
        }
        val repoParts = repoId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        val parts: MutableList<String> = ArrayList()
        parts.add(repoType + "s") // e.g., "models"
        for (part in repoParts) {
            if (part != null && !part.isEmpty()) {
                parts.add(part)
            }
        }
        // Join parts with "--" separator
        return java.lang.String.join("--", parts)
    }

    @JvmStatic
    fun deleteDirectoryRecursively(dir: File?): Boolean {
        if (dir == null || !dir.exists()) {
            return false
        }

        val dirPath = dir.toPath()
        try {
            Files.walkFileTree(dirPath, object : SimpleFileVisitor<Path>() {
                @Throws(IOException::class)
                override fun visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult {
                    Files.delete(file)
                    return FileVisitResult.CONTINUE
                }

                @Throws(IOException::class)
                override fun postVisitDirectory(
                    directory: Path,
                    exc: IOException?
                ): FileVisitResult {
                    Files.delete(directory)
                    return FileVisitResult.CONTINUE
                }

                @Throws(IOException::class)
                override fun visitFileFailed(file: Path, exc: IOException): FileVisitResult {
                    return FileVisitResult.TERMINATE
                }
            })
            return true
        } catch (e: IOException) {
            return false
        }
    }

    fun getPointerPath(storageFolder: File?, commitHash: String, relativePath: String): File {
        val commitFolder = File(storageFolder, "snapshots/$commitHash")
        return File(commitFolder, relativePath)
    }

    @JvmStatic
    fun getPointerPathParent(storageFolder: File?, sha: String): File {
        return File(storageFolder, "snapshots/$sha")
    }

    @JvmStatic
    fun getLastFileName(path: String?): String? {
        if (path == null || path.isEmpty()) {
            return path
        }
        val pos = path.lastIndexOf('/')
        return if ((pos == -1)) path else path.substring(pos + 1)
    }

    @JvmStatic
    fun createSymlink(target: String?, linkPath: String?) {
        val targetPath = Paths.get(target)
        val link = Paths.get(linkPath)
        try {
            Files.createSymbolicLink(link, targetPath)
        } catch (e: IOException) {
            Log.e(TAG, "createSymlink error", e)
        }
    }

    @JvmStatic
    @Throws(IOException::class)
    fun createSymlink(target: Path?, linkPath: Path?) {
        Files.createSymbolicLink(linkPath, target)
    }

    @JvmStatic
    fun moveWithPermissions(src: File, dest: File) {
        try {
            Files.move(src.toPath(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING)
            dest.setReadable(true, true)
            dest.setWritable(true, true)
            dest.setExecutable(false, false)
        } catch (e: IOException) {
            Log.e(TAG, "moveWithPermissions Failed", e)
        }
    }
}
