package com.alibaba.mnnllm.android.agent

import android.content.Context
import android.content.Intent
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.FileProvider
import com.alibaba.mnnllm.android.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.File
import java.util.Locale

object AgentWorkspaceFileBrowser {
    fun show(context: Context) {
        val workspace = AgenticPythonEngine.workspaceDir(context)
        showDirectory(context, workspace, workspace)
    }

    private fun showDirectory(context: Context, workspace: File, directory: File) {
        val children = listDirectoryChildren(directory)
        if (children.isEmpty()) {
            MaterialAlertDialogBuilder(context)
                .setTitle(directoryTitle(workspace, directory))
                .setMessage(R.string.workspace_files_empty)
                .setPositiveButton(android.R.string.ok, null)
                .show()
            return
        }

        val list = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 8, 24, 8)
        }
        if (directory.canonicalFile != workspace.canonicalFile) {
            list.addView(createParentRow(context, workspace, directory.parentFile ?: workspace))
        }
        children.forEach { file ->
            list.addView(createFileRow(context, workspace, file))
        }

        val scroll = ScrollView(context).apply {
            addView(
                list,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
            )
        }

        MaterialAlertDialogBuilder(context)
            .setTitle(directoryTitle(workspace, directory))
            .setView(scroll)
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun createFileRow(context: Context, workspace: File, file: File): TextView {
        val relativePath = runCatching {
            file.relativeTo(workspace).path.replace("\\", "/")
        }.getOrElse { file.name }
        return TextView(context).apply {
            text = buildString {
                append(if (file.isDirectory) "[Folder] " else "[File] ")
                append(relativePath)
                append("\n")
                append(if (file.isDirectory) "${file.listFiles()?.size ?: 0} items" else formatFileSize(file.length()))
            }
            textSize = 14f
            setPadding(0, 14, 0, 14)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setOnClickListener {
                if (file.isDirectory) {
                    showDirectory(context, workspace, file)
                } else {
                    showFile(context, workspace, file)
                }
            }
        }
    }

    private fun createParentRow(context: Context, workspace: File, parent: File): TextView {
        return TextView(context).apply {
            text = "[Folder] .."
            textSize = 14f
            setPadding(0, 14, 0, 14)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setOnClickListener { showDirectory(context, workspace, parent) }
        }
    }

    private fun listDirectoryChildren(directory: File): List<File> {
        if (!directory.exists() || !directory.isDirectory) return emptyList()
        return directory.listFiles().orEmpty()
            .sortedWith(compareByDescending<File> { it.isDirectory }.thenBy { it.name.lowercase(Locale.US) })
    }

    private fun directoryTitle(workspace: File, directory: File): String {
        val relativePath = runCatching {
            directory.relativeTo(workspace).path.replace("\\", "/")
        }.getOrElse { "" }
        return if (relativePath.isBlank() || relativePath == ".") {
            "Workspace files"
        } else {
            relativePath
        }
    }

    private fun showFile(context: Context, workspace: File, file: File) {
        val relativePath = runCatching {
            file.relativeTo(workspace).path.replace("\\", "/")
        }.getOrElse { file.name }
        val message = if (canPreviewText(file.name)) {
            readTextPreview(file)
        } else {
            "No built-in preview for this file type.\n\nPath: $relativePath\nSize: ${formatFileSize(file.length())}"
        }
        val text = TextView(context).apply {
            this.text = message
            textSize = 13f
            setPadding(24, 8, 24, 8)
        }
        val scroll = ScrollView(context).apply {
            addView(
                text,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
            )
        }
        MaterialAlertDialogBuilder(context)
            .setTitle(relativePath)
            .setView(scroll)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton("Open with app") { _, _ -> openFileWithExternalApp(context, file) }
            .show()
    }

    private fun openFileWithExternalApp(context: Context, file: File) {
        try {
            val uri = FileProvider.getUriForFile(
                context,
                context.packageName + ".fileprovider",
                file
            )
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, guessMimeType(file.name))
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            context.startActivity(Intent.createChooser(intent, file.name))
        } catch (e: Exception) {
            Toast.makeText(context, R.string.workspace_file_open_failed, Toast.LENGTH_SHORT).show()
        }
    }

    private fun canPreviewText(name: String): Boolean {
        val lower = name.lowercase(Locale.US)
        return lower.endsWith(".txt") ||
            lower.endsWith(".md") ||
            lower.endsWith(".csv") ||
            lower.endsWith(".json") ||
            lower.endsWith(".py") ||
            lower.endsWith(".log") ||
            lower.endsWith(".xml") ||
            lower.endsWith(".html") ||
            lower.endsWith(".htm")
    }

    private fun readTextPreview(file: File): String {
        return try {
            val limit = 128 * 1024
            val buffer = ByteArray(limit + 1)
            val count = file.inputStream().use { input ->
                input.read(buffer)
            }
            if (count <= 0) {
                return ""
            }
            val truncated = count > limit
            val text = buffer.copyOfRange(0, minOf(count, limit)).toString(Charsets.UTF_8)
            if (truncated) {
                "$text\n\n... truncated ..."
            } else {
                text
            }
        } catch (_: Exception) {
            "Failed to read this file."
        }
    }

    private fun guessMimeType(name: String): String {
        val lower = name.lowercase(Locale.US)
        return when {
            lower.endsWith(".xlsx") -> "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            lower.endsWith(".xls") -> "application/vnd.ms-excel"
            lower.endsWith(".csv") -> "text/csv"
            lower.endsWith(".pdf") -> "application/pdf"
            lower.endsWith(".txt") -> "text/plain"
            lower.endsWith(".json") -> "application/json"
            lower.endsWith(".png") -> "image/png"
            lower.endsWith(".jpg") || lower.endsWith(".jpeg") -> "image/jpeg"
            else -> "*/*"
        }
    }

    private fun formatFileSize(bytes: Long): String {
        if (bytes < 1024) return "$bytes B"
        val kb = bytes / 1024.0
        if (kb < 1024) return String.format(Locale.US, "%.1f KB", kb)
        return String.format(Locale.US, "%.1f MB", kb / 1024.0)
    }
}
