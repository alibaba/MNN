package com.alibaba.mnnllm.api.openai.ui

import android.content.Intent
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import timber.log.Timber

// This file contains many legacy issues, but doesn't affect runtime

class LogAdapter : RecyclerView.Adapter<LogAdapter.LogViewHolder>() {
    
    private val logEntries = mutableListOf<LogEntryData>()
    
    /**
     * Log entry data
     */
    data class LogEntryData(
        val message: String,
        val clickableInfo: String? = null
    )
    
    class LogViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val textView: TextView = itemView.findViewById(R.id.text_log_item)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LogViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_log_entry, parent, false)
        return LogViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: LogViewHolder, position: Int) {
        val entry = logEntries[position]
        holder.textView.text = entry.message
        
        // Set click event
        if (entry.clickableInfo != null) {
            holder.itemView.setOnClickListener {
                handleLogClick(holder.itemView, entry.clickableInfo)
            }
            // Set clickable style
            holder.itemView.isClickable = true
            holder.itemView.isFocusable = true
        } else {
            holder.itemView.setOnClickListener(null)
            holder.itemView.isClickable = false
            holder.itemView.isFocusable = false
        }
    }
    
    override fun getItemCount(): Int = logEntries.size
    
    /**
     * Handle log click event
     */
    private fun handleLogClick(view: View, clickableInfo: String) {
        try {
            // Try to open file in Android Studio
            val parts = clickableInfo.split(":")
            if (parts.size >= 2) {
                val fileName = parts[0]
                val lineNumber = parts[1].toIntOrNull() ?: 1
                
                // Build Android Studio deep link
                val projectPath = view.context.getString(R.string.project_path)
                val filePath = findFileInProject(fileName)
                
                if (filePath != null) {
                    // Try to open file using Android Studio protocol
                    val intent = Intent(Intent.ACTION_VIEW).apply {
                        data = Uri.parse("androidstudio://open?file=$filePath&line=$lineNumber")
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    
                    try {
                        view.context.startActivity(intent)
                        Timber.tag("LogAdapter").d("Attempting to open $fileName:$lineNumber in Android Studio")
                    } catch (e: Exception) {
                        // If Android Studio protocol fails, try other methods
                        fallbackFileOpen(view, fileName, lineNumber, filePath)
                    }
                } else {
                    // File not found, but don't show toast to avoid disruption
                }
            }
        } catch (e: Exception) {
            Timber.tag("LogAdapter").e(e, "Failed to handle log click")
            // Don't show error toast to avoid disruption
        }
    }
    
    /**
     * Find file in project
     */
    private fun findFileInProject(fileName: String): String? {
        val projectRoot = "c:/Project/github/AIddlx/0528/MNN/apps/Android/MnnLlmChat"
        
        // Recursively search for file
        return searchFileRecursively(java.io.File("$projectRoot/app/src/main"), fileName)
    }
    
    /**
     * Recursively search for file
     */
    private fun searchFileRecursively(directory: java.io.File, fileName: String): String? {
        if (!directory.exists() || !directory.isDirectory) {
            return null
        }
        
        try {
            directory.listFiles()?.forEach { file ->
                when {
                    file.isFile && file.name == fileName -> {
                        return file.absolutePath
                    }
                    file.isDirectory -> {
                        val result = searchFileRecursively(file, fileName)
                        if (result != null) {
                            return result
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Timber.tag("LogAdapter").w(e, "Error searching for file: $fileName")
        }
        
        return null
    }
    
    /**
     * Fallback file opening method
     */
    private fun fallbackFileOpen(view: View, fileName: String, lineNumber: Int, filePath: String) {
        // Show file path information
        val message = view.context.getString(R.string.file_info_template, fileName, lineNumber, filePath)
        Toast.makeText(view.context, message, Toast.LENGTH_LONG).show()
        
        // Log to logcat for manual navigation by developers
        Timber.tag("CodeLocation").i("Click to navigate: $fileName:$lineNumber at $filePath")
    }
    
    fun addLogMessage(message: String) {
        addLogEntry(LogEntryData(message))
    }
    
    fun addLogEntry(entry: LogEntryData) {
        logEntries.add(entry)
        notifyItemInserted(logEntries.size - 1)
        
        // Limit log entries to prevent memory overflow
        if (logEntries.size > 200) {
            logEntries.removeAt(0)
            notifyItemRemoved(0)
        }
    }
    
    fun clearLogs() {
        val size = logEntries.size
        logEntries.clear()
        notifyItemRangeRemoved(0, size)
    }
    
    fun getAllLogs(): List<String> = logEntries.map { it.message }
}