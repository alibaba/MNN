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

//这个文件包含大量遗留问题，暂不影响运行

class LogAdapter : RecyclerView.Adapter<LogAdapter.LogViewHolder>() {
    
    private val logEntries = mutableListOf<LogEntryData>()
    
    /**
     * 日志条目数据
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
        
        // 设置点击事件
        if (entry.clickableInfo != null) {
            holder.itemView.setOnClickListener {
                handleLogClick(holder.itemView, entry.clickableInfo)
            }
            // 设置可点击样式
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
     * 处理日志点击事件
     */
    private fun handleLogClick(view: View, clickableInfo: String) {
        try {
            // 尝试在Android Studio中打开文件
            val parts = clickableInfo.split(":")
            if (parts.size >= 2) {
                val fileName = parts[0]
                val lineNumber = parts[1].toIntOrNull() ?: 1
                
                // 构建Android Studio的深度链接
                val projectPath = "c:/Project/github/AIddlx/0528/MNN/apps/Android/MnnLlmChat"
                val filePath = findFileInProject(fileName)
                
                if (filePath != null) {
                    // 尝试使用Android Studio协议打开文件
                    val intent = Intent(Intent.ACTION_VIEW).apply {
                        data = Uri.parse("androidstudio://open?file=$filePath&line=$lineNumber")
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    
                    try {
                        view.context.startActivity(intent)
                        Timber.tag("LogAdapter").d("Attempting to open $fileName:$lineNumber in Android Studio")
                    } catch (e: Exception) {
                        // 如果Android Studio协议失败，尝试其他方式
                        fallbackFileOpen(view, fileName, lineNumber, filePath)
                    }
                } else {
                  //  Toast.makeText(view.context, "无法找到文件: $fileName", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Timber.tag("LogAdapter").e(e, "Failed to handle log click")
           // Toast.makeText(view.context, "无法打开文件: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    /**
     * 在项目中查找文件
     */
    private fun findFileInProject(fileName: String): String? {
        val projectRoot = "c:/Project/github/AIddlx/0528/MNN/apps/Android/MnnLlmChat"
        
        // 递归搜索文件
        return searchFileRecursively(java.io.File("$projectRoot/app/src/main"), fileName)
    }
    
    /**
     * 递归搜索文件
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
     * 备用文件打开方式
     */
    private fun fallbackFileOpen(view: View, fileName: String, lineNumber: Int, filePath: String) {
        // 显示文件路径信息
        val message = view.context.getString(R.string.file_info_template, fileName, lineNumber, filePath)
        Toast.makeText(view.context, message, Toast.LENGTH_LONG).show()
        
        // 记录到logcat，方便开发者手动跳转
        Timber.tag("CodeLocation").i("Click to navigate: $fileName:$lineNumber at $filePath")
    }
    
    fun addLogMessage(message: String) {
        addLogEntry(LogEntryData(message))
    }
    
    fun addLogEntry(entry: LogEntryData) {
        logEntries.add(entry)
        notifyItemInserted(logEntries.size - 1)
        
        // 限制日志条数，避免内存溢出
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