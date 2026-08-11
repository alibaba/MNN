package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.model.SessionItem
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.PrintStream

/**
 * Dumper plugin for debugging chat history issues.
 * 
 * Usage:
 *   dumpapp history list              - List all sessions
 *   dumpapp history show <sessionId>  - Show chat data for a session
 *   dumpapp history check <sessionId> - Check session data integrity
 */
class HistoryDumperPlugin : DumperPlugin {

    override fun getName(): String = "history"

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "list" -> handleList(writer)
            "show" -> handleShow(writer, args.drop(1))
            "check" -> handleCheck(writer, args.drop(1))
            "diag" -> handleDiag(writer, args.drop(1))
            else -> doUsage(writer)
        }
    }

    private fun handleList(writer: PrintStream) {
        val context = MnnLlmApplication.getAppContext()
        val chatDataManager = ChatDataManager.getInstance(context)
        val sessions = chatDataManager.allSessions

        writer.println("=== All Sessions (${sessions.size} total) ===")
        if (sessions.isEmpty()) {
            writer.println("  No sessions found")
            return
        }

        sessions.forEachIndexed { index, session ->
            val chatData = chatDataManager.getChatDataBySession(session.sessionId)
            writer.println("[$index] Session:")
            writer.println("    sessionId: ${session.sessionId}")
            writer.println("    modelId: ${session.modelId}")
            writer.println("    title: ${session.title ?: "(no title)"}")
            writer.println("    lastChatTime: ${session.lastChatTime}")
            writer.println("    chatDataCount: ${chatData.size}")
            writer.println()
        }
    }

    private fun handleShow(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp history show <sessionId>")
            return
        }

        val sessionId = args[0]
        val context = MnnLlmApplication.getAppContext()
        val chatDataManager = ChatDataManager.getInstance(context)
        val chatData = chatDataManager.getChatDataBySession(sessionId)

        writer.println("=== Chat Data for Session: $sessionId ===")
        writer.println("Total items: ${chatData.size}")
        writer.println()

        if (chatData.isEmpty()) {
            writer.println("  No chat data found for this session")
            return
        }

        chatData.forEachIndexed { index, item ->
            writer.println("[$index] ChatDataItem:")
            writer.println("    type: ${getTypeName(item.type)}")
            writer.println("    time: ${item.time}")
            writer.println("    text: ${item.text?.take(100) ?: "(null)"}${if ((item.text?.length ?: 0) > 100) "..." else ""}")
            writer.println("    displayText: ${item.displayText?.take(50) ?: "(null)"}${if ((item.displayText?.length ?: 0) > 50) "..." else ""}")
            writer.println("    thinkingText: ${if (item.thinkingText.isNullOrEmpty()) "(none)" else "${item.thinkingText?.take(50)}..."}")
            writer.println("    imageUris: ${item.imageUris?.size ?: 0}")
            writer.println("    audioUri: ${item.audioUri ?: "(none)"}")
            writer.println("    videoUri: ${item.videoUri ?: "(none)"}")
            writer.println()
        }
    }

    private fun handleCheck(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp history check <sessionId>")
            return
        }

        val sessionId = args[0]
        val context = MnnLlmApplication.getAppContext()
        val chatDataManager = ChatDataManager.getInstance(context)
        
        writer.println("=== Integrity Check for Session: $sessionId ===")
        
        // Check if session exists
        val allSessions = chatDataManager.allSessions
        val session = allSessions.find { it.sessionId == sessionId }
        
        if (session == null) {
            writer.println("ERROR: Session not found in session table")
            writer.println("  Available sessions: ${allSessions.map { it.sessionId }}")
            return
        }
        
        writer.println("Session found:")
        writer.println("  modelId: ${session.modelId}")
        writer.println("  title: ${session.title}")
        writer.println("  lastChatTime: ${session.lastChatTime}")
        
        // Check chat data
        val chatData = chatDataManager.getChatDataBySession(sessionId)
        writer.println()
        writer.println("Chat data count: ${chatData.size}")
        
        if (chatData.isEmpty()) {
            writer.println("WARNING: No chat data found for this session")
            writer.println("  This explains why history is not displayed in ChatActivity")
            return
        }
        
        // Analyze chat data
        val userMessages = chatData.filter { it.type == 0 }
        val assistantMessages = chatData.filter { it.type == 1 }
        
        writer.println("  User messages: ${userMessages.size}")
        writer.println("  Assistant messages: ${assistantMessages.size}")
        
        // Check for potential issues
        writer.println()
        writer.println("Potential issues:")
        
        var hasIssues = false
        
        chatData.forEachIndexed { index, item ->
            if (item.text.isNullOrEmpty() && item.displayText.isNullOrEmpty()) {
                writer.println("  [$index] Empty text content")
                hasIssues = true
            }
        }
        
        if (!hasIssues) {
            writer.println("  No obvious issues found")
        }
    }

    private fun handleDiag(writer: PrintStream, args: List<String>) {
        val context = MnnLlmApplication.getAppContext()
        val chatDataManager = ChatDataManager.getInstance(context)
        val sessions = chatDataManager.allSessions

        writer.println("=== History Diagnostic Report ===")
        writer.println()
        writer.println("Database Summary:")
        writer.println("  Total sessions: ${sessions.size}")
        
        var totalChatItems = 0
        var sessionsWithData = 0
        var sessionsWithoutData = 0
        
        sessions.forEach { session ->
            val chatData = chatDataManager.getChatDataBySession(session.sessionId)
            totalChatItems += chatData.size
            if (chatData.isNotEmpty()) {
                sessionsWithData++
            } else {
                sessionsWithoutData++
            }
        }
        
        writer.println("  Total chat items: $totalChatItems")
        writer.println("  Sessions with data: $sessionsWithData")
        writer.println("  Sessions without data: $sessionsWithoutData")
        writer.println()
        
        if (sessionsWithoutData > 0) {
            writer.println("Sessions WITHOUT chat data (potential issue):")
            sessions.forEach { session ->
                val chatData = chatDataManager.getChatDataBySession(session.sessionId)
                if (chatData.isEmpty()) {
                    writer.println("  - ${session.sessionId} (model: ${session.modelId}, title: ${session.title})")
                }
            }
            writer.println()
        }
        
        writer.println("Sessions WITH chat data:")
        sessions.forEach { session ->
            val chatData = chatDataManager.getChatDataBySession(session.sessionId)
            if (chatData.isNotEmpty()) {
                writer.println("  - ${session.sessionId}: ${chatData.size} items (model: ${session.modelId})")
            }
        }
    }

    private fun getTypeName(type: Int): String {
        return when (type) {
            0 -> "HEADER"
            1 -> "ASSISTANT"
            2 -> "USER"
            else -> "UNKNOWN($type)"
        }
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp history <command>")
        writer.println()
        writer.println("Commands:")
        writer.println("  list                - List all sessions with chat data count")
        writer.println("  show <sessionId>    - Show all chat data for a session")
        writer.println("  check <sessionId>   - Check session data integrity")
        writer.println("  diag                - Run diagnostic report on all history")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp history list")
        writer.println("  dumpapp history show 1234567890")
        writer.println("  dumpapp history check 1234567890")
        writer.println("  dumpapp history diag")
    }
}
