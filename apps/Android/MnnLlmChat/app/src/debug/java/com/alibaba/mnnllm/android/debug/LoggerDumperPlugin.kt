package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.utils.TimberConfig
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.PrintStream

class LoggerDumperPlugin : DumperPlugin {

    override fun getName(): String {
        return "logs"
    }

    override fun dump(dumpContext: DumperContext) {
        val writer = dumpContext.stdout
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        val command = args[0]

        when (command) {
            "get" -> doGet(writer)
            "set" -> {
                if (args.size < 2) {
                    doUsage(writer)
                } else {
                    doSet(writer, args[1])
                }
            }
            "file" -> {
                if (args.size < 2) {
                    doUsage(writer)
                } else {
                    doFile(writer, args[1])
                }
            }
            else -> doUsage(writer)
        }
    }

    private fun doGet(writer: PrintStream) {
        val level = TimberConfig.currentLogLevel
        val levelStr = when (level) {
            android.util.Log.VERBOSE -> "VERBOSE"
            android.util.Log.DEBUG -> "DEBUG"
            android.util.Log.INFO -> "INFO"
            android.util.Log.WARN -> "WARN"
            android.util.Log.ERROR -> "ERROR"
            android.util.Log.ASSERT -> "ASSERT"
            else -> "UNKNOWN ($level)"
        }
        writer.println("Current Log Level: $levelStr")
    }

    private fun doSet(writer: PrintStream, levelStr: String) {
        val level = when (levelStr.uppercase()) {
            "VERBOSE", "V" -> android.util.Log.VERBOSE
            "DEBUG", "D" -> android.util.Log.DEBUG
            "INFO", "I" -> android.util.Log.INFO
            "WARN", "W" -> android.util.Log.WARN
            "ERROR", "E" -> android.util.Log.ERROR
            "ASSERT", "A" -> android.util.Log.ASSERT
            else -> {
                writer.println("Unknown log level: $levelStr")
                return
            }
        }
        TimberConfig.setLogLevel(level)
        writer.println("Log Level set to $levelStr")
    }
    
    private fun doFile(writer: PrintStream, action: String) {
        when (action.lowercase()) {
            "enable" -> {
                TimberConfig.setFileLoggingEnabled(true)
                writer.println("File logging ENABLED")
            }
            "disable" -> {
                TimberConfig.setFileLoggingEnabled(false)
                writer.println("File logging DISABLED")
            }
            "status" -> {
                val status = if (TimberConfig.isFileLoggingEnabled()) "ENABLED" else "DISABLED"
                writer.println("File logging level: $status")
            }
            else -> {
                writer.println("Unknown file action: $action")
            }
        }
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp logs <command>")
        writer.println("Commands:")
        writer.println("  get             - Get current log level")
        writer.println("  set <LEVEL>     - Set log level (VERBOSE, DEBUG, INFO, WARN, ERROR, ASSERT)")
        writer.println("  file <ACTION>   - Control file logging")
        writer.println("    ACTION: enable, disable, status")
    }
}
