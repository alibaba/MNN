package com.alibaba.mnnllm.api.openai.debug

import android.content.Intent
import android.util.Pair
import android.net.Uri
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.service.OpenAIService
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.PrintWriter
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class OpenApiDumperPlugin : DumperPlugin {
    override fun getName(): String = "openai"

    override fun dump(dumpContext: DumperContext) {
        val writer = PrintWriter(dumpContext.stdout)
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            printUsage(writer)
            return
        }

        val command = args[0]
        try {
            when (command) {
                "status" -> dumpStatus(writer)
                "start" -> startApiService(writer)
                "stop" -> stopApiService(writer)
                "reset-config" -> resetConfig(writer)
                "test-chat" -> testChat(writer, args.drop(1))
                else -> {
                    writer.println("Unknown command: $command")
                    printUsage(writer)
                }
            }
        } catch (e: Exception) {
            writer.println("Error executing command: ${e.message}")
            e.printStackTrace(writer)
        } finally {
            writer.flush()
        }
    }

    private fun dumpStatus(writer: PrintWriter) {
        val service = OpenAIService.getInstance()
        val context = MnnLlmApplication.getAppContext()
        
        if (service == null) {
            writer.println("Status: Service is NOT running (Instance is null)")
        } else {
            writer.println("Status: Service is running")
            writer.println("Is Running (Internal): ${service.isServerRunning()}")
            writer.println("Port: ${service.getServerPort()}")
            writer.println("Model ID: ${service.getCurrentModelId()}")
        }
        
        writer.println("--- Config ---")
        writer.println("Port (Config): ${ApiServerConfig.getPort(context)}")
        writer.println("IP (Config): ${ApiServerConfig.getIpAddress(context)}")
        writer.println("Auth Enabled: ${ApiServerConfig.isAuthEnabled(context)}")
        writer.println("API Key: ${ApiServerConfig.getApiKey(context)}")
    }
    
    private fun resetConfig(writer: PrintWriter) {
        val context = MnnLlmApplication.getAppContext()
        ApiServerConfig.resetToDefault(context)
        writer.println("Configuration reset to default. New API Key generated.")
        dumpStatus(writer)
    }
    
    private fun startApiService(writer: PrintWriter) {
        val context = MnnLlmApplication.getInstance()
        try {
            val intent = Intent(context, OpenAIService::class.java)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
            writer.println("Service start requested via Intent.")
        } catch (e: Exception) {
            writer.println("Failed to start service: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun stopApiService(writer: PrintWriter) {
         val service = OpenAIService.getInstance()
         if (service != null) {
             OpenAIService.releaseService(service, force = true)
             writer.println("Service stop requested.")
         } else {
             writer.println("Service is not running.")
         }
    }

    private fun testChat(writer: PrintWriter, args: List<String>) {
        val prompt = if (args.isEmpty()) "Hello" else args.joinToString(" ")
        writer.println("Testing chat with prompt: $prompt")

        val session = ServiceLocator.getChatSessionProvider().getLlmSession()
        if (session == null) {
            writer.println("Error: No active LLM session. Ensure the chat is open.")
            return
        }

        val latch = CountDownLatch(1)
        val history = listOf(Pair("user", prompt))
        val responseBuilder = StringBuilder()

        writer.println("Submitting request...")
        
        try {
            session.submitFullHistory(history, object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress != null) {
                        responseBuilder.append(progress)
                        return false // Continue
                    } else {
                        latch.countDown()
                        return false // Done
                    }
                }
            })

            val completed = latch.await(60, TimeUnit.SECONDS)
            if (completed) {
                writer.println("Response received:")
                writer.println("--------------------------------------------------")
                writer.println(responseBuilder.toString())
                writer.println("--------------------------------------------------")
            } else {
                writer.println("Error: Timeout waiting for response.")
            }
        } catch (e: Exception) {
            writer.println("Error during chat generation: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun printUsage(writer: PrintWriter) {
        writer.println("Usage: dumpapp openai <command>")
        writer.println("Commands:")
        writer.println("  status            Show current service status and config")
        writer.println("  start             Start the OpenAIService")
        writer.println("  stop              Stop the service")
        writer.println("  reset-config      Reset API settings to default (new API key)")
        writer.println("  test-chat [msg]   Send a test message to the LLM")
    }
}