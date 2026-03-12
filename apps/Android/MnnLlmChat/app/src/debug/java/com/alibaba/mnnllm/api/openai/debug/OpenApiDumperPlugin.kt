package com.alibaba.mnnllm.api.openai.debug

import android.content.Intent
import android.util.Pair
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.service.OpenAIService
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.File
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
                "diag" -> dumpDiagnostics(writer)
                "start" -> startApiService(writer, args.drop(1))
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
            writer.println("Start Requests: ${service.getStartRequestCount()}")
            writer.println("Bootstrap Count: ${service.getBootstrapCount()}")
        }
        
        writer.println("--- Config ---")
        writer.println("Port (Config): ${ApiServerConfig.getPort(context)}")
        writer.println("IP (Config): ${ApiServerConfig.getIpAddress(context)}")
        writer.println("HTTPS URL (Config): ${ApiServerConfig.useHttpsUrl(context)}")
        writer.println("TLS Runtime Supported: false")
        writer.println("Auth Enabled: ${ApiServerConfig.isAuthEnabled(context)}")
        writer.println("API Key: ${ApiServerConfig.getApiKey(context)}")
    }
    
    private fun resetConfig(writer: PrintWriter) {
        val context = MnnLlmApplication.getAppContext()
        ApiServerConfig.resetToDefault(context)
        writer.println("Configuration reset to default. New API Key generated.")
        dumpStatus(writer)
    }

    private fun dumpDiagnostics(writer: PrintWriter) {
        val context = MnnLlmApplication.getAppContext()
        val service = OpenAIService.getInstance()
        val port = ApiServerConfig.getPort(context)
        val configuredHost = ApiServerConfig.getIpAddress(context)
        val listeners = readListeningSockets(port)
        val hasNonLoopbackListener = listeners.any { !it.isLoopback() }

        writer.println("OpenAI Diagnostics")
        writer.println("Config Host: $configuredHost")
        writer.println("Config Port: $port")
        writer.println("Config HTTPS URL: ${ApiServerConfig.useHttpsUrl(context)}")
        writer.println("TLS Runtime Supported: false")
        writer.println("Service Instance: ${if (service == null) "null" else "running"}")
        writer.println("Service Internal Running: ${service?.isServerRunning() ?: false}")
        writer.println("Service Model ID: ${service?.getCurrentModelId() ?: "null"}")
        writer.println("START_REQUEST_COUNT=${service?.getStartRequestCount() ?: 0}")
        writer.println("BOOTSTRAP_COUNT=${service?.getBootstrapCount() ?: 0}")
        writer.println("LISTENER_PRESENT=${listeners.isNotEmpty()}")
        writer.println("LISTENER_NON_LOOPBACK=$hasNonLoopbackListener")

        if (listeners.isEmpty()) {
            writer.println("LISTENER_DETAILS=none")
            return
        }

        listeners.forEach { listener ->
            writer.println(
                "LISTENER_DETAILS=${listener.table}:${listener.address}:${listener.port} uid=${listener.uid} inode=${listener.inode}"
            )
        }
    }
    
    private fun startApiService(writer: PrintWriter, args: List<String>) {
        val context = MnnLlmApplication.getInstance()
        val modelId = parseModelArg(args)
        try {
            // Auto-enable API service via prefs so dumpapp can bootstrap without UI
            val prefsName = "${context.packageName}_preferences"
            val prefs = context.getSharedPreferences(prefsName, android.content.Context.MODE_PRIVATE)
            if (!prefs.getBoolean("enable_api_service", false)) {
                prefs.edit().putBoolean("enable_api_service", true).apply()
                writer.println("enable_api_service was false, auto-enabled via plugin.")
            }
            val intent = Intent(context, OpenAIService::class.java)
            if (!modelId.isNullOrBlank()) {
                intent.putExtra("modelId", modelId)
            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
            writer.println("Service start requested via Intent.")
            if (!modelId.isNullOrBlank()) {
                writer.println("Model ID: $modelId")
            }
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
        writer.println("  diag              Show config + runtime listener diagnostics")
        writer.println("  start [--model <modelId>]  Start the OpenAIService")
        writer.println("  stop              Stop the service")
        writer.println("  reset-config      Reset API settings to default (new API key)")
        writer.println("  test-chat [msg]   Send a test message to the LLM")
    }

    private fun parseModelArg(args: List<String>): String? {
        if (args.isEmpty()) {
            return null
        }
        for (index in 0 until args.size - 1) {
            if (args[index] == "--model") {
                return args[index + 1]
            }
        }
        return null
    }

    private data class TcpListener(
        val table: String,
        val address: String,
        val port: Int,
        val uid: String,
        val inode: String
    ) {
        fun isLoopback(): Boolean {
            return address == "127.0.0.1" || address == "::1" || address == "::ffff:127.0.0.1"
        }
    }

    private fun readListeningSockets(port: Int): List<TcpListener> {
        val targetPortHex = port.toString(16).uppercase().padStart(4, '0')
        val listeners = mutableListOf<TcpListener>()
        listeners += parseProcNetFile("/proc/net/tcp", "tcp", targetPortHex)
        listeners += parseProcNetFile("/proc/net/tcp6", "tcp6", targetPortHex)
        return listeners
    }

    private fun parseProcNetFile(path: String, table: String, targetPortHex: String): List<TcpListener> {
        val file = File(path)
        if (!file.exists()) {
            return emptyList()
        }

        val listeners = mutableListOf<TcpListener>()
        val lines = runCatching { file.readLines() }.getOrElse { return emptyList() }
        lines.drop(1).forEach { line ->
            val cols = line.trim().split(Regex("\\s+"))
            if (cols.size < 10) {
                return@forEach
            }

            val local = cols[1]
            val state = cols[3]
            if (state != "0A") {
                return@forEach
            }

            val parts = local.split(":")
            if (parts.size != 2) {
                return@forEach
            }
            val addrHex = parts[0]
            val portHex = parts[1]
            if (!portHex.equals(targetPortHex, ignoreCase = true)) {
                return@forEach
            }

            val decodedAddress = decodeAddress(table, addrHex)
            val decodedPort = portHex.toIntOrNull(16) ?: return@forEach
            listeners += TcpListener(
                table = table,
                address = decodedAddress,
                port = decodedPort,
                uid = cols[7],
                inode = cols[9]
            )
        }
        return listeners
    }

    private fun decodeAddress(table: String, hexAddress: String): String {
        return if (table == "tcp") {
            decodeIpv4LittleEndian(hexAddress)
        } else {
            decodeIpv6OrMapped(hexAddress)
        }
    }

    private fun decodeIpv4LittleEndian(hexAddress: String): String {
        if (hexAddress.length != 8) {
            return hexAddress
        }
        val bytes = hexAddress.chunked(2).mapNotNull { it.toIntOrNull(16) }
        if (bytes.size != 4) {
            return hexAddress
        }
        return bytes.reversed().joinToString(".")
    }

    private fun decodeIpv6OrMapped(hexAddress: String): String {
        if (hexAddress.length != 32) {
            return hexAddress
        }
        if (hexAddress == "00000000000000000000000000000000") {
            return "::"
        }

        val mappedV4Prefix = "0000000000000000FFFF0000"
        if (hexAddress.startsWith(mappedV4Prefix)) {
            val ipv4 = decodeIpv4LittleEndian(hexAddress.substring(mappedV4Prefix.length))
            return "::ffff:$ipv4"
        }
        return hexAddress
    }
}
