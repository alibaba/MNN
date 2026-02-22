package com.alibaba.mnnllm.android.hotspot

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "LocalWebServer"

/**
 * A minimal HTTP server that serves a landing page on a given [port].
 *
 * Call [start] to begin accepting connections and [stop] to shut down.
 * The server runs on a background thread pool and is safe to start/stop
 * from any thread.
 */
class LocalWebServer(private val port: Int) {

    private var serverSocket: ServerSocket? = null
    private val running = AtomicBoolean(false)
    private var executor: ExecutorService? = null

    /** Starts listening on [port]. No-op if already running. */
    fun start() {
        if (running.getAndSet(true)) return
        val pool = Executors.newCachedThreadPool()
        executor = pool
        pool.execute {
            try {
                val ss = ServerSocket(port)
                serverSocket = ss
                Log.d(TAG, "Web server started on port $port")
                while (running.get()) {
                    try {
                        val client = ss.accept()
                        pool.execute { handleClient(client) }
                    } catch (e: Exception) {
                        if (running.get()) Log.e(TAG, "Accept error", e)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Server error", e)
                running.set(false)
            }
        }
    }

    /** Stops the server, releases the port, and shuts down the thread pool. */
    fun stop() {
        running.set(false)
        try {
            serverSocket?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing server socket", e)
        }
        serverSocket = null
        executor?.shutdownNow()
        executor = null
        Log.d(TAG, "Web server stopped")
    }

    private fun handleClient(socket: Socket) {
        try {
            socket.use {
                val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
                // Consume the HTTP request headers
                var line = reader.readLine()
                while (!line.isNullOrEmpty()) {
                    line = reader.readLine()
                }

                val html = buildLandingPage()
                val body = html.toByteArray(StandardCharsets.UTF_8)
                val header = buildString {
                    append("HTTP/1.1 200 OK\r\n")
                    append("Content-Type: text/html; charset=UTF-8\r\n")
                    append("Content-Length: ${body.size}\r\n")
                    append("Connection: close\r\n")
                    append("\r\n")
                }.toByteArray(StandardCharsets.UTF_8)
                val out: OutputStream = socket.getOutputStream()
                out.write(header)
                out.write(body)
                out.flush()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling client", e)
        }
    }

    private fun buildLandingPage(): String = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Connected!</title>
    <style>
        body { font-family: sans-serif; display: flex; justify-content: center;
               align-items: center; min-height: 100vh; margin: 0; background: #f0f4f8; }
        .card { background: white; border-radius: 16px; padding: 2rem;
                box-shadow: 0 4px 24px rgba(0,0,0,0.1); text-align: center; max-width: 400px; }
        h1 { color: #2d7a44; }
        p  { color: #555; }
    </style>
</head>
<body>
    <div class="card">
        <h1>&#x2705; Connected!</h1>
        <p>You are connected to the local hotspot server.</p>
        <p>Replace this page with your app's content.</p>
    </div>
</body>
</html>"""
}
