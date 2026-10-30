package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.delay

internal class BenchmarkEnergyRecorder(context: Context) : Closeable {
    private val appContext = context.applicationContext
    private val timestamp = System.currentTimeMillis()
    private val monitor = BatteryEnergyMonitor(appContext)
    private val samples = BenchmarkEnergySamples()

    init { monitor.start() }

    fun onProgress(progress: BenchmarkProgress) {
        if (progress.progressType == ProgressType.RUNNING_TEST) {
            samples.record(progress, monitor.capture().timeNs)
        }
    }

    suspend fun finish(expectedRounds: Int, stopped: () -> Boolean, progress: suspend (Int) -> Unit): BenchmarkEnergyResult {
        val start = monitor.capture().timeNs
        val deadline = start + 10_000_000_000L
        var shownSecond = -1
        while (!stopped()) {
            val now = SystemClock.elapsedRealtimeNanos()
            if (now >= deadline) break
            val second = ((now - start) / 1_000_000_000L).toInt()
            if (second != shownSecond) {
                progress(second)
                shownSecond = second
            }
            delay(minOf(100L, ((deadline - now) / 1_000_000L).coerceAtLeast(1L)))
        }
        val end = monitor.capture().timeNs
        val records = samples.completed.toList()
        val load = monitor.measure(records)
        val idle = monitor.measure(listOf(EnergyInterval(start, end)))
        val failure = when {
            stopped() -> EnergyUnavailableReason.INTERRUPTED
            samples.inFlight || samples.invalid || records.size != expectedRounds -> EnergyUnavailableReason.INCOMPLETE
            else -> null
        }
        return BenchmarkEnergyResult(load, idle, records, failure)
    }

    fun save(modelId: String, config: String, parameters: CommandParameters, result: BenchmarkEnergyResult) {
        try {
            val directory = File(appContext.filesDir, "benchmark_energy").apply { mkdirs() }
            val file = File(directory, "performance-energy-$timestamp.json")
            val data = mapOf(
                "schemaVersion" to 2, "protocol" to "performance-with-post-idle-10s", "timestampMs" to timestamp,
                "modelId" to modelId, "modelConfig" to JsonParser.parseString(config), "parameters" to parameters,
                "device" to Build.MODEL, "build" to Build.DISPLAY,
                "measurementBoundary" to "whole-device-battery", "method" to "current-voltage-integration",
                "sampleIntervalMs" to 50, "currentConvention" to monitor.currentConvention,
                "warmupIncluded" to false, "baselineTargetSeconds" to 10,
                "tokenCounts" to "configured-per-round; see parameters.nPrompt and parameters.nGenerate",
                "result" to result, "reason" to result.reason,
                "baselineWatts" to result.baselineWatts, "incrementalWatts" to result.incrementalWatts,
                "incrementalJoules" to result.incrementalJoules,
                "readings" to monitor.snapshot()
            )
            val temporary = File(directory, "${file.name}.tmp")
            temporary.writeText(GsonBuilder().setPrettyPrinting().create().toJson(data))
            check(temporary.renameTo(file))
            Log.i("BenchmarkEnergy", "Saved ${file.absolutePath}")
        } catch (error: Exception) {
            Log.e("BenchmarkEnergy", "Could not save energy report", error)
        }
    }

    override fun close() = monitor.close()
}
