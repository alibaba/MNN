package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import com.alibaba.mnnllm.android.R
import kotlin.math.sqrt

/**
 * Helper class for benchmark results calculation and formatting
 * Separates result processing logic from UI
 */
object BenchmarkResultsHelper {
    
    /**
     * Process test results and return formatted statistics
     */
    fun processTestResults(context: Context, testResults: List<TestInstance>): BenchmarkStatistics {
        if (testResults.isEmpty()) {
            return BenchmarkStatistics.empty()
        }
        
        val allPrefillSpeeds = mutableListOf<Double>()
        val allDecodeSpeeds = mutableListOf<Double>()
        
        var totalTokensProcessed = 0
        var configText = context.getString(R.string.benchmark_config) + "\n"
        
        testResults.forEach { testInstance ->
            // Calculate speeds for this test instance
            if (testInstance.prefillUs.isNotEmpty()) {
                val prefillSpeeds = testInstance.getTokensPerSecond(testInstance.nPrompt, testInstance.prefillUs)
                allPrefillSpeeds.addAll(prefillSpeeds)
            }
            
            if (testInstance.decodeUs.isNotEmpty()) {
                val decodeSpeeds = testInstance.getTokensPerSecond(testInstance.nGenerate, testInstance.decodeUs)
                allDecodeSpeeds.addAll(decodeSpeeds)
            }
            
            totalTokensProcessed += testInstance.nPrompt + testInstance.nGenerate
            configText += "PP: ${testInstance.nPrompt} • TG: ${testInstance.nGenerate}\n"
        }
        
        android.util.Log.d("BenchmarkResultsHelper", "Processing results: prefillSpeeds=${allPrefillSpeeds.size}, decodeSpeeds=${allDecodeSpeeds.size}")
        
        val prefillStats = if (allPrefillSpeeds.isNotEmpty()) {
            val avg = allPrefillSpeeds.average()
            val stdev = calculateStdev(allPrefillSpeeds)
            android.util.Log.d("BenchmarkResultsHelper", "Prefill stats: avg=$avg, stdev=$stdev, speeds=$allPrefillSpeeds")
            SpeedStatistics(
                average = avg,
                stdev = stdev,
                label = context.getString(R.string.benchmark_prompt_processing)
            )
        } else {
            android.util.Log.d("BenchmarkResultsHelper", "No prefill speeds available")
            null
        }
        
        val decodeStats = if (allDecodeSpeeds.isNotEmpty()) {
            val avg = allDecodeSpeeds.average()
            val stdev = calculateStdev(allDecodeSpeeds)
            android.util.Log.d("BenchmarkResultsHelper", "Decode stats: avg=$avg, stdev=$stdev, speeds=$allDecodeSpeeds")
            SpeedStatistics(
                average = avg,
                stdev = stdev,
                label = context.getString(R.string.benchmark_token_generation)
            )
        } else {
            android.util.Log.d("BenchmarkResultsHelper", "No decode speeds available")
            null
        }
        
        return BenchmarkStatistics(
            configText = configText.trim(),
            prefillStats = prefillStats,
            decodeStats = decodeStats,
            totalTokensProcessed = totalTokensProcessed,
            totalTests = testResults.size
        )
    }
    
    /**
     * Calculate standard deviation for a list of doubles
     */
    private fun calculateStdev(values: List<Double>): Double {
        if (values.size <= 1) return 0.0
        val mean = values.average()
        val sqSum = values.map { (it - mean) * (it - mean) }.sum()
        return sqrt(sqSum / (values.size - 1))
    }
    
    /**
     * Format speed statistics for display
     */
    fun formatSpeedStatistics(stats: SpeedStatistics): String {
        return "%.2f ± %.2f t/s".format(stats.average, stats.stdev)
    }
    
    /**
     * Format speed value (average and stdev) for display in a single line like "avg ± stdev t/s".
     */
    fun formatSpeedStatisticsLine(stats: SpeedStatistics): String {
        return "%.2f ± %.2f t/s".format(stats.average, stats.stdev)
    }
    
    /**
     * Return only the human-readable label (first line).
     */
    fun formatSpeedLabelOnly(stats: SpeedStatistics): String {
        return stats.label
    }
    
    /**
     * Format speed value (average only) for display
     */
    fun formatSpeedValue(stats: SpeedStatistics): String {
        return "%.2f t/s".format(stats.average)
    }
    
    /**
     * Format speed label with standard deviation for display
     */
    fun formatSpeedLabel(stats: SpeedStatistics): String {
        val result = "${stats.label}\n±%.2f".format(stats.stdev)
        android.util.Log.d("BenchmarkResultsHelper", "formatSpeedLabel: avg=${stats.average}, stdev=${stats.stdev}, result='$result'")
        return result
    }
    
    /**
     * Format memory usage for display
     */
    fun formatMemoryUsage(memoryKb: Long): String {
        return "${memoryKb} KB"
    }
    
    /**
     * Obtain total device memory in KB using /proc/meminfo. Falls back to –1 if unavailable.
     */
    fun getTotalMemoryKb(): Long {
        return try {
            val reader = java.io.RandomAccessFile("/proc/meminfo", "r")
            val line = reader.readLine() // MemTotal: 16384256 kB
            reader.close()
            val parts = line.split(Regex("\\s+"))
            parts[1].toLongOrNull() ?: -1
        } catch (e: Exception) {
            -1
        }
    }
    
    /**
     * Format peak memory usage value and label.
     * Returns Pair<value,label> where value is like "12.0%" and label is "Peak Memory\n3 GB / 24 GB".
     */
    fun formatMemoryUsageDetailed(context: Context, maxMemoryKb: Long): Pair<String, String> {
        val totalKb = getTotalMemoryKb()
        fun kbToGb(kb: Long): String {
            val gb = kb.toDouble() / 1024.0 / 1024.0
            return "%.1f GB".format(gb)
        }
        return Pair(kbToGb(maxMemoryKb), kbToGb(totalKb))
    }
    
    /**
     * Format model parameters info for display
     */
    fun formatModelParams(context: Context, totalTokens: Int, totalTests: Int): String {
        return context.getString(R.string.benchmark_model_params, totalTokens, totalTests)
    }
}

/**
 * Data class representing processed benchmark statistics
 */
data class BenchmarkStatistics(
    val configText: String,
    val prefillStats: SpeedStatistics?,
    val decodeStats: SpeedStatistics?,
    val totalTokensProcessed: Int,
    val totalTests: Int
) {
    companion object {
        fun empty() = BenchmarkStatistics(
            configText = "",
            prefillStats = null,
            decodeStats = null,
            totalTokensProcessed = 0,
            totalTests = 0
        )
    }
}

/**
 * Data class representing speed statistics
 */
data class SpeedStatistics(
    val average: Double,
    val stdev: Double,
    val label: String
) 