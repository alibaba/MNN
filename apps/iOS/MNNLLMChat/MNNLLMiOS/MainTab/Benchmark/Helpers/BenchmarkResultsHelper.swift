//
//  BenchmarkResultsHelper.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation
import Darwin

/**
 * Helper class for processing and formatting benchmark test results.
 * Provides statistical analysis, formatting utilities, and device information
 * for benchmark result display and sharing.
 */
class BenchmarkResultsHelper {
    static let shared = BenchmarkResultsHelper()
    
    private init() {}
    
    // MARK: - Results Processing & Statistics
    
    /// Processes test results to generate comprehensive benchmark statistics
    /// - Parameter testResults: Array of completed test instances
    /// - Returns: Processed statistics including speed metrics and configuration details
    func processTestResults(_ testResults: [TestInstance]) -> BenchmarkStatistics {
        guard !testResults.isEmpty else {
            return BenchmarkStatistics.empty
        }
        
        let firstTest = testResults[0]
        let configText = "Backend: CPU, Threads: \(firstTest.threads), Memory: Low, Precision: Low"
        
        var prefillStats: SpeedStatistics?
        var decodeStats: SpeedStatistics?
        var totalTokensProcessed = 0
        
        // Calculate prefill (prompt processing) statistics
        let allPrefillSpeeds = testResults.flatMap { test in
            test.getTokensPerSecond(tokens: test.nPrompt, timesUs: test.prefillUs)
        }
        
        if !allPrefillSpeeds.isEmpty {
            let avgPrefill = allPrefillSpeeds.reduce(0, +) / Double(allPrefillSpeeds.count)
            let stdevPrefill = calculateStandardDeviation(values: allPrefillSpeeds, mean: avgPrefill)
            prefillStats = SpeedStatistics(average: avgPrefill, stdev: stdevPrefill, label: "Prompt Processing")
        }
        
        // Calculate decode (token generation) statistics
        let allDecodeSpeeds = testResults.flatMap { test in
            test.getTokensPerSecond(tokens: test.nGenerate, timesUs: test.decodeUs)
        }
        
        if !allDecodeSpeeds.isEmpty {
            let avgDecode = allDecodeSpeeds.reduce(0, +) / Double(allDecodeSpeeds.count)
            let stdevDecode = calculateStandardDeviation(values: allDecodeSpeeds, mean: avgDecode)
            decodeStats = SpeedStatistics(average: avgDecode, stdev: stdevDecode, label: "Token Generation")
        }
        
        // Calculate total tokens processed across all tests
        totalTokensProcessed = testResults.reduce(0) { sum, test in
            return sum + (test.nPrompt * test.prefillUs.count) + (test.nGenerate * test.decodeUs.count)
        }
        
        return BenchmarkStatistics(
            configText: configText,
            prefillStats: prefillStats,
            decodeStats: decodeStats,
            totalTokensProcessed: totalTokensProcessed,
            totalTests: testResults.count
        )
    }
    
    /// Calculates standard deviation for a set of values
    /// - Parameters:
    ///   - values: Array of numeric values
    ///   - mean: Pre-calculated mean of the values
    /// - Returns: Standard deviation value
    private func calculateStandardDeviation(values: [Double], mean: Double) -> Double {
        guard values.count > 1 else { return 0.0 }
        
        let variance = values.reduce(0) { sum, value in
            let diff = value - mean
            return sum + (diff * diff)
        } / Double(values.count - 1)
        
        return sqrt(variance)
    }
    
    // MARK: - Formatting & Display
    
    /// Formats speed statistics with average and standard deviation
    /// - Parameter stats: Speed statistics to format
    /// - Returns: Formatted string like "42.5 ± 3.2 tok/s"
    func formatSpeedStatisticsLine(_ stats: SpeedStatistics) -> String {
        return String(format: "%.1f ± %.1f tok/s", stats.average, stats.stdev)
    }
    
    /// Returns the label-only portion of speed statistics
    /// - Parameter stats: Speed statistics object
    /// - Returns: Human-readable label (e.g., "Prompt Processing")
    func formatSpeedLabelOnly(_ stats: SpeedStatistics) -> String {
        return stats.label
    }
    
    /// Formats model parameter summary for display
    /// - Parameters:
    ///   - totalTokens: Total number of tokens processed
    ///   - totalTests: Total number of tests completed
    /// - Returns: Formatted summary string
    func formatModelParams(totalTokens: Int, totalTests: Int) -> String {
        return "Total Tokens: \(totalTokens), Tests: \(totalTests)"
    }
    
    /// Formats memory usage with percentage and absolute values
    /// - Parameters:
    ///   - maxMemoryKb: Peak memory usage in kilobytes
    ///   - totalKb: Total system memory in kilobytes
    /// - Returns: Tuple containing formatted value and percentage label
    func formatMemoryUsage(maxMemoryKb: Int64, totalKb: Int64) -> (valueText: String, labelText: String) {
        let maxMemoryMB = Double(maxMemoryKb) / 1024.0
        let totalMemoryGB = Double(totalKb) / (1024.0 * 1024.0)
        let percentage = (Double(maxMemoryKb) / Double(totalKb)) * 100.0
        
        let valueText = String(format: "%.1f MB", maxMemoryMB)
        let labelText = String(format: "%.1f%% of %.1f GB", percentage, totalMemoryGB)
        
        return (valueText, labelText)
    }
    
    // MARK: - Device & System Information
    
    /// Gets comprehensive device information including model and iOS version
    /// - Returns: Formatted device info string (e.g., "iPhone 14 Pro, iOS 17.0")
    func getDeviceInfo() -> String {
        return DeviceInfoHelper.shared.getDeviceInfo()
    }
    
    /// Gets total system memory in kilobytes
    /// - Returns: System memory size in KB
    func getTotalSystemMemoryKb() -> Int64 {
        return Int64(ProcessInfo.processInfo.physicalMemory) / 1024
    }
}
