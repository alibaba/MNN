//
//  BenchmarkResults.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/// Structure containing comprehensive benchmark results for display and sharing.
/// Aggregates test results, memory usage, and metadata for result presentation.
struct BenchmarkResults {
    let modelDisplayName: String
    let maxMemoryKb: Int64
    let testResults: [TestInstance]
    let timestamp: String
    let totalTimeSeconds: Float
    
    init(modelDisplayName: String, maxMemoryKb: Int64, testResults: [TestInstance], timestamp: String, totalTimeSeconds: Float = 0.0) {
        self.modelDisplayName = modelDisplayName
        self.maxMemoryKb = maxMemoryKb
        self.testResults = testResults
        self.timestamp = timestamp
        self.totalTimeSeconds = totalTimeSeconds
    }
}
