//
//  BenchmarkStatistics.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/// Structure containing comprehensive statistical analysis of benchmark results.
/// Aggregates performance metrics, configuration details, and test summary information.
struct BenchmarkStatistics {
    let configText: String
    let prefillStats: SpeedStatistics?
    let decodeStats: SpeedStatistics?
    let totalTokensProcessed: Int
    let totalTests: Int
    
    static let empty = BenchmarkStatistics(
        configText: "",
        prefillStats: nil,
        decodeStats: nil,
        totalTokensProcessed: 0,
        totalTests: 0
    )
}
