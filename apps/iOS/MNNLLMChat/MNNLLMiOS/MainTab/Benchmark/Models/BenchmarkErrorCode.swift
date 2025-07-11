//
//  BenchmarkErrorCode.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/**
 * Enumeration of possible error codes that can occur during benchmark execution.
 * Provides specific error identification for different failure scenarios.
 */
enum BenchmarkErrorCode: Int {
    case benchmarkFailedUnknown = 30
    case testInstanceFailed = 40
    case modelNotInitialized = 50
    case benchmarkRunning = 99
    case benchmarkStopped = 100
    case nativeError = 0
    case modelError = 2
}
