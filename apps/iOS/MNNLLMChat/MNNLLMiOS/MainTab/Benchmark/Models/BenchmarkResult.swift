//
//  BenchmarkResult.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/// Structure containing the results of a completed benchmark test.
/// Encapsulates test instance data along with success status and error information.
struct BenchmarkResult {
    let testInstance: TestInstance
    let success: Bool
    let errorMessage: String?
    
    init(testInstance: TestInstance, success: Bool, errorMessage: String? = nil) {
        self.testInstance = testInstance
        self.success = success
        self.errorMessage = errorMessage
    }
}
