//
//  ProgressType.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/**
 * Enumeration representing different stages of benchmark execution progress.
 * Used to track and display the current state of benchmark operations.
 */
enum ProgressType: Int, CaseIterable {
    case unknown = 0
    case initializing
    case warmingUp
    case runningTest
    case processingResults
    case completed
    case stopping
    
    var description: String {
        switch self {
        case .unknown: return "Unknown"
        case .initializing: return "Initializing benchmark..."
        case .warmingUp: return "Warming up..."
        case .runningTest: return "Running test"
        case .processingResults: return "Processing results..."
        case .completed: return "All tests completed"
        case .stopping: return "Stopping benchmark..."
        }
    }
}
