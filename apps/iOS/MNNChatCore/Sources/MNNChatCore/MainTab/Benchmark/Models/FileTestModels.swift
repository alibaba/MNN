//
//  FileTestModels.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/9/28.
//

import Foundation

/// Represents a single test item from the JSON file
struct FileTestItem: Codable, Identifiable {
    let id = UUID()
    let prompt: String
    var answer: String
    
    enum CodingKeys: String, CodingKey {
        case prompt
        case answer
    }
    
    init(prompt: String, answer: String = "") {
        self.prompt = prompt
        self.answer = answer
    }
}

/// Progress information for file testing
struct FileTestProgress {
    let totalItems: Int
    let currentIndex: Int
    let currentPrompt: String
    let isCompleted: Bool
    
    var progressPercentage: Double {
        guard totalItems > 0 else { return 0.0 }
        return Double(currentIndex) / Double(totalItems)
    }
    
    var progressText: String {
        return "\(currentIndex)/\(totalItems)"
    }
}

/// State of the file test process
enum FileTestState {
    case idle
    case fileSelected(URL)
    case testing(FileTestProgress)
    case completed(URL) // URL of the output file
    case error(String)
}

/// Result of file test operation
struct FileTestResult {
    let inputFileURL: URL
    let outputFileURL: URL
    let totalItems: Int
    let completedItems: Int
    let startTime: Date
    let endTime: Date
    let errors: [String]
    
    var duration: TimeInterval {
        return endTime.timeIntervalSince(startTime)
    }
    
    var successRate: Double {
        guard totalItems > 0 else { return 0.0 }
        return Double(completedItems) / Double(totalItems)
    }
}

/// Configuration for file testing
struct FileTestConfiguration {
    let maxRetries: Int
    let timeoutPerItem: TimeInterval
    let outputDirectory: URL
    
    static let `default` = FileTestConfiguration(
        maxRetries: 3,
        timeoutPerItem: 60.0,
        outputDirectory: FileManager.default.temporaryDirectory
    )
}

/// Errors that can occur during file testing operations
enum FileTestError: Error, LocalizedError {
    case fileNotFound
    case invalidFileFormat
    case parsingError(String)
    case processingError(String)
    case modelNotSelected
    case modelInitializationFailed
    case engineNotAvailable
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound:
            return "Test file not found"
        case .invalidFileFormat:
            return "Invalid file format"
        case .parsingError(let message):
            return "Parsing error: \(message)"
        case .processingError(let message):
            return "Processing error: \(message)"
        case .modelNotSelected:
            return "No model selected"
        case .modelInitializationFailed:
            return "Failed to initialize model"
        case .engineNotAvailable:
            return "LLM engine not available"
        }
    }
}

// MARK: - Extensions

extension DateFormatter {
    /// Date formatter for file timestamps
    static let fileTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
}