//
//  BatchTest.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/10/10.
//

import Foundation

/// Errors that can occur during batch file testing
enum BatchFileTestError: LocalizedError {
    case unsupportedFileFormat
    case invalidFileContent
    case fileNotFound
    case permissionDenied

    var errorDescription: String? {
        switch self {
        case .unsupportedFileFormat:
            return "Unsupported file format. Please use .txt, .json, or .jsonl files."
        case .invalidFileContent:
            return "Invalid file content. Please check the file format and try again."
        case .fileNotFound:
            return "The selected file could not be found."
        case .permissionDenied:
            return "Permission denied. Unable to access the selected file."
        }
    }
}

/// Represents a single test item in a batch
struct BatchTestItem {
    let prompt: String
    let image: String?
    let audio: String?

    init(prompt: String, image: String?, audio: String?) {
        self.prompt = prompt
        self.image = image
        self.audio = audio
    }
}

/// Represents the result of a single test execution
struct BatchTestResult {
    let item: BatchTestItem
    let response: String
    let timestamp: Date

    init(item: BatchTestItem, response: String, timestamp: Date) {
        self.item = item
        self.response = response
        self.timestamp = timestamp
    }
}

/// Available file formats for batch test output
enum BatchTestFileFormat: String, CaseIterable {
    case txt
    case json
    case jsonl

    var displayName: String {
        switch self {
        case .txt:
            return "Text (.txt)"
        case .json:
            return "JSON (.json)"
        case .jsonl:
            return "JSON Lines (.jsonl)"
        }
    }
}

/// Configuration for batch test operations
struct BatchTestConfiguration {
    let outputFormat: BatchTestFileFormat
    let includeTimestamp: Bool
    let includePrompt: Bool

    init(outputFormat: BatchTestFileFormat, includeTimestamp: Bool, includePrompt: Bool) {
        self.outputFormat = outputFormat
        self.includeTimestamp = includeTimestamp
        self.includePrompt = includePrompt
    }
}
