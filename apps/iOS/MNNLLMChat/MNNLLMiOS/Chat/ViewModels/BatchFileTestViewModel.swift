//
//  BatchFileTestViewModel.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/10/10.
//

import Foundation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
class BatchFileTestViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var testItems: [BatchTestItem] = []

    @Published var testResults: [String] = []

    @Published var testProgress: Double = 0.0

    @Published var isTesting: Bool = false

    @Published var showingShareSheet: Bool = false

    @Published var pendingSharePresentation: Bool = false

    @Published var shareFileURL: URL?

    @Published var errorMessage: String = ""

    @Published var showingError: Bool = false

    @Published var selectedFormat: BatchTestFileFormat = .jsonl

    // MARK: - Private Properties

    private var chatViewModel: LLMChatViewModel

    private let batchService = BatchFileTestService()

    private var batchTestTask: Task<Void, Never>?

    // MARK: - Initialization

    /// Initializes the view model with a chat view model reference
    /// - Parameter chatViewModel: The LLM chat view model for processing requests
    init(chatViewModel: LLMChatViewModel) {
        self.chatViewModel = chatViewModel
    }

    // MARK: - Public Methods

    /// Starts the batch testing process
    /// Executes all test items and collects results
    func startBatchTest() {
        guard !testItems.isEmpty else {
            showErrorMessage("No test items available. Please load a test file first.")
            return
        }

        guard !isTesting else {
            showErrorMessage("Batch test is already in progress.")
            return
        }

        isTesting = true
        testProgress = 0.0
        testResults.removeAll()

        batchTestTask = Task {
            await performBatchTest()
        }
    }

    /// Stops the current batch testing process
    func stopBatchTest() {
        batchTestTask?.cancel()
        batchTestTask = nil
        isTesting = false
        testProgress = 0.0
    }

    /// Loads and parses a test file from the given URL
    /// - Parameter url: The URL of the file to load
    func loadTestFile(from url: URL) {
        do {
            let content = try String(contentsOf: url)
            testItems = try parseFileContent(content, fileExtension: url.pathExtension.lowercased())

            if testItems.isEmpty {
                showErrorMessage("No valid test items found in the selected file.")
            }
        } catch {
            showErrorMessage("Failed to load test file: \(error.localizedDescription)")
        }
    }

    /// Generates and shares the results file
    /// - Parameter scenePhase: Current scene phase to control presentation timing
    func generateResultsFile(scenePhase: ScenePhase) {
        guard !testResults.isEmpty else {
            showErrorMessage("No test results available to share.")
            return
        }

        do {
            let fileURL = try generateOutputFile(results: testResults, format: selectedFormat)
            shareFileURL = fileURL
            presentShareSheetIfActive(scenePhase: scenePhase)
        } catch {
            showErrorMessage("Failed to generate results file: \(error.localizedDescription)")
        }
    }

    /// Presents the share sheet if the scene is active, otherwise marks for pending presentation
    /// - Parameter scenePhase: Current scene phase
    func presentShareSheetIfActive(scenePhase: ScenePhase) {
        if scenePhase == .active {
            showingShareSheet = true
            pendingSharePresentation = false
        } else {
            pendingSharePresentation = true
        }
    }

    /// Handles scene phase changes for pending share presentations
    /// - Parameter scenePhase: New scene phase
    func handleScenePhaseChange(_ scenePhase: ScenePhase) {
        if scenePhase == .active, pendingSharePresentation {
            showingShareSheet = true
            pendingSharePresentation = false
        }
    }

    /// Resets the share sheet presentation state
    func onShareSheetDismiss() {
        pendingSharePresentation = false
    }

    /// Resets all test data and state
    func reset() {
        stopBatchTest()
        testItems.removeAll()
        testResults.removeAll()
        testProgress = 0.0
        shareFileURL = nil
        showingShareSheet = false
        pendingSharePresentation = false
        errorMessage = ""
        showingError = false
    }

    /// Handles file selection from document picker
    /// - Parameter result: Result from document picker
    func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case let .success(urls):
            guard let selectedURL = urls.first else { return }

            // Start accessing security-scoped resource
            guard selectedURL.startAccessingSecurityScopedResource() else {
                showErrorMessage("Unable to access the selected file.")
                return
            }

            defer {
                selectedURL.stopAccessingSecurityScopedResource()
            }

            loadTestFile(from: selectedURL)

        case let .failure(error):
            showErrorMessage("File selection failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Private Methods

    /// Parses file content based on file extension
    /// - Parameters:
    ///   - content: Raw file content
    ///   - fileExtension: File extension to determine parsing method
    /// - Returns: Array of parsed test items
    private func parseFileContent(_ content: String, fileExtension: String) throws -> [BatchTestItem] {
        switch fileExtension {
        case "jsonl":
            return try parseJSONLContent(content)
        case "json":
            return try parseJSONContent(content)
        case "txt":
            return parseTextContent(content)
        default:
            throw BatchFileTestError.unsupportedFileFormat
        }
    }

    /// Parses JSONL (JSON Lines) content
    /// - Parameter content: JSONL content string
    /// - Returns: Array of parsed test items
    private func parseJSONLContent(_ content: String) throws -> [BatchTestItem] {
        let lines = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var items: [BatchTestItem] = []

        for line in lines {
            guard let data = line.data(using: .utf8) else { continue }

            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let prompt = json["prompt"] as? String
                {
                    items.append(BatchTestItem(prompt: prompt))
                }
            } catch {
                // Skip invalid JSON lines
                continue
            }
        }

        return items
    }

    /// Parses JSON content
    /// - Parameter content: JSON content string
    /// - Returns: Array of parsed test items
    private func parseJSONContent(_ content: String) throws -> [BatchTestItem] {
        guard let data = content.data(using: .utf8) else {
            throw BatchFileTestError.invalidFileContent
        }

        do {
            if let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                return jsonArray.compactMap { dict in
                    guard let prompt = dict["prompt"] as? String else { return nil }
                    return BatchTestItem(prompt: prompt)
                }
            } else if let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompts = jsonObject["prompts"] as? [String]
            {
                return prompts.map { BatchTestItem(prompt: $0) }
            }
        } catch {
            throw BatchFileTestError.invalidFileContent
        }

        return []
    }

    /// Parses plain text content
    /// - Parameter content: Text content string
    /// - Returns: Array of parsed test items
    private func parseTextContent(_ content: String) -> [BatchTestItem] {
        return content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { BatchTestItem(prompt: $0) }
    }

    /// Performs the actual batch testing operation
    private func performBatchTest() async {
        let totalItems = testItems.count
        guard totalItems > 0 else { return }

        // Extract prompts from test items
        let prompts = testItems.map { $0.prompt }

        chatViewModel.getBatchLLMResponse(prompts: prompts) { [weak self] (responses: [String]) in
            Task { @MainActor in
                guard let self = self else { return }

                // Update results and progress
                self.testResults = responses
                self.testProgress = 1.0
                self.isTesting = false
            }
        }
    }

    /// Generates output file with test results
    /// - Parameters:
    ///   - results: Array of test result strings
    ///   - format: Output file format
    /// - Returns: URL of the generated file
    private func generateOutputFile(results: [String], format: BatchTestFileFormat) throws -> URL {
        // Convert simple string results to structured BatchTestResult array
        let structuredResults: [BatchTestResult] = results.enumerated().map { index, output in
            let prompt = index < testItems.count ? testItems[index].prompt : ""
            let item = BatchTestItem(prompt: prompt)
            return BatchTestResult(item: item, response: output, timestamp: Date())
        }

        // Use default configuration and generate output
        let configuration = BatchTestConfiguration(
            outputFormat: format,
            includeTimestamp: true,
            includePrompt: true
        )

        return try batchService.generateOutputFile(
            results: structuredResults,
            configuration: configuration
        )
    }

    /// Shows an error message to the user
    /// - Parameter message: Error message to display
    private func showErrorMessage(_ message: String) {
        errorMessage = message
        showingError = true
    }
}

// MARK: - Supporting Types

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

    init(prompt: String) {
        self.prompt = prompt
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


/// Service class for batch file operations
class BatchFileTestService {
    /// Generates an output file with the given results and configuration
    /// - Parameters:
    ///   - results: Array of test results
    ///   - configuration: Output configuration
    /// - Returns: URL of the generated file
    func generateOutputFile(results: [BatchTestResult], configuration: BatchTestConfiguration) throws -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileName = "batch_test_results_\(Int(Date().timeIntervalSince1970)).\(configuration.outputFormat.rawValue)"
        let fileURL = documentsPath.appendingPathComponent(fileName)

        let content: String

        switch configuration.outputFormat {
        case .txt:
            content = generateTextOutput(results: results, configuration: configuration)
        case .json:
            content = try generateJSONOutput(results: results, configuration: configuration)
        case .jsonl:
            content = try generateJSONLOutput(results: results, configuration: configuration)
        }

        try content.write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }

    /// Generates text format output
    private func generateTextOutput(results: [BatchTestResult], configuration: BatchTestConfiguration) -> String {
        var lines: [String] = []

        for (index, result) in results.enumerated() {
            lines.append("=== Test \(index + 1) ===")
            
            lines.append("Prompt: \(result.item.prompt)")

            lines.append("Response: \(result.response)")

            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    /// Generates JSON format output
    private func generateJSONOutput(results: [BatchTestResult], configuration: BatchTestConfiguration) throws -> String {
        let jsonResults = results.map { result -> [String: Any] in
            var dict: [String: Any] = [
                "response": result.response,
            ]

            if configuration.includePrompt {
                dict["prompt"] = result.item.prompt
            }

            if configuration.includeTimestamp {
                dict["timestamp"] = ISO8601DateFormatter().string(from: result.timestamp)
            }

            return dict
        }

        let jsonData = try JSONSerialization.data(withJSONObject: jsonResults, options: .prettyPrinted)
        return String(data: jsonData, encoding: .utf8) ?? ""
    }

    /// Generates JSONL format output
    private func generateJSONLOutput(results: [BatchTestResult], configuration: BatchTestConfiguration) throws -> String {
        var lines: [String] = []

        for result in results {
            var dict: [String: Any] = [
                "response": result.response,
            ]

            if configuration.includePrompt {
                dict["prompt"] = result.item.prompt
            }

            if configuration.includeTimestamp {
                dict["timestamp"] = ISO8601DateFormatter().string(from: result.timestamp)
            }

            let jsonData = try JSONSerialization.data(withJSONObject: dict)
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                lines.append(jsonString)
            }
        }

        return lines.joined(separator: "\n")
    }
}
