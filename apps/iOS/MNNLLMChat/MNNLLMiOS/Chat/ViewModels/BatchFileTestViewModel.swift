//
//  BatchFileTestViewModel.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/10/10.
//

import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// ViewModel for managing batch file testing operations
/// Handles loading test files, executing batch tests, and managing results
@MainActor
class BatchFileTestViewModel: ObservableObject {
    // MARK: - Published Properties
    
    /// Array of test items loaded from file
    @Published var testItems: [BatchTestItem] = []
    
    /// Array of test results from batch execution
    @Published var testResults: [String] = []
    
    /// Whether a batch test is currently running
    @Published var isTesting: Bool = false
    
    /// Selected output file format
    @Published var selectedFormat: BatchTestFileFormat = .jsonl
    
    /// URL for sharing the results file
    @Published var shareFileURL: URL?
    
    /// Whether to show the share sheet
    @Published var showingShareSheet: Bool = false
    
    /// Error message to display
    @Published var errorMessage: String = ""
    
    /// Whether to show error alert
    @Published var showingError: Bool = false
    
    /// Current batch test type
    @Published var currentTestType: BatchTestType = .text
    
    // MARK: - Private Properties
    
    /// Reference to the chat view model for LLM operations
    private let chatViewModel: LLMChatViewModel
    
    /// Service for batch file operations
    private let batchService = BatchFileTestService()
    
    /// Flag to track pending share sheet presentation
    private var pendingSharePresentation = false
    
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
        testResults.removeAll()

        performBatchTest()
    }



    /// Loads test file from the specified URL with proper permission handling
    /// - Parameter url: URL of the test file to load
    func loadTestFile(from url: URL) {
        // For files selected through DocumentPicker, we always need security-scoped access
        // This is required for accessing files outside the app's sandbox
        let hasAccess = url.startAccessingSecurityScopedResource()
        
        // Ensure we stop accessing the resource when done
        defer {
            if hasAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }
        
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
    
    /// Loads test files from LocalBatchTest folder based on test type
    /// - Parameter testType: The type of test to load
    func loadLocalBatchTest(for testType: BatchTestType) {
        currentTestType = testType
        
        guard let bundlePath = Bundle.main.path(forResource: "LocalBatchTest", ofType: nil),
              let localBatchTestURL = URL(string: "file://\(bundlePath)") else {
            showErrorMessage("LocalBatchTest folder not found in bundle.")
            return
        }
        
        let testFolderURL = localBatchTestURL.appendingPathComponent(testType.localBatchTestFolder)
        let promptFileURL = testFolderURL.appendingPathComponent("prompt.jsonl")
        
        do {
            let content = try String(contentsOf: promptFileURL)
            testItems = try parseJSONLContent(content, testType: testType, baseURL: testFolderURL)
            
            if testItems.isEmpty {
                showErrorMessage("No valid test items found in \(testType.displayName) file.")
            }
        } catch {
            showErrorMessage("Failed to load \(testType.displayName) file: \(error.localizedDescription)")
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

    /// Presents the share sheet if the app is in active state
    /// - Parameter scenePhase: Current scene phase
    private func presentShareSheetIfActive(scenePhase: ScenePhase) {
        if scenePhase == .active {
            showingShareSheet = true
            pendingSharePresentation = false
        } else {
            pendingSharePresentation = true
        }
    }

    /// Handles scene phase changes for share sheet presentation
    /// - Parameter scenePhase: New scene phase
    func handleScenePhaseChange(_ scenePhase: ScenePhase) {
        if scenePhase == .active && pendingSharePresentation {
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
        isTesting = false
        testItems.removeAll()
        testResults.removeAll()
        shareFileURL = nil
        showingShareSheet = false
        pendingSharePresentation = false
        errorMessage = ""
        showingError = false
        currentTestType = .text
    }

    // MARK: - Private Methods

    /// Parses file content based on extension
    /// - Parameters:
    ///   - content: File content string
    ///   - fileExtension: File extension
    /// - Returns: Array of parsed test items
    private func parseFileContent(_ content: String, fileExtension: String) throws -> [BatchTestItem] {
        switch fileExtension {
        case "jsonl":
            return try parseJSONLContent(content, testType: .text, baseURL: nil)
        case "txt":
            return parseTextContent(content)
        default:
            throw NSError(domain: "BatchFileTestViewModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unsupported file format: \(fileExtension)"])
        }
    }

    /// Parses JSONL content with support for images and audio
    /// - Parameters:
    ///   - content: JSONL content string
    ///   - testType: Type of test being parsed
    ///   - baseURL: Base URL for resolving relative paths
    /// - Returns: Array of parsed test items
    private func parseJSONLContent(_ content: String, testType: BatchTestType, baseURL: URL?) throws -> [BatchTestItem] {
        let lines = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var items: [BatchTestItem] = []

        for line in lines {
            guard let data = line.data(using: .utf8),
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let prompt = json["prompt"] as? String else {
                continue
            }

            let processedPrompt = batchService.processMediaReferences(json: json, prompt: prompt, baseURL: baseURL)

            let item = BatchTestItem(
                prompt: processedPrompt,
                testType: testType
            )
            items.append(item)
        }

        return items
    }

    /// Parses plain text content
    /// - Parameter content: Text content string
    /// - Returns: Array of parsed test items
    private func parseTextContent(_ content: String) -> [BatchTestItem] {
        return content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { BatchTestItem(prompt: $0, testType: .text) }
    }

    /// Performs the actual batch testing operation
    private func performBatchTest() {
        let totalItems = testItems.count
        guard totalItems > 0 else { return }

        // Extract prompts from test items
        let prompts = testItems.map { $0.prompt }

        chatViewModel.getBatchLLMResponse(prompts: prompts) { [weak self] (responses: [String]) in
            Task { @MainActor in
                guard let self = self else { return }

                // Update results
                self.testResults = responses
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
            let item = index < testItems.count ? testItems[index] : BatchTestItem(prompt: prompt, testType: .text)
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
