//
//  BatchFileTestView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/9/29.
//

import Foundation
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Activity item source for sharing text files
/// Provides custom text content for sharing operations
class TextFileActivityItemSource: NSObject, UIActivityItemSource {
    private let content: String
    private let filename: String

    /// Initializes the activity item source with content and filename
    /// - Parameters:
    ///   - content: Text content to share
    ///   - filename: Suggested filename for the shared content
    init(content: String, filename: String) {
        self.content = content
        self.filename = filename
        super.init()
    }

    /// Returns the placeholder item for the activity
    /// - Parameter activityViewController: The activity view controller
    /// - Returns: Placeholder string
    func activityViewControllerPlaceholderItem(_: UIActivityViewController) -> Any {
        return content
    }

    /// Returns the actual item for the specified activity type
    /// - Parameters:
    ///   - activityViewController: The activity view controller
    ///   - activityType: The type of activity being performed
    /// - Returns: The content string
    func activityViewController(_: UIActivityViewController, itemForActivityType _: UIActivity.ActivityType?) -> Any? {
        return content
    }

    /// Returns the subject for activities that support it
    /// - Parameters:
    ///   - activityViewController: The activity view controller
    ///   - activityType: The type of activity being performed
    /// - Returns: The filename as subject
    func activityViewController(_: UIActivityViewController, subjectForActivityType _: UIActivity.ActivityType?) -> String {
        return filename
    }
}

/// SwiftUI wrapper for UIActivityViewController
/// Presents the system share sheet for sharing content
struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    let onDismiss: () -> Void

    /// Creates the UIActivityViewController
    /// - Parameter context: The representable context
    /// - Returns: Configured UIActivityViewController
    func makeUIViewController(context _: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
        controller.completionWithItemsHandler = { _, _, _, _ in
            onDismiss()
        }
        return controller
    }

    /// Updates the view controller (no-op for this implementation)
    /// - Parameters:
    ///   - uiViewController: The view controller to update
    ///   - context: The representable context
    func updateUIViewController(_: UIActivityViewController, context _: Context) {
        // No updates needed
    }
}

/// Main view for batch file testing functionality
/// Provides UI for file selection, test execution, and result sharing
struct BatchFileTestView: View {
    // MARK: - Environment and State

    /// Reference to the chat view model for LLM operations
    @ObservedObject var chatViewModel: LLMChatViewModel

    /// View model for batch file testing operations
    @StateObject private var viewModel: BatchFileTestViewModel

    /// Current scene phase for controlling share sheet presentation
    @Environment(\.scenePhase) private var scenePhase

    /// Controls the visibility of the file importer
    @State private var showFileImporter = false

    // MARK: - Initialization

    /// Initializes the view with a chat view model
    /// - Parameter chatViewModel: The LLM chat view model for processing requests
    init(chatViewModel: LLMChatViewModel) {
        self.chatViewModel = chatViewModel
        _viewModel = StateObject(wrappedValue: BatchFileTestViewModel(chatViewModel: chatViewModel))
    }

    // MARK: - View Body

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header section
                    headerSection
                    
                    // File selection section
                    fileSelectionSection
                    
                    // Test items display
                    testItemsSection
                    
                    // Control buttons
                    controlButtonsSection
                    
                    // Progress indicator
                    if viewModel.isTesting {
                        progressSection
                    }
                    
                    // Results section
                    if !viewModel.testResults.isEmpty {
                        resultsSection
                    }
                    
                    Spacer()
                }
            }
            .padding()
            .navigationTitle("Batch File Test")
            .navigationBarTitleDisplayMode(.inline)
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.plainText, .json, .data],
            allowsMultipleSelection: false
        ) { result in
            viewModel.handleFileSelection(result)
        }
        .sheet(isPresented: $viewModel.showingShareSheet, onDismiss: {
            viewModel.onShareSheetDismiss()
        }) {
            if let fileURL = viewModel.shareFileURL {
                ShareSheet(activityItems: [fileURL], onDismiss: {
                    viewModel.onShareSheetDismiss()
                })
            }
        }
        .alert("Error", isPresented: $viewModel.showingError) {
            Button("OK") {}
        } message: {
            Text(viewModel.errorMessage)
        }
        .onChange(of: scenePhase) { oldPhase, newPhase in
            viewModel.handleScenePhaseChange(newPhase)
        }
    }

    // MARK: - View Components

    /// Header section with title and description
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Batch File Testing")
                .font(.title2)
                .fontWeight(.bold)

            Text("Select a file containing test prompts and run batch tests against the LLM model.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// File selection section
    private var fileSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("File Selection")
                .font(.headline)

            Button(action: {
                showFileImporter = true
            }) {
                HStack {
                    Image(systemName: "doc.badge.plus")
                    Text("Select Test File")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isTesting ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .disabled(viewModel.isTesting)

            Text("Supported formats: .txt, .json, .jsonl")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    /// Test items display section
    private var testItemsSection: some View {
        Group {
            if !viewModel.testItems.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Loaded Test Items (\(viewModel.testItems.count))")
                        .font(.headline)

                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(Array(viewModel.testItems.enumerated()), id: \.offset) { index, item in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Test \(index + 1)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)

                                    Text(item.prompt)
                                        .font(.body)
                                        .padding(8)
                                        .background(Color.gray.opacity(0.1))
                                        .cornerRadius(6)
                                }
                            }
                        }
                    }
                    .frame(maxHeight: 200)
                }
            }
        }
    }

    /// Control buttons section
    private var controlButtonsSection: some View {
        Group {
            if !viewModel.testItems.isEmpty {
                HStack(spacing: 16) {
                    // Start/Stop button
                    Button(action: {
                        if viewModel.isTesting {
                            viewModel.stopBatchTest()
                        } else {
                            viewModel.startBatchTest()
                        }
                    }) {
                        HStack {
                            Image(systemName: viewModel.isTesting ? "stop.fill" : "play.fill")
                            Text(viewModel.isTesting ? "Stop Test" : "Start Test")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(viewModel.isTesting ? Color.red : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                    .disabled(viewModel.testItems.isEmpty)

                    // Reset button
                    Button(action: {
                        viewModel.reset()
                    }) {
                        HStack {
                            Image(systemName: "arrow.clockwise")
                            Text("Reset")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
            }
        }
    }

    /// Progress indicator section
    private var progressSection: some View {
        VStack(spacing: 8) {
            Text("Testing in Progress...")
                .font(.headline)
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(8)
    }

    /// Results section
    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Test Results (\(viewModel.testResults.count))")
                    .font(.headline)

                Spacer()

                // Format picker
                Picker("Format", selection: $viewModel.selectedFormat) {
                    ForEach(BatchTestFileFormat.allCases, id: \.self) { format in
                        Text(format.displayName).tag(format)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }

            // Share button
            Button(action: {
                viewModel.generateResultsFile(scenePhase: scenePhase)
            }) {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text("Share Results")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }

            // Results preview
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(viewModel.testResults.enumerated()), id: \.offset) { index, result in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Result \(index + 1)")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text(result)
                                .font(.body)
                                .padding(8)
                                .background(Color.green.opacity(0.1))
                                .cornerRadius(6)
                        }
                    }
                }
            }
            .frame(maxHeight: 200)
        }
    }
}
