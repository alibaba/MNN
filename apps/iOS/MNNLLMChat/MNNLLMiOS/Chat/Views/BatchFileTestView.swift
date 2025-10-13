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
/// Document picker for selecting files from the system
/// Provides a SwiftUI wrapper around UIDocumentPickerViewController
struct DocumentPicker: UIViewControllerRepresentable {
    let onDocumentPicked: (Result<[URL], Error>) -> Void

    /// Creates the document picker view controller
    /// - Parameter context: The representable context
    /// - Returns: Configured UIDocumentPickerViewController
    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        // Support for .txt, .json, and .jsonl files
        let supportedTypes: [UTType] = [
            .text,
            .json,
            UTType(filenameExtension: "jsonl") ?? .plainText
        ]
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: supportedTypes, asCopy: true)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    /// Updates the document picker view controller
    /// - Parameters:
    ///   - uiViewController: The document picker view controller
    ///   - context: The representable context
    func updateUIViewController(_: UIDocumentPickerViewController, context _: Context) {
        // No updates needed
    }

    /// Creates the coordinator for handling delegate callbacks
    /// - Returns: DocumentPickerCoordinator instance
    func makeCoordinator() -> DocumentPickerCoordinator {
        DocumentPickerCoordinator(onDocumentPicked: onDocumentPicked)
    }
}

/// Coordinator for handling document picker delegate methods
/// Manages the communication between UIDocumentPickerViewController and SwiftUI
class DocumentPickerCoordinator: NSObject, UIDocumentPickerDelegate {
    let onDocumentPicked: (Result<[URL], Error>) -> Void

    /// Initializes the coordinator with a completion handler
    /// - Parameter onDocumentPicked: Callback for when documents are selected
    init(onDocumentPicked: @escaping (Result<[URL], Error>) -> Void) {
        self.onDocumentPicked = onDocumentPicked
    }

    /// Called when documents are successfully picked
    /// - Parameter controller: The document picker controller
    func documentPicker(_: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        onDocumentPicked(.success(urls))
    }

    /// Called when document picking is cancelled
    /// - Parameter controller: The document picker controller
    func documentPickerWasCancelled(_: UIDocumentPickerViewController) {
        // User cancelled, no action needed
    }
}

/// Share sheet for sharing content with other apps
/// Provides a SwiftUI wrapper around UIActivityViewController
struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    let onDismiss: () -> Void

    /// Creates the activity view controller
    /// - Parameter context: The representable context
    /// - Returns: Configured UIActivityViewController
    func makeUIViewController(context _: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
        controller.completionWithItemsHandler = { _, _, _, _ in
            onDismiss()
        }
        return controller
    }

    /// Updates the activity view controller
    /// - Parameters:
    ///   - uiViewController: The activity view controller
    ///   - context: The representable context
    func updateUIViewController(_: UIActivityViewController, context _: Context) {
        // No updates needed
    }
}

/// View for batch file testing functionality
/// Provides UI for loading test files, running batch tests, and sharing results
struct BatchFileTestView: View {
    // MARK: - Properties

    /// ViewModel for managing batch test operations
    @StateObject private var viewModel: BatchFileTestViewModel

    /// Current scene phase for managing share sheet presentation
    @Environment(\.scenePhase) private var scenePhase

    /// Presentation state for document picker
    @State private var showingDocumentPicker = false

    /// Presentation state for test type selection
    @State private var showingTestTypeSelection = false

    // MARK: - Initialization

    /// Initializes the view with a chat view model
    /// - Parameter chatViewModel: The LLM chat view model for processing requests
    init(chatViewModel: LLMChatViewModel) {
        _viewModel = StateObject(wrappedValue: BatchFileTestViewModel(chatViewModel: chatViewModel))
    }

    // MARK: - Body

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // File Loading Section
                fileLoadingSection

                // Test Items Display
                testItemsSection

                // Test Controls
                testControlsSection

                // Results Section
                resultsSection

                Spacer()
            }
            .padding()
            .navigationTitle("Batch File Test")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showingDocumentPicker) {
                DocumentPicker { result in
                    handleFileSelection(result)
                }
            }
            .sheet(isPresented: $viewModel.showingShareSheet, onDismiss: viewModel.onShareSheetDismiss) {
                if let shareURL = viewModel.shareFileURL {
                    ShareSheet(activityItems: [shareURL], onDismiss: viewModel.onShareSheetDismiss)
                }
            }
            .alert("Error", isPresented: $viewModel.showingError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage)
            }
            .actionSheet(isPresented: $showingTestTypeSelection) {
                ActionSheet(
                    title: Text("Select Test Type"),
                    message: Text("Choose the type of batch test to run"),
                    buttons: [
                        .default(Text("Text Test")) {
                            viewModel.loadLocalBatchTest(for: .text)
                        },
                        .default(Text("Image Test")) {
                            viewModel.loadLocalBatchTest(for: .image)
                        },
                        .default(Text("Audio Test")) {
                            viewModel.loadLocalBatchTest(for: .audio)
                        },
                        .cancel(),
                    ]
                )
            }
        }
        .onChange(of: scenePhase) { newPhase in
            viewModel.handleScenePhaseChange(newPhase)
        }
    }

    // MARK: - View Components

    /// File loading section with options for local and external files
    private var fileLoadingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Load Test Files")
                .font(.headline)

            HStack {
                Button("Load Local Tests") {
                    showingTestTypeSelection = true
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                Button("Load External File") {
                    showingDocumentPicker = true
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
        .frame(maxWidth: .infinity)
    }

    /// Test items display section
    private var testItemsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Test Items")
                    .font(.headline)

                Spacer()

                Text("\(viewModel.testItems.count) items")
                    .foregroundColor(.secondary)
            }

            if !viewModel.testItems.isEmpty {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(viewModel.testItems.enumerated()), id: \.offset) { index, item in
                            testItemRow(item: item, index: index)
                        }
                    }
                }
                .frame(maxHeight: 400)
            } else {
                Text("No test items loaded")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    /// Individual test item row
    private func testItemRow(item: BatchTestItem, index: Int) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Item \(index + 1)")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                // Test type indicator
                Text(item.testType.displayName)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(testTypeColor(item.testType))
                    .foregroundColor(.white)
                    .cornerRadius(4)
            }

            Text(item.prompt)
                .font(.caption)
                .lineLimit(2)
                .foregroundColor(.primary)
        }
        .padding(8)
        .background(Color(.systemBackground))
        .cornerRadius(6)
    }

    /// Returns color for test type indicator
    private func testTypeColor(_ testType: BatchTestType) -> Color {
        switch testType {
        case .text:
            return .blue
        case .image:
            return .orange
        case .audio:
            return .green
        }
    }

    /// Test controls section
    private var testControlsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Test Controls")
                .font(.headline)

            if viewModel.isTesting {
                VStack(spacing: 8) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())

                    Text("Testing, please wait...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                HStack(spacing: 12) {
                    Button("Start Batch Test") {
                        viewModel.startBatchTest()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.testItems.isEmpty)

                    Spacer()
                    
                    Button("Reset") {
                        viewModel.reset()
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    /// Results section
    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Results")
                    .font(.headline)

                Spacer()

                if !viewModel.testResults.isEmpty {
                    Text("\(viewModel.testResults.count) results")
                        .foregroundColor(.secondary)
                }
            }

            if !viewModel.testResults.isEmpty {
                VStack(spacing: 8) {
                    // Format selection
                    Text("Export Format:")
                        .font(.caption)

                    Picker("Format", selection: $viewModel.selectedFormat) {
                        ForEach(BatchTestFileFormat.allCases, id: \.self) { format in
                            Text(format.displayName).tag(format)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    HStack {
                        Button("Share Results") {
                            viewModel.generateResultsFile(scenePhase: scenePhase)
                        }
                        .buttonStyle(.borderedProminent)
                        Spacer()
                    }
                }
            } else {
                Text("No results available")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    // MARK: - Helper Methods

    /// Handles file selection from document picker
    /// - Parameter result: Result from document picker
    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case let .success(urls):
            guard let selectedURL = urls.first else { return }
            
            // Pass the URL directly to ViewModel for proper permission handling
            viewModel.loadTestFile(from: selectedURL)

        case let .failure(error):
            viewModel.errorMessage = "File selection failed: \(error.localizedDescription)"
            viewModel.showingError = true
        }
    }
}
