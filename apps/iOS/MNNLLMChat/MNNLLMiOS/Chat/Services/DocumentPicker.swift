//
//  DocumentPicker.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/10/13.
//

import UIKit
import SwiftUI
import UniformTypeIdentifiers

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
