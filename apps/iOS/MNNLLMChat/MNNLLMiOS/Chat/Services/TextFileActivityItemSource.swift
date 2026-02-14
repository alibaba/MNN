//
//  TextFileActivityItemSource.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/10/13.
//

import SwiftUI

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
