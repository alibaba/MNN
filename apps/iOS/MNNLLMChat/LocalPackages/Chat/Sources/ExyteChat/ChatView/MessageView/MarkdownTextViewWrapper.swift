//
//  MarkdownTextViewWrapper.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/9/10.
//

import MarkdownParser
import MarkdownView
import SwiftUI

/// A UIViewRepresentable wrapper for MarkdownTextView that enables the use of MarkdownTextView (UIKit) within SwiftUI views.
///
/// This wrapper handles the conversion of markdown text to rendered content and manages
/// the lifecycle of the underlying MarkdownTextView. It includes performance optimizations
/// such as content caching and smart cache key generation.
///
/// ## Features
/// - Seamless integration of UIKit MarkdownTextView in SwiftUI
/// - Intelligent content caching for improved performance
/// - Support for theme and interface style changes
/// - Optional scroll view binding for synchronized scrolling
///
/// ## Usage
/// ```swift
/// MarkdownTextViewWrapper(
///     text: "**Bold text** and *italic text* with `code blocks`",
///     bindScrollView: false
/// )
/// ```
///
/// - Parameters:
///   - text: The markdown text to be rendered
///   - bindScrollView: Whether to bind the content offset from parent scroll view
public struct MarkdownTextViewWrapper: UIViewRepresentable {
    /// The markdown text to be rendered
    let text: String
    /// Whether to bind the content offset from parent scroll view
    let bindScrollView: Bool

    /// A container class for caching preprocessed markdown content
    ///
    /// This class wraps the `MarkdownTextView.PreprocessedContent` to enable
    /// storage in `NSCache` which requires NSObject conformance.
    private final class PreprocessedBox: NSObject {
        /// The cached preprocessed content
        let content: MarkdownTextView.PreprocessedContent

        /// Initializes a new preprocessed content box
        /// - Parameter content: The preprocessed content to cache
        init(_ content: MarkdownTextView.PreprocessedContent) {
            self.content = content
        }
    }

    /// Centralized cache for preprocessed markdown content
    ///
    /// This cache improves performance by storing preprocessed markdown content
    /// and reusing it when the same content is rendered again.
    private static let cache = NSCache<NSString, PreprocessedBox>()

    /// Initializes a new MarkdownTextViewWrapper
    /// - Parameters:
    ///   - text: The markdown text to be rendered
    ///   - bindScrollView: Whether to bind the content offset from parent scroll view. Defaults to `false`.
    public init(text: String, bindScrollView: Bool = false) {
        self.text = text
        self.bindScrollView = bindScrollView
    }

    /// Creates the underlying MarkdownTextView
    /// - Parameter context: The representable context
    /// - Returns: A configured MarkdownTextView instance
    public func makeUIView(context _: Context) -> MarkdownTextView {
        let markdownView = MarkdownTextView()
        markdownView.theme = MarkdownTheme.default
        return markdownView
    }

    /// Updates the MarkdownTextView with new content
    ///
    /// This method implements intelligent caching to avoid reprocessing the same content.
    /// The cache key includes text hash, theme properties, and interface style to ensure
    /// proper cache invalidation when any of these factors change.
    ///
    /// - Parameters:
    ///   - uiView: The MarkdownTextView to update
    ///   - context: The representable context
    public func updateUIView(_ uiView: MarkdownTextView, context _: Context) {
        // Build cache key based on text + theme + interface style to avoid stale colors
        let theme = uiView.theme
        let style = uiView.traitCollection.userInterfaceStyle
        let keyString = "v1|\(text.hashValue)|b:\(Int(theme.fonts.body.pointSize))|c:\(Int(theme.fonts.code.pointSize))|s:\(style.rawValue)" as NSString

        if let cached = Self.cache.object(forKey: keyString) {
            uiView.setMarkdownManually(cached.content)
        } else {
            let parser = MarkdownParser()
            let parseResult = parser.parse(text)
            let preprocessedContent = MarkdownTextView.PreprocessedContent(
                parserResult: parseResult,
                theme: theme
            )
            Self.cache.setObject(PreprocessedBox(preprocessedContent), forKey: keyString)
            uiView.setMarkdownManually(preprocessedContent)
        }

        if bindScrollView {
            DispatchQueue.main.async {
                if let scrollView = uiView.findParentScrollView() {
                    uiView.bindContentOffset(from: scrollView)
                }
            }
        }
    }

    /// Calculates the size that fits the proposed size
    /// - Parameters:
    ///   - proposal: The proposed view size
    ///   - uiView: The MarkdownTextView
    ///   - context: The representable context
    /// - Returns: The calculated size that fits the content
    public func sizeThatFits(_ proposal: ProposedViewSize, uiView: MarkdownTextView, context _: Context) -> CGSize? {
        let width = proposal.width ?? UIView.layoutFittingExpandedSize.width
        let boundingSize = uiView.boundingSize(for: width)
        return boundingSize
    }
}

// MARK: - UIView Extension

/// Extension for UIView to find parent ScrollView
extension UIView {
    /// Finds the parent ScrollView in the view hierarchy
    ///
    /// This method traverses up the view hierarchy to locate the nearest
    /// UIScrollView ancestor. This is useful for binding scroll positions
    /// or implementing scroll-related functionality.
    ///
    /// - Returns: The parent UIScrollView if found, otherwise nil
    func findParentScrollView() -> UIScrollView? {
        var currentView: UIView? = superview
        while let view = currentView {
            if let scrollView = view as? UIScrollView {
                return scrollView
            }
            currentView = view.superview
        }
        return nil
    }
}
