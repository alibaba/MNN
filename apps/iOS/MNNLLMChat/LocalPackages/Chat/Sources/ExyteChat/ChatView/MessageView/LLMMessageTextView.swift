//
//  LLMMessageTextView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/7.
//

import MarkdownParser
import MarkdownView
import SwiftUI

/// A specialized text view designed for LLM chat messages with typewriter animation.
///
/// This SwiftUI component provides an enhanced text display specifically designed for AI chat applications.
/// It supports both plain text and Markdown rendering with an optional typewriter animation effect
/// that creates a dynamic, engaging user experience during AI response streaming.
///
/// ## Key Features
/// - Typewriter animation for streaming AI responses
/// - Markdown support with custom styling
/// - Smart animation control based on message type and content length
/// - Automatic animation management with proper cleanup
/// - Performance-optimized character-by-character rendering
///
/// ## Usage Examples
///
/// ### Basic AI Message with Typewriter Effect
/// ```swift
/// LLMMessageTextView(
///     text: "Hello! This is an AI response with typewriter animation.",
///     messageUseMarkdown: false,
///     messageId: "msg_001",
///     isAssistantMessage: true,
///     isStreamingMessage: true
/// )
/// ```
///
/// ### Markdown Message with Custom Styling
/// ```swift
/// LLMMessageTextView(
///     text: "**Bold text** and *italic text* with `code blocks`",
///     messageUseMarkdown: true,
///     messageId: "msg_002",
///     isAssistantMessage: true,
///     isStreamingMessage: true
/// )
/// ```
///
/// ### User Message (No Animation)
/// ```swift
/// LLMMessageTextView(
///     text: "This is a user message",
///     messageUseMarkdown: false,
///     messageId: "msg_003",
///     isAssistantMessage: false,
///     isStreamingMessage: false
/// )
/// ```
///
/// ## Animation Configuration
/// - `typingSpeed`: 0.015 seconds per character (adjustable)
/// - `chunkSize`: 1 character per animation frame
/// - Minimum text length for animation: 5 characters
/// - Auto-cleanup on view disappear or streaming completion
@available(iOS 17.0, *)
struct LLMMessageTextView: View {
    /// The text content to be displayed
    let text: String?
    /// Whether to render the text as Markdown
    let messageUseMarkdown: Bool
    /// Unique identifier for the message
    let messageId: String
    /// Whether this is an assistant (AI) message
    let isAssistantMessage: Bool
    /// Whether the message is currently being streamed
    let isStreamingMessage: Bool
    /// Whether the streaming output is complete
    let isOutputComplete: Bool
    /// Callback invoked when animation completes
    let onAnimationComplete: (() -> Void)?

    /// The currently displayed text during animation
    @State private var displayedText: String = ""

    /// Survives cell recreation so the typewriter resumes instead of replaying (fixes head-flicker).
    private static var displayedCountCache: [String: Int] = [:]
    /// Timer for controlling the typewriter animation
    @State private var animationTimer: Timer?

    /// Time interval between each character display (in seconds)
    private let typingSpeed: TimeInterval = 0.015
    /// Number of characters to display per animation frame
    private let chunkSize: Int = 1

    /// Initializes a new LLMMessageTextView
    /// - Parameters:
    ///   - text: The text content to be displayed
    ///   - messageUseMarkdown: Whether to render the text as Markdown. Defaults to `true`.
    ///   - messageId: Unique identifier for the message
    ///   - isAssistantMessage: Whether this is an assistant (AI) message. Defaults to `false`.
    ///   - isStreamingMessage: Whether the message is currently being streamed. Defaults to `false`.
    ///   - isOutputComplete: Whether the streaming output is complete. Defaults to `false`.
    ///   - onAnimationComplete: Callback invoked when animation completes. Defaults to `nil`.
    init(text: String?,
         messageUseMarkdown: Bool = true,
         messageId: String,
         isAssistantMessage: Bool = false,
         isStreamingMessage: Bool = false,
         isOutputComplete: Bool = false,
         onAnimationComplete: (() -> Void)? = nil)
    {
        self.text = text
        self.messageUseMarkdown = messageUseMarkdown
        self.messageId = messageId
        self.isAssistantMessage = isAssistantMessage
        self.isStreamingMessage = isStreamingMessage
        self.isOutputComplete = isOutputComplete
        self.onAnimationComplete = onAnimationComplete
    }

    var body: some View {
        Group {
            if let text = text, !text.isEmpty {
                if isAssistantMessage && isStreamingMessage {
                    typewriterView()
                } else {
                    staticView(text)
                }
            }
        }
        .onAppear {
            if let text = text, isAssistantMessage && isStreamingMessage {
                if isOutputComplete {
                    // Cell recycled after completion: show full text and finalize state
                    Self.displayedCountCache[messageId] = nil
                    stopAnimation()
                    displayedText = text
                    NotificationCenter.default.post(
                        name: NSNotification.Name("StreamingAnimationCompleted"),
                        object: nil,
                        userInfo: ["messageId": messageId]
                    )
                } else {
                    if displayedText.isEmpty,
                       let cached = Self.displayedCountCache[messageId], cached > 0 {
                        displayedText = String(text.prefix(min(cached, text.count)))
                    }
                    if displayedText.isEmpty {
                        startTypewriterAnimation(for: text)
                    } else {
                        continueTypewriterAnimation(with: text)
                    }
                }
            } else if let text = text {
                displayedText = text
            }
        }
        .onDisappear {
            stopAnimation()
            Self.displayedCountCache[messageId] = nil
            if let text = text, isOutputComplete, displayedText.count < text.count {
                displayedText = text
                NotificationCenter.default.post(
                    name: NSNotification.Name("StreamingAnimationCompleted"),
                    object: nil,
                    userInfo: ["messageId": messageId]
                )
            }
        }
        .onChange(of: text) { _, newText in
            handleTextChange(newText)
        }
        .onChange(of: isStreamingMessage) { _, _ in
            handleStreamingStateChange()
        }
        .onChange(of: isOutputComplete) { _, _ in
            handleStreamingStateChange()
        }
    }

    /// Renders text with typewriter animation effect
    /// - Returns: A view displaying the animated text with optional Markdown support
    @ViewBuilder
    private func typewriterView() -> some View {
        if messageUseMarkdown {
            MarkdownTextViewWrapper(text: displayedText)
        } else {
            Text(displayedText)
                .font(.body)
                .foregroundColor(.black)
        }
    }

    /// Renders static text without animation
    /// - Parameter text: The text to be displayed
    /// - Returns: A view displaying the complete text with optional Markdown support
    @ViewBuilder
    private func staticView(_ text: String) -> some View {
        if messageUseMarkdown && isAssistantMessage {
            MarkdownTextViewWrapper(text: text)
        } else {
            Text(text)
                .font(.body)
                .foregroundColor(.black)
        }
    }

    /// Handles streaming state changes.
    ///
    /// Called when streaming state changes to manage animation start, stop, and notification sending.
    private func handleStreamingStateChange() {
        if isStreamingMessage, isAssistantMessage {
            // Start or continue streaming output animation
            if let text = text {
                if displayedText.isEmpty {
                    startTypewriterAnimation(for: text)
                } else {
                    continueTypewriterAnimation(with: text)
                }
            }
        } else if isOutputComplete {
            // Output complete, display full text and send notification
            if let text = text {
                displayedText = text
            }
            stopAnimation()
            // Send animation completion notification
            NotificationCenter.default.post(
                name: NSNotification.Name("StreamingAnimationCompleted"),
                object: nil,
                userInfo: ["messageId": messageId]
            )
        } else {
            // Non-streaming or fully complete, display static text
            if let text = text {
                displayedText = text
            }
            stopAnimation()
        }
    }

    /// Handles text content changes during streaming
    ///
    /// This method intelligently manages animation continuation, restart, or direct display
    /// based on the relationship between old and new text content.
    ///
    /// - Parameter newText: The updated text content
    private func handleTextChange(_ newText: String?) {
        guard let newText = newText else {
            displayedText = ""
            stopAnimation()
            return
        }

        if isAssistantMessage, isStreamingMessage {
            // Check if new text is an extension of current displayed text
            if newText.hasPrefix(displayedText), newText != displayedText {
                // Continue typewriter animation
                continueTypewriterAnimation(with: newText)
            } else if newText != displayedText {
                // Restart animation with new content
                restartTypewriterAnimation(with: newText)
            }
        } else {
            // Display text directly without animation
            displayedText = newText
            stopAnimation()
        }
    }

    /// Initiates typewriter animation for the given text
    /// - Parameter text: The text to animate
    private func startTypewriterAnimation(for text: String) {
        displayedText = ""
        continueTypewriterAnimation(with: text)
    }

    /// Continues or resumes typewriter animation
    ///
    /// This method sets up a timer-based animation that progressively reveals
    /// characters at the configured typing speed.
    ///
    /// - Parameter text: The complete text to animate
    private func continueTypewriterAnimation(with text: String) {
        guard displayedText.count < text.count else { return }

        stopAnimation()

        animationTimer = Timer.scheduledTimer(withTimeInterval: typingSpeed, repeats: true) { _ in
            DispatchQueue.main.async {
                self.appendNextCharacters(from: text)
            }
        }
    }

    /// Restarts typewriter animation with new content
    /// - Parameter text: The new text to animate
    private func restartTypewriterAnimation(with text: String) {
        stopAnimation()
        displayedText = ""
        startTypewriterAnimation(for: text)
    }

    /// Appends the next character(s) to the displayed text
    ///
    /// This method is called by the animation timer to progressively reveal
    /// text characters. It handles proper string indexing and animation completion.
    ///
    /// - Parameter text: The source text to extract characters from
    private func appendNextCharacters(from text: String) {
        let currentLength = displayedText.count
        guard currentLength < text.count else {
            stopAnimation()
            return
        }

        let endIndex = min(currentLength + chunkSize, text.count)
        let startIndex = text.index(text.startIndex, offsetBy: currentLength)
        let targetIndex = text.index(text.startIndex, offsetBy: endIndex)

        let newChars = text[startIndex ..< targetIndex]
        displayedText.append(String(newChars))
        Self.displayedCountCache[messageId] = displayedText.count

        if displayedText.count >= text.count {
            Self.displayedCountCache[messageId] = nil
            stopAnimation()
            // Send animation completion notification when text display is complete and output is also complete
            if isOutputComplete {
                NotificationCenter.default.post(
                    name: NSNotification.Name("StreamingAnimationCompleted"),
                    object: nil,
                    userInfo: ["messageId": messageId]
                )
            }
        }
    }

    /// Stops and cleans up the typewriter animation
    ///
    /// This method should be called when animation is no longer needed
    /// to prevent memory leaks and unnecessary timer execution.
    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil

        // Notify that animation has completed
        onAnimationComplete?()
    }
}

// MARK: - Preview Provider

@available(iOS 17.0, *)
struct LLMMessageTextView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            LLMMessageTextView(
                text: "This is a typewriter animation demo text. Hello, this demonstrates the streaming effect!",
                messageUseMarkdown: false,
                messageId: "test1",
                isAssistantMessage: true,
                isStreamingMessage: true,
                isOutputComplete: false
            )

            LLMMessageTextView(
                text: "**Bold text** and *italic text* with markdown support.",
                messageUseMarkdown: true,
                messageId: "test2",
                isAssistantMessage: true,
                isStreamingMessage: false,
                isOutputComplete: true
            )

            LLMMessageTextView(
                text: "Regular user message without animation.",
                messageUseMarkdown: false,
                messageId: "test3",
                isAssistantMessage: false,
                isStreamingMessage: false,
                isOutputComplete: false
            )
        }
        .padding()
    }
}
