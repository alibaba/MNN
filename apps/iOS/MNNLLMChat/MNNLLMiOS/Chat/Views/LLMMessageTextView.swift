//
//  LLMMessageTextView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/7.
//

import SwiftUI
import MarkdownUI

/**
 * LLMMessageTextView - A specialized text view designed for LLM chat messages with typewriter animation
 * 
 * This SwiftUI component provides an enhanced text display specifically designed for AI chat applications.
 * It supports both plain text and Markdown rendering with an optional typewriter animation effect
 * that creates a dynamic, engaging user experience during AI response streaming.
 * 
 * Key Features:
 * - Typewriter animation for streaming AI responses
 * - Markdown support with custom styling
 * - Smart animation control based on message type and content length
 * - Automatic animation management with proper cleanup
 * - Performance-optimized character-by-character rendering
 * 
 * Usage Examples:
 * 
 * 1. Basic AI Message with Typewriter Effect:
 * ```swift
 * LLMMessageTextView(
 *     text: "Hello! This is an AI response with typewriter animation.",
 *     messageUseMarkdown: false,
 *     messageId: "msg_001",
 *     isAssistantMessage: true,
 *     isStreamingMessage: true
 * )
 * ```
 * 
 * 2. Markdown Message with Custom Styling:
 * ```swift
 * LLMMessageTextView(
 *     text: "**Bold text** and *italic text* with `code blocks`",
 *     messageUseMarkdown: true,
 *     messageId: "msg_002",
 *     isAssistantMessage: true,
 *     isStreamingMessage: true
 * )
 * ```
 * 
 * 3. User Message (No Animation):
 * ```swift
 * LLMMessageTextView(
 *     text: "This is a user message",
 *     messageUseMarkdown: false,
 *     messageId: "msg_003",
 *     isAssistantMessage: false,
 *     isStreamingMessage: false
 * )
 * ```
 * 
 * Animation Configuration:
 * - typingSpeed: 0.015 seconds per character (adjustable)
 * - chunkSize: 1 character per animation frame
 * - Minimum text length for animation: 5 characters
 * - Auto-cleanup on view disappear or streaming completion
 */
struct LLMMessageTextView: View {
    let text: String?
    let messageUseMarkdown: Bool
    let messageId: String
    let isAssistantMessage: Bool
    let isStreamingMessage: Bool // Whether this message is currently being streamed
    
    @State private var displayedText: String = ""
    @State private var animationTimer: Timer?
    
    // Typewriter animation configuration
    private let typingSpeed: TimeInterval = 0.015 // Time interval per character
    private let chunkSize: Int = 1 // Number of characters to display per frame
    
    init(text: String?, 
         messageUseMarkdown: Bool = false,
         messageId: String,
         isAssistantMessage: Bool = false,
         isStreamingMessage: Bool = false) {
        self.text = text
        self.messageUseMarkdown = messageUseMarkdown
        self.messageId = messageId
        self.isAssistantMessage = isAssistantMessage
        self.isStreamingMessage = isStreamingMessage
    }
    
    var body: some View {
        Group {
            if let text = text, !text.isEmpty {
                if isAssistantMessage && isStreamingMessage && shouldUseTypewriter {
                    typewriterView(text)
                } else {
                    staticView(text)
                }
            }
        }
        .onAppear {
            if let text = text, isAssistantMessage && isStreamingMessage && shouldUseTypewriter {
                startTypewriterAnimation(for: text)
            } else if let text = text {
                displayedText = text
            }
        }
        .onDisappear {
            stopAnimation()
        }
        .onChange(of: text) { oldText, newText in
            handleTextChange(newText)
        }
        .onChange(of: isStreamingMessage) { oldIsStreaming, newIsStreaming in
            if !newIsStreaming {
                // Streaming ended, display complete text
                if let text = text {
                    displayedText = text
                }
                stopAnimation()
            }
        }
    }
    
    /**
     * Determines whether typewriter animation should be used
     * 
     * Animation is enabled only for assistant messages with more than 5 characters
     * to avoid unnecessary animation for short responses.
     */
    private var shouldUseTypewriter: Bool {
        // Enable typewriter effect only for assistant messages with sufficient length
        return isAssistantMessage && (text?.count ?? 0) > 5
    }
    
    /**
     * Renders text with typewriter animation effect
     * 
     * - Parameter text: The complete text to be animated
     * - Returns: A view displaying the animated text with optional Markdown support
     */
    @ViewBuilder
    private func typewriterView(_ text: String) -> some View {
        if messageUseMarkdown {
            Markdown(displayedText)
                .markdownBlockStyle(\.blockquote) { configuration in
                  configuration.label
                    .padding()
                    .markdownTextStyle {
                        FontSize(13)
                        FontWeight(.light)
                        BackgroundColor(nil)
                    }
                    .overlay(alignment: .leading) {
                      Rectangle()
                        .fill(Color.gray)
                        .frame(width: 4)
                    }
                    .background(Color.gray.opacity(0.2))
                }
        } else {
            Text(displayedText)
        }
    }
    
    /**
     * Renders static text without animation
     * 
     * - Parameter text: The text to be displayed
     * - Returns: A view displaying the complete text with optional Markdown support
     */
    @ViewBuilder
    private func staticView(_ text: String) -> some View {
        if messageUseMarkdown {
            Markdown(text)
                .markdownBlockStyle(\.blockquote) { configuration in
                  configuration.label
                    .padding()
                    .markdownTextStyle {
                        FontSize(13)
                        FontWeight(.light)
                        BackgroundColor(nil)
                    }
                    .overlay(alignment: .leading) {
                      Rectangle()
                        .fill(Color.gray)
                        .frame(width: 4)
                    }
                    .background(Color.gray.opacity(0.2))
                }
        } else {
            Text(text)
        }
    }
    
    /**
     * Handles text content changes during streaming
     * 
     * This method intelligently manages animation continuation, restart, or direct display
     * based on the relationship between old and new text content.
     * 
     * - Parameter newText: The updated text content
     */
    private func handleTextChange(_ newText: String?) {
        guard let newText = newText else {
            displayedText = ""
            stopAnimation()
            return
        }
        
        if isAssistantMessage && isStreamingMessage && shouldUseTypewriter {
            // Check if new text is an extension of current displayed text
            if newText.hasPrefix(displayedText) && newText != displayedText {
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
    
    /**
     * Initiates typewriter animation for the given text
     * 
     * - Parameter text: The text to animate
     */
    private func startTypewriterAnimation(for text: String) {
        displayedText = ""
        continueTypewriterAnimation(with: text)
    }
    
    /**
     * Continues or resumes typewriter animation
     * 
     * This method sets up a timer-based animation that progressively reveals
     * characters at the configured typing speed.
     * 
     * - Parameter text: The complete text to animate
     */
    private func continueTypewriterAnimation(with text: String) {
        guard displayedText.count < text.count else { return }
        
        stopAnimation()
        
        animationTimer = Timer.scheduledTimer(withTimeInterval: typingSpeed, repeats: true) { timer in
            DispatchQueue.main.async {
                self.appendNextCharacters(from: text)
            }
        }
    }
    
    /**
     * Restarts typewriter animation with new content
     * 
     * - Parameter text: The new text to animate
     */
    private func restartTypewriterAnimation(with text: String) {
        stopAnimation()
        displayedText = ""
        startTypewriterAnimation(for: text)
    }
    
    /**
     * Appends the next character(s) to the displayed text
     * 
     * This method is called by the animation timer to progressively reveal
     * text characters. It handles proper string indexing and animation completion.
     * 
     * - Parameter text: The source text to extract characters from
     */
    private func appendNextCharacters(from text: String) {
        let currentLength = displayedText.count
        guard currentLength < text.count else {
            stopAnimation()
            return
        }
        
        let endIndex = min(currentLength + chunkSize, text.count)
        let startIndex = text.index(text.startIndex, offsetBy: currentLength)
        let targetIndex = text.index(text.startIndex, offsetBy: endIndex)
        
        let newChars = text[startIndex..<targetIndex]
        displayedText.append(String(newChars))
        
        if displayedText.count >= text.count {
            stopAnimation()
        }
    }
    
    /**
     * Stops and cleans up the typewriter animation
     * 
     * This method should be called when animation is no longer needed
     * to prevent memory leaks and unnecessary timer execution.
     */
    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }
}

// MARK: - Preview Provider
struct LLMMessageTextView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            LLMMessageTextView(
                text: "This is a typewriter animation demo text. Hello, this demonstrates the streaming effect!",
                messageUseMarkdown: false,
                messageId: "test1",
                isAssistantMessage: true,
                isStreamingMessage: true
            )
            
            LLMMessageTextView(
                text: "**Bold text** and *italic text* with markdown support.",
                messageUseMarkdown: true,
                messageId: "test2",
                isAssistantMessage: true,
                isStreamingMessage: true
            )
            
            LLMMessageTextView(
                text: "Regular user message without animation.",
                messageUseMarkdown: false,
                messageId: "test3",
                isAssistantMessage: false,
                isStreamingMessage: false
            )
        }
        .padding()
    }
}