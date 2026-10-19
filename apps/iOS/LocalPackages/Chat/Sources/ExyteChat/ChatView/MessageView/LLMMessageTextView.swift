//
//  LLMMessageTextView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/7.
//

import MarkdownUI
import Splash
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
    @Environment(\.colorScheme) private var colorScheme

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
    let plainTextFont: SwiftUI.Font
    let plainTextColor: SwiftUI.Color

    /// Revealed text plus its frozen-block split, mutated in place so a tick
    /// costs O(revealed chunk) rather than O(answer). `revision` drives redraws.
    @State private var buffer = StreamingTextBuffer()
    @State private var revision: UInt64 = 0
    private var displayedText: String { buffer.revealed }
    /// Direct completion signal from the inference pipeline. This remains
    /// reliable even when the environment provider does not invalidate a cell.
    @State private var hasReceivedOutputCompletion = false
    /// Prevent duplicate completion notifications across SwiftUI redraws.
    @State private var didNotifyAnimationCompletion = false

    /// Survives cell recreation so the typewriter resumes instead of replaying (fixes head-flicker).
    /// Values are UTF-8 offsets into the message text.
    private static var displayedCountCache: [String: Int] = [:]
    /// Timer for controlling the typewriter animation
    @State private var animationTimer: Timer?
    /// Updated independently of the running timer; new tokens must not postpone its next tick.
    @State private var animationTarget: String = ""

    /// Time interval between each character display (in seconds)
    private let typingSpeed: TimeInterval = 0.015

    /// Completed paragraphs are frozen. Only the live tail needs
    /// parsing on each tick, so total answer length must not slow the cadence.
    /// A very long unbroken paragraph still uses a bounded refresh rate.
    private func tickInterval() -> TimeInterval {
        // UTF-8 bytes, roughly 2 per character for mixed CJK/Latin answers.
        switch buffer.tailUTF8Count {
        case ..<4000:
            return typingSpeed
        case 4000 ..< 12000:
            return 0.08
        default:
            return 0.16
        }
    }

    /// Keep enough headroom to drain incoming tokens without a completion jump.
    private func chunkSize(for remainingCount: Int, interval: TimeInterval) -> Int {
        if interval <= typingSpeed {
            return adaptiveChunkSize(for: remainingCount)
        }
        return max(adaptiveChunkSize(for: remainingCount), Int((interval * 300).rounded(.up)))
    }
    private var outputIsComplete: Bool {
        isOutputComplete || hasReceivedOutputCompletion
    }

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
         onAnimationComplete: (() -> Void)? = nil,
         plainTextFont: SwiftUI.Font = .body,
         plainTextColor: SwiftUI.Color = .black)
    {
        self.text = text
        self.messageUseMarkdown = messageUseMarkdown
        self.messageId = messageId
        self.isAssistantMessage = isAssistantMessage
        self.isStreamingMessage = isStreamingMessage
        self.isOutputComplete = isOutputComplete
        self.onAnimationComplete = onAnimationComplete
        self.plainTextFont = plainTextFont
        self.plainTextColor = plainTextColor
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
            if isOutputComplete { hasReceivedOutputCompletion = true }
            if let text = text, isAssistantMessage && isStreamingMessage {
                if displayedText.isEmpty,
                   let cached = Self.displayedCountCache[messageId], cached > 0 {
                    let utf8 = text.utf8
                    let end = utf8.index(utf8.startIndex, offsetBy: min(cached, utf8.count))
                    setDisplayedText(String(text[..<end]))
                }
                if outputIsComplete, isFullyRevealed(text) {
                    notifyAnimationCompleted()
                } else if displayedText.isEmpty {
                    startTypewriterAnimation(for: text)
                } else {
                    continueTypewriterAnimation(with: text)
                }
            } else if let text = text {
                setDisplayedText(text)
            }
        }
        .onDisappear {
            stopAnimation()
            // A Markdown code block can rebuild the surrounding cell. Preserve
            // progress so the replacement view resumes instead of jumping to the
            // complete response or replaying from the beginning.
            Self.displayedCountCache[messageId] = outputIsComplete ? nil : buffer.revealedUTF8Count
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
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("StreamingOutputCompleted"))) {
            notification in
            guard notification.userInfo?["messageId"] as? String == messageId else { return }
            hasReceivedOutputCompletion = true
            handleStreamingStateChange()
        }
    }

    /// Renders text with typewriter animation effect
    ///
    /// Streaming markdown is rendered incrementally: the revealed text is
    /// split at paragraph boundaries (blank lines outside code fences) into
    /// frozen blocks plus a live tail. Frozen blocks are `Equatable` views
    /// SwiftUI skips re-rendering, so each tick costs O(tail) instead of
    /// O(document) — total streaming cost becomes O(n) rather than O(n²),
    /// which is what made long generations look slower and slower.
    /// - Returns: A view displaying the animated text with optional Markdown support
    @ViewBuilder
    private func typewriterView() -> some View {
        // Reading `revision` ties this body to buffer mutations.
        let _ = revision
        let split = buffer.split(dropsBlankBlocks: messageUseMarkdown)
        if messageUseMarkdown && isAssistantMessage {
            if split.blocks.isEmpty {
                markdownView(split.tail)
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    // Frozen strings are the same instances every tick, so the
                    // Equatable check is a pointer comparison, not a text scan.
                    ForEach(split.blocks.indices, id: \.self) { index in
                        FrozenMarkdownBlock(content: split.blocks[index], colorScheme: colorScheme)
                            .equatable()
                    }
                    if !split.tail.isEmpty {
                        markdownView(split.tail)
                    }
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(split.plainBlocks.indices, id: \.self) { index in
                    FrozenPlainTextBlock(content: split.plainBlocks[index], font: plainTextFont, color: plainTextColor)
                        .equatable()
                }
                if !split.tail.isEmpty {
                    Text(split.tail).font(plainTextFont).foregroundColor(plainTextColor)
                }
            }
        }
    }

    private func setDisplayedText(_ text: String) {
        buffer.reset(to: text)
        revision &+= 1
    }

    private func isFullyRevealed(_ text: String) -> Bool {
        buffer.revealedUTF8Count >= text.utf8.count
    }

    /// Whether `newText` continues what is already revealed. Streams only
    /// append, so comparing the bytes just before the reveal point is enough
    /// to tell a continuation from a replaced message without an O(n) scan.
    private func extendsRevealedText(_ newText: String) -> Bool {
        let revealedCount = buffer.revealedUTF8Count
        let utf8 = newText.utf8
        guard utf8.count >= revealedCount else { return false }
        let window = min(revealedCount, 64)
        let end = utf8.index(utf8.startIndex, offsetBy: revealedCount)
        let start = utf8.index(end, offsetBy: -window)
        return utf8[start..<end].elementsEqual(buffer.revealed.utf8.suffix(window))
    }

    /// Renders static text without animation
    /// - Parameter text: The text to be displayed
    /// - Returns: A view displaying the complete text with optional Markdown support
    @ViewBuilder
    private func staticView(_ text: String) -> some View {
        if messageUseMarkdown && isAssistantMessage {
            markdownView(text)
        } else {
            Text(text)
                .font(plainTextFont)
                .foregroundColor(plainTextColor)
        }
    }

    @ViewBuilder
    private func markdownView(_ content: String) -> some View {
        Self.renderMarkdown(markdownForRendering(content), colorScheme: colorScheme)
    }

    /// Shared markdown construction for the live tail and frozen blocks.
    @ViewBuilder
    fileprivate static func renderMarkdown(_ content: String, colorScheme: ColorScheme) -> some View {
        // MarkdownUI is a native SwiftUI view and participates directly in
        // every streaming state update. Keeping one renderer for partial and
        // completed content avoids stale UIKit parse/layout state across turns.
        // The block style must sit inside the theme: `.markdownTheme` replaces
        // the whole theme, so applied closer to the view it would discard the
        // custom code block (header with language and copy button).
        MarkdownUI.Markdown(content)
            .markdownBlockStyle(\.codeBlock) { configuration in
                codeBlock(configuration, colorScheme: colorScheme)
            }
            .markdownTheme(.gitHub)
            .markdownCodeSyntaxHighlighter(ChatSplashCodeSyntaxHighlighter(theme: codeTheme(for: colorScheme)))
        // No `.textSelection`: its long-press callout would compete with the
        // message menu, which offers "Select Text" instead.
    }

    @ViewBuilder
    private static func codeBlock(_ configuration: CodeBlockConfiguration, colorScheme: ColorScheme) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text(configuration.language ?? "code")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
                Spacer(minLength: 8)
                CodeBlockCopyButton(code: configuration.content)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            ScrollView(.horizontal) {
                configuration.label
                    .relativeLineSpacing(.em(0.2))
                    .markdownTextStyle {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.85))
                    }
                    .padding(12)
            }
        }
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .markdownMargin(top: 0, bottom: 12)
    }

    private var codeTheme: Splash.Theme {
        Self.codeTheme(for: colorScheme)
    }

    private static func codeTheme(for colorScheme: ColorScheme) -> Splash.Theme {
        switch colorScheme {
        case .dark:
            return .wwdc17(withFont: .init(size: 14))
        default:
            return .sunset(withFont: .init(size: 14))
        }
    }

    /// Completes an open code fence only in the transient rendering copy. This
    /// keeps partial code visible while tokens stream and also handles output
    /// truncated by the token limit without altering the stored model text.
    private func markdownForRendering(_ content: String) -> String {
        guard let fence = unclosedCodeFence(in: content) else { return content }

        let separator = content.hasSuffix("\n") ? "" : "\n"
        return content + separator + String(repeating: String(fence.marker), count: fence.length)
    }

    private func unclosedCodeFence(in content: String) -> (marker: Character, length: Int)? {
        var openingMarker: Character?
        var openingLength = 0

        for line in content.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmedLine = line.drop(while: { $0 == " " || $0 == "\t" })
            guard let marker = trimmedLine.first, marker == "`" || marker == "~" else { continue }

            let markerLength = trimmedLine.prefix(while: { $0 == marker }).count
            guard markerLength >= 3 else { continue }

            if openingMarker == nil {
                openingMarker = marker
                openingLength = markerLength
            } else if openingMarker == marker, markerLength >= openingLength {
                let suffix = String(trimmedLine.dropFirst(markerLength))
                if suffix.trimmingCharacters(in: .whitespaces).isEmpty {
                    openingMarker = nil
                    openingLength = 0
                }
            }
        }

        guard let openingMarker else { return nil }
        return (openingMarker, openingLength)
    }

    /// Handles streaming state changes.
    ///
    /// Called when streaming state changes to manage animation start, stop, and notification sending.
    private func handleStreamingStateChange() {
        // The persistent timer captures the original View value. Keep the
        // completion signal in State so that timer observes later completion.
        if isOutputComplete { hasReceivedOutputCompletion = true }
        if isStreamingMessage, isAssistantMessage {
            // Start or continue streaming output animation
            if let text = text {
                if outputIsComplete, isFullyRevealed(text) {
                    notifyAnimationCompleted()
                } else if displayedText.isEmpty {
                    startTypewriterAnimation(for: text)
                } else {
                    continueTypewriterAnimation(with: text)
                }
            }
        } else if outputIsComplete {
            // Output complete, display full text and send notification
            if let text = text {
                setDisplayedText(text)
            }
            stopAnimation()
            notifyAnimationCompleted()
        } else {
            // Non-streaming or fully complete, display static text
            if let text = text {
                setDisplayedText(text)
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
            setDisplayedText("")
            stopAnimation()
            return
        }

        if isAssistantMessage, isStreamingMessage {
            if extendsRevealedText(newText) {
                // Continue typewriter animation
                if !isFullyRevealed(newText) { continueTypewriterAnimation(with: newText) }
            } else {
                // Restart animation with new content
                restartTypewriterAnimation(with: newText)
            }
        } else {
            // Display text directly without animation
            setDisplayedText(newText)
            stopAnimation()
        }
    }

    /// Initiates typewriter animation for the given text
    /// - Parameter text: The text to animate
    private func startTypewriterAnimation(for text: String) {
        setDisplayedText("")
        continueTypewriterAnimation(with: text)
    }

    /// Continues or resumes typewriter animation
    ///
    /// This method sets up a timer-based animation that progressively reveals
    /// characters at the configured typing speed.
    ///
    /// - Parameter text: The complete text to animate
    private func continueTypewriterAnimation(with text: String) {
        animationTarget = text
        guard !isFullyRevealed(animationTarget) else { return }
        // Preserve the next scheduled tick when another input batch arrives.
        // Restarting here can starve the timer when input arrives every 80ms
        // but long-text animation ticks every 80/160ms.
        guard animationTimer?.isValid != true else { return }

        let timer = Timer(timeInterval: tickInterval(), repeats: true) { _ in
            self.appendNextCharacters(from: self.animationTarget)
        }
        animationTimer = timer
        // Keep revealing during scrolling, too. Timer already runs on main.
        RunLoop.main.add(timer, forMode: .common)
    }

    /// Restarts typewriter animation with new content
    /// - Parameter text: The new text to animate
    private func restartTypewriterAnimation(with text: String) {
        stopAnimation()
        startTypewriterAnimation(for: text)
    }

    /// Appends the next character(s) to the displayed text
    ///
    /// This method is called by the animation timer to progressively reveal
    /// text characters. It handles proper string indexing and animation completion.
    ///
    /// - Parameter text: The source text to extract characters from
    private func appendNextCharacters(from text: String) {
        guard !isFullyRevealed(text) else {
            stopAnimation()
            return
        }

        // Reschedule when the live tail crosses a cadence boundary, including
        // returning to the fast cadence after a paragraph is frozen.
        let desiredInterval = tickInterval()
        if let timer = animationTimer, abs(timer.timeInterval - desiredInterval) > .ulpOfOne {
            stopAnimation()
            continueTypewriterAnimation(with: text)
            return
        }

        // The revealed text is a byte prefix of `text`, so the reveal point is
        // an O(1) UTF-8 offset; only the not-yet-shown backlog is walked.
        let utf8 = text.utf8
        let startIndex = utf8.index(utf8.startIndex, offsetBy: buffer.revealedUTF8Count)
        let remainingCount = text[startIndex...].count
        let interval = animationTimer?.timeInterval ?? typingSpeed
        let step = min(chunkSize(for: remainingCount, interval: interval), remainingCount)
        let targetIndex = text.index(startIndex, offsetBy: step)

        buffer.append(text[startIndex ..< targetIndex])
        revision &+= 1
        Self.displayedCountCache[messageId] = buffer.revealedUTF8Count

        if isFullyRevealed(text) {
            stopAnimation()
            if outputIsComplete {
                notifyAnimationCompleted()
            }
        }
    }

    /// Keep the animation visibly incremental while preventing a fast model
    /// from building a multi-second UI backlog that jumps at completion.
    private func adaptiveChunkSize(for remainingCount: Int) -> Int {
        switch remainingCount {
        case 81...:
            return 8
        case 41...:
            return 4
        case 17...:
            return 2
        default:
            return 1
        }
    }

    private func notifyAnimationCompleted() {
        guard outputIsComplete, !didNotifyAnimationCompletion else { return }
        didNotifyAnimationCompletion = true
        Self.displayedCountCache[messageId] = nil
        stopAnimation()
        NotificationCenter.default.post(
            name: NSNotification.Name("StreamingAnimationCompleted"),
            object: nil,
            userInfo: ["messageId": messageId]
        )
        onAnimationComplete?()
    }

    /// Stops and cleans up the typewriter animation
    ///
    /// This method should be called when animation is no longer needed
    /// to prevent memory leaks and unnecessary timer execution.
    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }
}

/// A completed markdown block in the streaming view. Equatable so SwiftUI
/// skips re-rendering it when its text is unchanged — old paragraphs cost
/// O(1) per streaming tick regardless of total answer length.
@available(iOS 17.0, *)
private struct FrozenMarkdownBlock: View, Equatable {
    let content: String
    let colorScheme: ColorScheme

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.content == rhs.content && lhs.colorScheme == rhs.colorScheme
    }

    var body: some View {
        LLMMessageTextView.renderMarkdown(content, colorScheme: colorScheme)
    }
}

/// Keep completed thinking paragraphs out of subsequent character updates.
private struct FrozenPlainTextBlock: View, Equatable {
    let content: String
    let font: SwiftUI.Font
    let color: SwiftUI.Color

    var body: some View {
        Text(content).font(font).foregroundColor(color)
    }
}

private struct ChatSplashCodeSyntaxHighlighter: CodeSyntaxHighlighter {
    private let syntaxHighlighter: SyntaxHighlighter<ChatTextOutputFormat>

    init(theme: Splash.Theme) {
        syntaxHighlighter = SyntaxHighlighter(format: ChatTextOutputFormat(theme: theme))
    }

    func highlightCode(_ code: String, language: String?) -> Text {
        guard language != nil else { return Text(code) }
        return syntaxHighlighter.highlight(code)
    }
}

private struct ChatTextOutputFormat: OutputFormat {
    let theme: Splash.Theme

    func makeBuilder() -> Builder {
        Builder(theme: theme)
    }

    struct Builder: OutputBuilder {
        let theme: Splash.Theme
        private var fragments: [Text] = []

        init(theme: Splash.Theme) {
            self.theme = theme
        }

        mutating func addToken(_ token: String, ofType type: TokenType) {
            let color = theme.tokenColors[type] ?? theme.plainTextColor
            fragments.append(Text(token).foregroundColor(Color(uiColor: color)))
        }

        mutating func addPlainText(_ text: String) {
            fragments.append(Text(text).foregroundColor(Color(uiColor: theme.plainTextColor)))
        }

        mutating func addWhitespace(_ whitespace: String) {
            fragments.append(Text(whitespace))
        }

        func build() -> Text {
            fragments.reduce(Text(""), +)
        }
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

/// Copies one code block; briefly shows a checkmark as confirmation.
private struct CodeBlockCopyButton: View {
    let code: String
    @State private var copied = false

    var body: some View {
        Button {
            UIPasteboard.general.string = code
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            copied = true
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                copied = false
            }
        } label: {
            Label(ChatTextLocalization.text(copied ? "chat.code.copied" : "chat.code.copy"),
                  systemImage: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.secondary)
        }
        .buttonStyle(.borderless)
    }
}

/// Revealed streaming text and its split into frozen blocks plus a live tail.
///
/// A block boundary is a blank line outside any code fence, taken only once
/// more content follows it, so a frozen block never changes (the stream is
/// append-only). Paragraphs, headings, lists and tables are blank-line
/// delimited already; an open code fence pins everything after it into the
/// tail. Scanning resumes where it stopped, so the total cost over an answer
/// is O(n) instead of O(n) per tick.
final class StreamingTextBuffer {
    struct Split {
        let blocks: [String]
        /// Blocks without their trailing newline, for the plain-text layout
        /// where the next Text already starts on a new line.
        let plainBlocks: [String]
        let tail: String
    }

    private(set) var revealed = ""
    private var blocks: [String] = []
    private var plainBlocks: [String] = []
    /// UTF-8 offsets: start of the live tail, and first line not yet scanned.
    private var tailStart = 0
    private var scanOffset = 0
    private var fenceMarker: UInt8?
    private var fenceLength = 0
    private var dropsBlankBlocks = true
    private var cachedSplit: (count: Int, split: Split)?

    var revealedUTF8Count: Int { revealed.utf8.count }
    var tailUTF8Count: Int { revealed.utf8.count - tailStart }

    func reset(to text: String) {
        revealed = text
        blocks = []
        plainBlocks = []
        tailStart = 0
        scanOffset = 0
        fenceMarker = nil
        fenceLength = 0
        cachedSplit = nil
    }

    func append(_ text: Substring) {
        revealed.append(contentsOf: text)
        scan()
    }

    func split(dropsBlankBlocks: Bool) -> Split {
        if dropsBlankBlocks != self.dropsBlankBlocks {
            self.dropsBlankBlocks = dropsBlankBlocks
            let text = revealed
            reset(to: text)
        }
        scan()
        let count = revealed.utf8.count
        if let cachedSplit, cachedSplit.count == count { return cachedSplit.split }
        let utf8 = revealed.utf8
        let tail = String(revealed[utf8.index(utf8.startIndex, offsetBy: tailStart)...])
        let split = Split(blocks: blocks, plainBlocks: plainBlocks, tail: tail)
        cachedSplit = (count, split)
        return split
    }

    private static func isInlineSpace(_ byte: UInt8) -> Bool { byte == 0x20 || byte == 0x09 }
    private static func isSpace(_ byte: UInt8) -> Bool { isInlineSpace(byte) || byte == 0x0A || byte == 0x0D }

    private func scan() {
        let utf8 = revealed.utf8
        let count = utf8.count
        var lineStart = scanOffset
        while lineStart < count {
            let lineStartIndex = utf8.index(utf8.startIndex, offsetBy: lineStart)
            // An unterminated last line can still grow; scan it next time.
            guard let newline = utf8[lineStartIndex...].firstIndex(of: 0x0A) else { break }
            let nextLineStart = utf8.distance(from: utf8.startIndex, to: newline) + 1
            let content = utf8[lineStartIndex..<newline].drop(while: Self.isInlineSpace)

            if let marker = content.first, marker == UInt8(ascii: "`") || marker == UInt8(ascii: "~") {
                let length = content.prefix(while: { $0 == marker }).count
                if length >= 3 {
                    if fenceMarker == nil {
                        fenceMarker = marker
                        fenceLength = length
                    } else if fenceMarker == marker, length >= fenceLength,
                              content.dropFirst(length).allSatisfy(Self.isInlineSpace) {
                        fenceMarker = nil
                    }
                }
            }

            if fenceMarker == nil, content.isEmpty {
                // Freeze only when content continues past the blank line;
                // otherwise revisit this line once more text arrives.
                guard nextLineStart < count else { break }
                let blockStart = utf8.index(utf8.startIndex, offsetBy: tailStart)
                let blockEnd = utf8.index(after: newline)
                if utf8[blockStart..<blockEnd].contains(where: { !Self.isSpace($0) }) {
                    let block = String(revealed[blockStart..<blockEnd])
                    blocks.append(block)
                    plainBlocks.append(String(block.dropLast()))
                    tailStart = nextLineStart
                } else if dropsBlankBlocks {
                    tailStart = nextLineStart
                }
            }
            lineStart = nextLineStart
        }
        scanOffset = lineStart
    }
}
