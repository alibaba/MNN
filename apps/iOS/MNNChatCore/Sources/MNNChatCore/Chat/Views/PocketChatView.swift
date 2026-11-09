//
//  LLMChatView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/9/29.
//  Restructured for the single-model MNN Pocket app (2026/09).
//

import AVFoundation
import ExyteChat
import ExyteMediaPicker
import Foundation
import SwiftUI

/// LLMChatView — the single screen of the MNN Pocket demo app.
///
/// Layout follows the Edge0-style reference: a dark welcome screen with the
/// product branding and a stack of candidate prompts while the conversation
/// is empty, then a plain dark chat with Prefill / Decode / Peak Memory
/// readouts attached to each answer. The built-in ExyteChat input bar is
/// themed dark and reused so keyboard avoidance and list insets keep working.
struct PocketChatView: View {
    // MARK: - State Properties

    /// View model for LLM chat functionality
    @StateObject private var viewModel: LLMChatViewModel

    /// Explicit route callback used by MainTabView to return to the model list.
    private let onBack: (() -> Void)?

    /// When true the first welcome prompt is sent automatically once the
    /// model is loaded (relaunch-race reproduction harness).
    private let autoSendPreset: Bool

    /// Audio recorder settings configuration
    private let recorderSettings = RecorderSettings(audioFormatID: kAudioFormatLinearPCM,
                                                    sampleRate: 44100, numberOfChannels: 2,
                                                    linearPCMBitDepth: 16)

    // MARK: - Properties

    /// Title of the chat interface
    private let title: String

    /// Path to the model file
    private let modelPath: String

    private var supportsAudioInput: Bool {
        if ModelUtils.isAudioModel(viewModel.modelInfo.name) {
            return true
        }
        return viewModel.modelInfo.tags.contains { $0.localizedCaseInsensitiveContains("audio") }
    }

    private var supportsVisualInput: Bool {
        viewModel.supportsImageInput
    }

    private var supportsVideoInput: Bool {
        guard supportsVisualInput else { return false }
        let nameContainsVideo = viewModel.modelInfo.name.localizedCaseInsensitiveContains("video")
        let tagsContainVideo = viewModel.modelInfo.tags.contains { $0.localizedCaseInsensitiveContains("video") }
        let categoriesContainVideo = (viewModel.modelInfo.categories ?? []).contains { category in
            category.localizedCaseInsensitiveContains("video")
        }
        return nameContainsVideo || tagsContainVideo || categoriesContainVideo || ModelUtils.isOmni(viewModel.modelInfo.name)
    }

    private var resolvedAvailableInput: AvailableInputType {
        if supportsAudioInput && supportsVisualInput {
            return .full
        } else if supportsAudioInput {
            return .textAndAudio
        } else if supportsVisualInput {
            return .textAndMedia
        } else {
            return .textOnly
        }
    }

    // MARK: - Initialization

    /// Initializes the chat view with model information and optional history
    /// - Parameters:
    ///   - modelInfo: Information about the model to use
    ///   - history: Optional chat history to restore
    ///   - onBack: Explicit back-route callback (used by MainTabView)
    ///   - autoSendPreset: Auto-send the first welcome prompt once loaded
    init(modelInfo: ModelInfo, history: ChatHistory? = nil, onBack: (() -> Void)? = nil, autoSendPreset: Bool = false) {
        title = modelInfo.name
        modelPath = modelInfo.localPath
        self.onBack = onBack
        self.autoSendPreset = autoSendPreset
        let viewModel = LLMChatViewModel(modelInfo: modelInfo, history: history)
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            // Model identity stays visible for the whole session, including
            // inside a conversation where the welcome screen's big title is
            // gone. Also carries the route back to the model picker.
            chatHeader

            // Demo layout: the current prompt pinned at the top of the
            // screen (sticky), the streaming answer scrolling beneath it.
            // Sending a new prompt clears the screen and repeats the cycle.
            if let question = currentQuestionText {
                StickyQuestionBar(text: question)
            } else {
                PocketWelcomeView(
                    modelName: viewModel.modelInfo.name,
                    isSpeculative: viewModel.modelInfo.isSpeculativeDecoding
                )
            }

            // The answer area: streaming markdown, scrollable. Hidden on the
            // welcome screen so no placeholder shows before the first prompt.
            if currentQuestionText != nil {
                AnswerScrollView(viewModel: viewModel)
            } else {
                Color.clear
            }

        }
        // The preset bar and the prompt input ride as a bottom safe-area
        // attachment rather than as VStack members: that way the answer
        // ScrollView gets its content inset managed by the system (the answer
        // can scroll clear of the input, and nothing is covered), and a
        // presented keyboard lifts only this attachment instead of squeezing
        // the answer area and clipping the sticky question bar.
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 0) {
                // Preset bar pinned above the input area: visible on the
                // welcome screen AND inside conversations, so candidate
                // prompts stay tappable at any point in the chat. A leading
                // new-chat icon returns to the welcome screen.
                PocketPresetPromptBar(
                    presets: PresetPrompts.welcomePrompts(for: viewModel.modelInfo),
                    compact: currentQuestionText != nil,
                    showNewChat: currentQuestionText != nil,
                    onNewChat: {
                        viewModel.resetConversation()
                    }
                ) { preset in
                    viewModel.sendPreset(preset)
                }
                .disabled(viewModel.chatInputUnavilable)

                PromptInputBar(viewModel: viewModel)
            }
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            viewModel.onStart()
            autoSendFirstPresetIfRequested()
        }
        .onDisappear(perform: viewModel.onStop)
        .onReceive(NotificationCenter.default.publisher(for: .dismissKeyboard)) { _ in
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }

    /// The currently displayed question (last user message), or nil while on
    /// the welcome screen.
    private var currentQuestionText: String? {
        viewModel.messages.last { $0.user.isCurrentUser }?.text
    }

    private var chatContent: some View {
        ChatView(
            messages: viewModel.messages,
            chatType: .conversation,
            didSendMessage: { draft in
                viewModel.sendToLLM(draft: draft)
            }
        )
        .setStreamingMessageProvider(viewModel)
        .setDefaultInputText($viewModel.defaultInputText)
        .setLocalization(ChatLocalization(
            inputPlaceholder: AppText.text("Ask anything", comment: "Pocket input placeholder"),
            signatureText: "",
            cancelButtonText: AppText.text("Cancel", comment: ""),
            recentToggleText: "",
            waitingForNetwork: "",
            recordingText: "",
            replyToText: ""
        ))
        .setAvailableInput(resolvedAvailableInput)
        .messageUseMarkdown(true)
        .useDirectMessageCopyGesture(true)
        .showDateHeaders(false)
        .showMessageTimeView(false)
        .setRecorderSettings(recorderSettings)
        .setThinkingMode(
            supportsThinkingMode: viewModel.supportsThinkingMode,
            isEnabled: viewModel.isThinkingModeEnabled,
            onToggle: {
                viewModel.toggleThinkingMode()
            }
        )
        .chatTheme(
            ChatTheme(
                colors: .init(
                    mainBG: Palette.background,
                    mainTint: Palette.caption,
                    mainText: Palette.primaryText,
                    mainCaptionText: Palette.caption,
                    messageMyBG: Palette.userBubble,
                    messageMyText: Palette.primaryText,
                    messageFriendBG: .clear,
                    messageFriendText: Palette.primaryText,
                    inputBG: Palette.surface,
                    inputText: Palette.primaryText,
                    inputPlaceholderText: Palette.caption,
                    menuBG: Palette.surface,
                    menuText: Palette.primaryText,
                    sendButtonBackground: Palette.accent
                ),
                images: .init(
                    attach: Image(systemName: "photo"),
                    attachCamera: Image("attachCamera", bundle: .current)
                )
            )
        )
        .disabled(!viewModel.isModelLoaded)
    }

    // MARK: - Header

    /// Session header: back route to the model picker plus the model pill.
    private var chatHeader: some View {
        HStack(alignment: .top, spacing: 10) {
            if let onBack {
                Button {
                    onBack()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(Palette.primaryText)
                        .frame(width: 32, height: 32)
                        .background(Circle().fill(Palette.surface))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(AppText.text("Back to models"))
            }

            modelPill

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.top, 6)
        .padding(.bottom, 6)
        .background(Palette.background)
    }

    /// Pill-shaped model identifier with the live status caption underneath,
    /// mirroring the reference header ("Edge0-35B" + star badge).
    private var modelPill: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                if viewModel.isModelLoaded {
                    // The product mark rather than a generic star; clipped
                    // because the asset's white backdrop would otherwise show
                    // as a square on the pill's grey fill. Sized slightly up
                    // from the old star so the bolt badge stays legible.
                    Image("mnn_icon")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 18, height: 18)
                        .clipShape(Circle())
                        .overlay(alignment: .bottomTrailing) {
                            if viewModel.modelInfo.isSpeculativeDecoding {
                                Image(systemName: "bolt.fill")
                                    .font(.system(size: 7, weight: .black))
                                    .foregroundColor(.white)
                                    .frame(width: 12, height: 12)
                                    .background(Circle().fill(Palette.accent))
                                    .overlay(Circle().strokeBorder(Palette.surface, lineWidth: 1))
                                    .offset(x: 2, y: 2)
                            }
                        }
                } else {
                    ProgressView()
                        .controlSize(.mini)
                }
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Palette.primaryText)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Capsule().fill(Palette.surface))

            Text(viewModel.chatStatus)
                .font(.system(size: 11))
                .foregroundColor(Palette.caption)
                .padding(.leading, 4)
        }
        .padding(.leading, 4)
    }

    // MARK: - Private Methods

    /// Sends the first welcome prompt automatically (relaunch-race harness).
    /// Waits in short intervals until the model is loaded, then taps the
    /// first preset, mirroring a manual user interaction.
    ///
    /// `--auto-preset-delay <seconds>` waits after the model is loaded before
    /// sending, reproducing a manual user's tap timing. Without it the prompt
    /// fires immediately after load.
    private func autoSendFirstPresetIfRequested() {
        guard autoSendPreset else { return }
        let delayArgIndex = CommandLine.arguments.firstIndex(of: "--auto-preset-delay")
        let delaySeconds: UInt32 = {
            guard let idx = delayArgIndex, idx + 1 < CommandLine.arguments.count,
                  let seconds = UInt32(CommandLine.arguments[idx + 1]) else { return 0 }
            return seconds
        }()
        Task { @MainActor in
            for _ in 0..<240 { // up to ~120s
                if viewModel.isModelLoaded { break }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            if delaySeconds > 0 {
                print("[AutoPreset] holding prompt for \(delaySeconds)s after model load")
                sleep(delaySeconds)
            }
            // Also wait out the hidden warmup inference if one is in flight,
            // so the harness prompt is not swallowed by the in-progress flag.
            for _ in 0..<60 {
                if viewModel.isModelLoaded, !viewModel.isProcessing { break }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            guard viewModel.isModelLoaded,
                  let preset = PresetPrompts.welcomePrompts(for: viewModel.modelInfo).first,
                  !viewModel.chatInputUnavilable else { return }
            print("[AutoPreset] sending first preset automatically")
            viewModel.sendPreset(preset)
        }
    }

    /// Opaque dark navigation bar matching the chat background.
    private func setupNavigationBarAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(Palette.background)
        appearance.shadowColor = .clear

        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
}

// MARK: - Light Pocket Palette

/// Light Pocket palette (white background, dark text). Shared with the model
/// picker in MNNLLMiOSApp, hence not file-private.
enum PocketPalette {
    static let background = Color(hex: "FFFFFF")
    static let surface = Color(hex: "F2F3F5")
    static let surfaceElevated = Color(hex: "E9EBEE")
    static let assistantBubble = Color(hex: "F2F3F5")
    static let userBubble = Color(hex: "E3E6FB")
    static let primaryText = Color(hex: "1A1C20")
    static let caption = Color(hex: "8E939B")
    static let accent = Color(hex: "4859FD")
}

/// Keeps the existing `Palette.*` call sites in this file unchanged.
private typealias Palette = PocketPalette

// MARK: - Preset Prompt Bar

/// Horizontally scrollable candidate-prompt bar. Shown on the welcome screen
/// and kept visible inside conversations so presets stay tappable at any
/// point. `compact` tightens the padding once a conversation is running.
struct PocketPresetPromptBar: View {
    let presets: [PresetPrompt]
    let compact: Bool
    var showNewChat: Bool = false
    var onNewChat: (() -> Void)? = nil
    let onSelect: (PresetPrompt) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                if showNewChat {
                    Button {
                        onNewChat?()
                    } label: {
                        Image(systemName: "square.and.pencil")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundColor(Palette.accent)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(Circle().fill(Palette.surface))
                    }
                    .buttonStyle(.plain)
                }
                ForEach(Array(presets.enumerated()), id: \.offset) { _, preset in
                    Button {
                        onSelect(preset)
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: preset.icon)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(Palette.accent)

                            Text(preset.title)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(Palette.primaryText)
                                .lineLimit(1)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Capsule().fill(Palette.surface))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, compact ? 6 : 10)
        }
        .background(Palette.background.opacity(0.97))
    }
}

// MARK: - Prompt Input Bar

/// Bottom prompt input, so arbitrary prompts can be typed instead of only
/// tapping a preset. The send button turns into a stop button while the model
/// is generating, letting a long answer be interrupted.
struct PromptInputBar: View {
    @ObservedObject var viewModel: LLMChatViewModel

    @State private var text: String = ""
    @FocusState private var isFocused: Bool

    private var trimmed: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// `chatInputUnavilable` folds "not loaded" and "generating" into one flag,
    /// but the two need different affordances here, so both are read directly.
    private var canSend: Bool {
        viewModel.isModelLoaded && !viewModel.isProcessing && !trimmed.isEmpty
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // Vertical axis so a long prompt wraps; it scrolls internally past
            // five lines instead of pushing the answer area off screen.
            TextField(AppText.text("Ask anything"), text: $text, axis: .vertical)
                .lineLimit(1 ... 5)
                .font(.system(size: 15))
                .foregroundColor(Palette.primaryText)
                .tint(Palette.accent)
                .focused($isFocused)
                // Typing stays allowed while generating so the next prompt can
                // be composed; only sending is gated.
                .disabled(!viewModel.isModelLoaded)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(Capsule().fill(Palette.surface))

            Button {
                if viewModel.isProcessing {
                    viewModel.cancelGeneration()
                } else {
                    send()
                }
            } label: {
                Image(systemName: viewModel.isProcessing ? "stop.fill" : "arrow.up")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 34, height: 34)
                    .background(
                        Circle().fill(
                            viewModel.isProcessing || canSend
                                ? Palette.accent
                                : Palette.caption.opacity(0.4)
                        )
                    )
            }
            .buttonStyle(.plain)
            .disabled(!viewModel.isProcessing && !canSend)
            .accessibilityLabel(
                viewModel.isProcessing
                    ? AppText.text("Stop generating")
                    : AppText.text("Send")
            )
        }
        .padding(.horizontal, 12)
        .padding(.top, 6)
        .padding(.bottom, 8)
        .background(Palette.background)
    }

    /// A multi-line TextField maps Return to a newline, so there is no
    /// return-to-send; the button is the only send path.
    private func send() {
        guard canSend else { return }
        let prompt = trimmed
        text = ""
        // Drop focus explicitly instead of relying on the `.dismissKeyboard`
        // notification that `sendToLLM` posts: that path resigns first
        // responder in UIKit, which leaves this binding to catch up.
        isFocused = false
        viewModel.sendUserPrompt(prompt)
    }
}

// MARK: - Welcome Screen

/// Empty-state welcome screen: centered branding with tagline above the
/// preset bar, in the style of the reference app.
struct PocketWelcomeView: View {
    /// Shown as the big title, so switching models retitles the welcome screen.
    let modelName: String
    /// Adds the bolt badge, matching how the model picker marks this variant.
    let isSpeculative: Bool

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 20)

            VStack(spacing: 18) {
                // The asset's backdrop is white, matching this screen, so it
                // needs no clipping to sit flush.
                Image("mnn_icon")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 96, height: 96)
                    .overlay(alignment: .bottomTrailing) {
                        if isSpeculative {
                            Image(systemName: "bolt.fill")
                                .font(.system(size: 15, weight: .black))
                                .foregroundColor(.white)
                                .frame(width: 32, height: 32)
                                .background(Circle().fill(Palette.accent))
                                .overlay(Circle().strokeBorder(Palette.background, lineWidth: 2.5))
                                .offset(x: 2, y: -2)
                        }
                    }
                    .accessibilityHidden(true)

                VStack(spacing: 10) {
                    Text(modelName)
                        .font(.system(size: 32, weight: .bold, design: .rounded))
                        .foregroundColor(Palette.primaryText)
                        .multilineTextAlignment(.center)

                    Text("完全在本地运行，数据不出设备")
                        .font(.system(size: 14))
                        .foregroundColor(Palette.caption)
                        .multilineTextAlignment(.center)
                }
            }
            .padding(.horizontal, 24)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Performance Metrics

/// Throughput readout for the finished answer, with the numbers as the visual
/// anchor and the labels/units kept quiet.
///
/// The engine hands over one preformatted line per metric (that same string
/// also goes to NSLog), so the figures are pulled back out here instead of
/// changing the native format. Elapsed seconds and token counts are dropped —
/// only the rates carry information at a glance.
struct PerfMetricsView: View {
    let raw: String

    private struct Metric: Identifiable {
        let id: String
        let label: String
        let value: String
        let unit: String
    }

    private var metrics: [Metric] {
        var found: [Metric] = []
        if let v = Self.firstCapture(in: raw, pattern: #"Prefill:.*?([0-9.]+)\s*tokens/s"#) {
            found.append(Metric(id: "prefill", label: "Prefill", value: v, unit: "tokens/s"))
        }
        if let v = Self.firstCapture(in: raw, pattern: #"Decode:.*?([0-9.]+)\s*tokens/s"#) {
            found.append(Metric(id: "decode", label: "Decode", value: v, unit: "tokens/s"))
        }
        if let v = Self.firstCapture(in: raw, pattern: #"Peak Memory:\s*([0-9.]+)\s*MB"#) {
            found.append(Metric(id: "memory", label: "Peak Memory", value: v, unit: "MB"))
        }
        return found
    }

    private static func firstCapture(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              match.numberOfRanges > 1,
              let range = Range(match.range(at: 1), in: text)
        else {
            return nil
        }
        return String(text[range])
    }

    var body: some View {
        let items = metrics
        if items.isEmpty {
            // Unrecognised format: show it verbatim rather than dropping data.
            Text(raw)
                .font(.system(size: 12))
                .foregroundColor(Palette.caption)
                .multilineTextAlignment(.leading)
        } else {
            // Falls back to a stacked layout on narrower screens, where three
            // metrics in a row would not fit.
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 18) {
                    ForEach(items) { metric(for: $0) }
                }
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(items) { metric(for: $0) }
                }
            }
        }
    }

    private func metric(for item: Metric) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(item.label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(Palette.caption)
                .lineLimit(1)

            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(item.value)
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundColor(Palette.accent)
                Text(item.unit)
                    .font(.system(size: 10))
                    .foregroundColor(Palette.caption)
            }
            .lineLimit(1)
        }
    }
}

// MARK: - Sticky Question Bar

/// The current prompt pinned at the top-left of the answer screen. Stays in
/// place while the answer streams and scrolls beneath it.
struct StickyQuestionBar: View {
    let text: String

    /// Head and tail of a long prompt with the middle elided, so this stays a
    /// header instead of taking over the screen (the 517-token summarization
    /// preset otherwise fills it entirely).
    ///
    /// Done by hand rather than with `.truncationMode(.middle)`: across more
    /// than one line UIKit only truncates the last line, which does not keep
    /// the ending visible.
    private var displayText: String {
        let normalized = Self.normalized(text)
        // Sized so a long prompt fills roughly a third of the screen: ~52 chars
        // per line at 15pt on a 440pt-wide device, so ~730 chars ≈ 14 lines.
        let headLimit = 540
        let tailLimit = 190
        guard normalized.count > headLimit + tailLimit + 8 else { return normalized }
        // The ellipsis gets its own paragraph so the elision reads as a break
        // rather than as part of either sentence.
        return Self.headSlice(normalized, limit: headLimit)
            + "\n\n…\n\n"
            + Self.tailSlice(normalized, limit: tailLimit)
    }

    /// Collapses soft wraps into spaces while keeping blank-line paragraph
    /// breaks, which markdown needs in order to render structure. Without the
    /// collapse a multi-line prompt grows the bar even with the length capped;
    /// without keeping the blank lines it would render as one runaway block.
    private static func normalized(_ s: String) -> String {
        let unified = s.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let marked = unified.replacingOccurrences(
            of: "\n{2,}", with: "\u{2}", options: .regularExpression
        )
        let singleLine = marked.replacingOccurrences(of: "\n", with: " ")
        let squeezed = singleLine.replacingOccurrences(
            of: "[ \t]+", with: " ", options: .regularExpression
        )
        return squeezed
            .replacingOccurrences(of: "\u{2}", with: "\n\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Slack allowed when backing off to a word boundary. Beyond this the text
    /// is space-free (CJK, URLs) and a hard cut reads better than dropping a
    /// large chunk.
    private static let boundarySlack = 20

    private static func headSlice(_ s: String, limit: Int) -> String {
        let cut = s.prefix(limit)
        // Newlines count as boundaries too, so a cut never lands mid-paragraph-break.
        guard let lastBreak = cut.lastIndex(where: { $0 == " " || $0 == "\n" }),
              cut.distance(from: lastBreak, to: cut.endIndex) <= boundarySlack
        else {
            return String(cut)
        }
        return String(cut[..<lastBreak])
    }

    private static func tailSlice(_ s: String, limit: Int) -> String {
        let cut = s.suffix(limit)
        // Drop the leading partial word (e.g. "maller" from "smaller").
        guard let firstBreak = cut.firstIndex(where: { $0 == " " || $0 == "\n" }),
              cut.distance(from: cut.startIndex, to: firstBreak) <= boundarySlack
        else {
            return String(cut)
        }
        return String(cut[cut.index(after: firstBreak)...])
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "quote.opening")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Palette.accent)
                .padding(.top, 3)

            // Rendered with the same engine as the answer, so markdown and math
            // in a prompt show as formatted text rather than raw syntax. Length
            // is already capped by displayText, so no line limit is needed (and
            // one would truncate the deliberately-kept ending).
            MarkdownTextViewWrapper(text: displayText, bindScrollView: false)
                .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Palette.background)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Palette.surfaceElevated)
                .frame(height: 1)
        }
    }
}

// MARK: - Answer Scroll View

/// Single-screen streaming answer: the latest assistant response rendered as
/// markdown, auto-scrolled to the bottom while tokens arrive. A new prompt
/// replaces the content (the view model resets the conversation).
struct AnswerScrollView: View {
    @ObservedObject var viewModel: LLMChatViewModel

    /// The streaming answer for the current question. Uses the view model's
    /// rate-limited copy: deriving it from `messages` here would re-render (and
    /// so re-parse the markdown) once per token.
    private var answerText: String {
        viewModel.streamingAnswerText
    }

    private var perfText: String? {
        guard let last = viewModel.messages.last(where: { !$0.user.isCurrentUser }),
              let perf = viewModel.pendingPerformanceDataReadable(for: last.id) else { return nil }
        return perf
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(showsIndicators: true) {
                VStack(alignment: .leading, spacing: 12) {
                    if answerText.isEmpty {
                        // Waiting for the first token: keep the area empty.
                        Color.clear
                            .frame(height: 24)
                            .frame(maxWidth: .infinity)
                    } else {
                        // Markdown-rendered streaming answer (code blocks,
                        // headings, lists...) using the same UIKit renderer
                        // the chat bubbles use.
                        MarkdownTextViewWrapper(text: answerText, bindScrollView: false)
                            .frame(maxWidth: .infinity)

                        if let perf = perfText {
                            PerfMetricsView(raw: perf)
                                .padding(.top, 2)
                        }
                    }

                    // Scroll anchor kept last, below the perf readout. Anchoring
                    // on the answer text instead left the readout outside the
                    // viewport when it arrived, forcing a manual drag.
                    Color.clear
                        .frame(height: 1)
                        .id("answer-bottom")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
            }
            .onChange(of: answerText) {
                // No animation while streaming: a 0.15s animation restarted on
                // every text update never finishes, and the competing animations
                // make the scroll position jitter. Snapping each frame tracks the
                // growing text smoothly instead.
                scrollToBottom(proxy, animated: !viewModel.isProcessing)
            }
            // The perf readout is attached after the final token, so answerText
            // is already settled by then and cannot carry this scroll.
            .onChange(of: perfText) {
                scrollToBottom(proxy, animated: true)
            }
        }
        .background(Palette.background)
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool) {
        guard animated else {
            proxy.scrollTo("answer-bottom", anchor: .bottom)
            return
        }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo("answer-bottom", anchor: .bottom)
        }
    }
}
