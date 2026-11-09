//
//  LLMChatView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/9/29.
//

import AVFoundation
import ExyteChat
import ExyteMediaPicker
import Foundation
import SwiftUI

struct LLMChatView: View {
    // MARK: - State Properties

    /// Controls the visibility of the settings sheet
    @State private var showSettings = false

    /// Controls the visibility of the batch file test sheet
    @State private var showBatchFileTest = false

    // Message menu sheets
    @State private var editingMessage: MessageSheetText?
    @State private var selectingMessage: MessageSheetText?
    @State private var sharingMessage: MessageSheetText?
    /// Regenerating an older answer drops the turns after it; asks first.
    @State private var pendingRegeneration: Message?

    /// View model for LLM chat functionality
    @StateObject private var viewModel: LLMChatViewModel

    /// Fallback dismissal when the chat is presented outside MainTabView.
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase

    // MARK: - Properties

    /// Title of the chat interface
    private let title: String

    /// Path to the model file
    private let modelPath: String

    /// Explicit route callback used by MainTabView to return to the model list.
    private let onBack: (() -> Void)?
    private var controls: ChatWorkspaceControls? = nil
    private var sharedDraft: Binding<String>? = nil
    private var managesLifecycle = true

    /// Audio recorder settings configuration
    private let recorderSettings = RecorderSettings(audioFormatID: kAudioFormatLinearPCM,
                                                    sampleRate: 44100, numberOfChannels: 2,
                                                    linearPCMBitDepth: 16)

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
        let categoriesContainVideo = (viewModel.modelInfo.categories ?? []).contains { $0.localizedCaseInsensitiveContains("video") }
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
    init(modelInfo: ModelInfo, history: ChatHistory? = nil, onBack: (() -> Void)? = nil) {
        title = modelInfo.name
        modelPath = modelInfo.localPath
        self.onBack = onBack
        let viewModel = LLMChatViewModel(modelInfo: modelInfo, history: history)
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    init(viewModel: LLMChatViewModel, draftText: Binding<String>, controls: ChatWorkspaceControls) {
        title = viewModel.modelInfo.name
        modelPath = viewModel.modelInfo.localPath
        onBack = controls.onManageModels
        self.controls = controls
        sharedDraft = draftText
        managesLifecycle = false
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    private var displayedMessages: [Message] {
        guard controls != nil else { return viewModel.messages }
        // The home screen replaces startup greetings and loading bubbles.
        guard let first = viewModel.messages.firstIndex(where: { $0.user.isCurrentUser }) else { return [] }
        return Array(viewModel.messages[first...]).filter { $0.user.id != "0" }
    }

    private var conversationTheme: ChatTheme {
        ChatTheme(colors: .init(
            mainBG: Color(uiColor: .systemBackground), mainTint: .secondary, mainText: .primary, mainCaptionText: .secondary,
            messageMyBG: .customBlue.opacity(0.2), messageMyText: .primary, messageMyTimeText: .secondary,
            messageFriendBG: .clear, messageFriendText: .primary, messageFriendTimeText: .secondary,
            inputBG: Color(uiColor: .secondarySystemBackground), inputText: .primary, inputPlaceholderText: .secondary,
            menuBG: Color(uiColor: .secondarySystemBackground), menuText: .primary, menuTextDelete: .red,
            statusError: .red, statusGray: .secondary, sendButtonBackground: .customBlue, recordDot: .red
        ), images: .init(
            camera: Image(systemName: "camera"), contact: Image(systemName: "person.crop.circle"),
            document: Image(systemName: "doc"), location: Image(systemName: "location"), photo: Image(systemName: "photo"),
            pickDocument: Image(systemName: "doc.badge.plus"), pickLocation: Image(systemName: "location.circle"),
            pickPhoto: Image(systemName: "photo.on.rectangle"), add: Image(systemName: "plus"),
            arrowSend: Image(systemName: "arrow.up"), attach: Image(systemName: "photo"),
            attachCamera: Image(systemName: "camera"), microphone: Image(systemName: "mic"),
            chevronDown: Image(systemName: "chevron.down"), chevronRight: Image(systemName: "chevron.right"),
            cross: Image(systemName: "xmark"), attachedDocument: Image(systemName: "doc"),
            checkmarks: Image(systemName: "checkmark"), error: Image(systemName: "exclamationmark.circle"),
            muteVideo: Image(systemName: "speaker.slash"), pauseAudio: Image(systemName: "pause.fill"),
            playAudio: Image(systemName: "play.fill"), sending: Image(systemName: "clock"),
            cancelRecord: Image(systemName: "xmark"), deleteRecord: Image(systemName: "trash"),
            lockRecord: Image(systemName: "lock"), pauseRecord: Image(systemName: "pause.fill"),
            playRecord: Image(systemName: "play.fill"), sendRecord: Image(systemName: "arrow.up"),
            stopRecord: Image(systemName: "stop.fill"), cancelReply: Image(systemName: "xmark"),
            replyToMessage: Image(systemName: "arrowshape.turn.up.left"), backButton: Image(systemName: "chevron.left"),
            scrollToBottom: Image(systemName: "chevron.down")
        ))
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            ChatView(messages: displayedMessages, chatType: .conversation) { draft in
                viewModel.sendToLLM(draft: draft)
            }
            .setStreamingMessageProvider(viewModel)
            .messageFooterBuilder { message in
                ChatResponseFooterView(viewModel: viewModel, message: message)
            }
            .setDefaultInputText($viewModel.defaultInputText)
            .setDraftText(sharedDraft)
            .onDraftAttachmentsChanged(controls?.onDraftAttachmentsChanged)
            .showNetworkStatus(false)
            .setInputAvailability(!viewModel.chatInputUnavilable,
                                  isGenerating: viewModel.isProcessing && !viewModel.isAnyDiffusionModel,
                                  isStopping: viewModel.isStopping, onStop: viewModel.cancelGeneration)
            .setInputTrailingAccessory(viewModel.isModelLoaded && !viewModel.isAnyDiffusionModel
                ? AnyView(ContextUsageButton(viewModel: viewModel)) : nil)
            .betweenListAndInputViewBuilder {
                Group { if let controls {
                    HStack {
                        ChatModelPicker(models: controls.models, selectedModel: viewModel.modelInfo,
                                        isBusy: viewModel.isProcessing, onSelect: controls.onSelectModel,
                                        onManageModels: controls.onManageModels)
                        if !viewModel.isAnyDiffusionModel { AgentModeChip(viewModel: viewModel) }
                        Spacer(minLength: 8)
                    }
                    .padding(.horizontal, 12).padding(.top, 8)
                    .background(Color(uiColor: .systemBackground))
                } }
            }
            .setAvailableInput(
                resolvedAvailableInput
//                viewModel.isSanaDiffusionModel ? .textAndMedia :
//                    self.title.lowercased().contains("omni") ? .full :
//                    self.title.lowercased().contains("vl") ? .textAndMedia :
//                    self.title.lowercased().contains("audio") ? .textAndAudio :
//                    (self.title.isEmpty ? .textOnly : .textOnly)
            )
            .messageUseMarkdown(true)
            .useDirectMessageCopyGesture(true)
            .messageMenuItems(menuItems(for:))
            .setRecorderSettings(recorderSettings)
            .setThinkingMode(
                supportsThinkingMode: viewModel.supportsThinkingMode,
                isEnabled: viewModel.isThinkingModeEnabled,
                onToggle: {
                    viewModel.toggleThinkingMode()
                }
            )
            .setMediaPickerSelectionParameters(
                MediaPickerParameters(mediaType: supportsVideoInput ? .photoAndVideo : .photo,
                                      selectionLimit: 1,
                                      showFullscreenPreview: false)
            )
            .chatTheme(conversationTheme)
            .mediaPickerTheme(
                main: .init(
                    text: .white,
                    albumSelectionBackground: .customPickerBg,
                    fullscreenPhotoBackground: .customPickerBg,
                    cameraBackground: .black,
                    cameraSelectionBackground: .black
                ),
                selection: .init(
                    emptyTint: .white,
                    emptyBackground: .black.opacity(0.25),
                    selectedTint: .customBlue,
                    fullscreenTint: .white
                )
            )
            .overlay(alignment: .top) {
                if controls == nil { PresetPromptBar(presets: PresetPrompts.all(for: viewModel.modelInfo)) { preset in
                    viewModel.sendPreset(preset)
                }
                .disabled(viewModel.chatInputUnavilable) }
            }
            .overlay(alignment: .bottom) {
                if controls == nil { DemoInputBar(isEnabled: !viewModel.chatInputUnavilable,
                             isProcessing: viewModel.isProcessing && !viewModel.isAnyDiffusionModel,
                             isStopping: viewModel.isStopping,
                             onStop: viewModel.cancelGeneration) { text in
                    viewModel.sendToLLM(draft: DraftMessage(
                        text: text,
                        thinkText: nil,
                        medias: [],
                        recording: nil,
                        replyMessage: nil,
                        createdAt: Date()
                    ))
                } }
            }
            .overlay {
                if controls != nil && displayedMessages.isEmpty && (sharedDraft?.wrappedValue ?? "").isEmpty && !viewModel.isProcessing {
                    ChatWelcomeView(isReady: viewModel.isModelLoaded, onPrompt: viewModel.sendUserPrompt,
                                    onDownload: returnToModelList)
                        .padding(.bottom, 100)
                }
            }
            // Loading overlay
            if !viewModel.isModelLoaded && controls == nil {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .overlay {
                        if viewModel.loadFailed {
                            ModelLoadFailureView(modelName: title, onRetry: viewModel.retryModelLoad,
                                                 onChooseModel: returnToModelList)
                        } else { VStack(spacing: 20) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(1.5)

                            Text(AppText.text("Model is loading...", comment: ""))
                                .font(.system(size: 15, weight: .regular))
                                .foregroundColor(.white)
                                .font(.headline)
                        } }
                    }
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            if controls != nil && !viewModel.isModelLoaded {
                HStack(spacing: 10) {
                    if viewModel.loadFailed {
                        Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                        Text(AppText.text("chat.loadFailed.title")).font(.caption)
                        Spacer()
                        Button(AppText.text("chat.retryLoad"), action: viewModel.retryModelLoad).font(.caption)
                        Button(AppText.text("chat.chooseModel"), action: returnToModelList).font(.caption)
                    } else {
                        ProgressView()
                        Text(AppText.text("Model is loading...")).font(.caption).foregroundStyle(.secondary)
                        Spacer()
                    }
                }
                .padding(.horizontal, 16).padding(.vertical, 8)
                .background(Color(uiColor: .systemBackground))
            }
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: controls?.onHistory ?? returnToModelList) {
                    if controls != nil { Image(systemName: "line.3.horizontal") }
                    else { Label(AppText.text("My Model"), systemImage: "chevron.left") }
                }
                .accessibilityLabel(AppText.text(controls != nil ? "ChatHistroyTitle" : "My Model"))
                .accessibilityIdentifier("chat.openHistory")
            }

            ToolbarItem(placement: .principal) {
                HStack {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(controls != nil ? AppText.text("workspace.chatTitle") : title)
                            .fontWeight(.semibold)
                            .font(.headline)
                            .foregroundColor(.primary)

                        Text(AppText.text(viewModel.chatStatus))
                            .font(.footnote)
                            .foregroundColor(Color(hex: "AFB3B8"))
                    }
                    Spacer()
                }
                .padding(.leading, 10)
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: { showSettings.toggle() }) {
                    Image(systemName: "gearshape")
                }
                .disabled(viewModel.isProcessing)
                .accessibilityLabel(AppText.text("Settings"))
                .accessibilityIdentifier("chat.settings")
                .sheet(isPresented: $showSettings) {
                    ModelSettingsView(showSettings: $showSettings, viewModel: viewModel)
                }
            }
        }
        .onAppear {
            if managesLifecycle { viewModel.onStart() }
            setupBatchTestCallbacks()
        }
        .onDisappear { if managesLifecycle { viewModel.onStop() } }
        .onChange(of: scenePhase) { _, phase in viewModel.handleScenePhase(phase) }
        .onReceive(NotificationCenter.default.publisher(for: .dismissKeyboard)) { _ in
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
        .sheet(isPresented: $showBatchFileTest) {
            BatchFileTestView(chatViewModel: viewModel)
        }
        .sheet(item: $editingMessage) { message in
            EditMessageSheet(text: message.text) { viewModel.editAndResend(message.messageId, text: $0) }
        }
        .sheet(item: $selectingMessage) { message in
            SelectableTextSheet(text: message.text)
        }
        .sheet(item: $sharingMessage) { message in
            ShareSheet(activityItems: [message.text]) { sharingMessage = nil }
                .presentationDetents([.medium, .large])
        }
        .confirmationDialog(AppText.text("chat.regenerateEarlier.title"),
                            isPresented: Binding(get: { pendingRegeneration != nil },
                                                 set: { if !$0 { pendingRegeneration = nil } }),
                            titleVisibility: .visible, presenting: pendingRegeneration) { prompt in
            Button(AppText.text("chat.regenerate"), role: .destructive) {
                viewModel.regenerateLastResponse(userMessageId: prompt.id)
            }
            Button(AppText.text("Cancel"), role: .cancel) {}
        } message: { _ in
            Text(AppText.text("chat.regenerateEarlier.message"))
        }
    }

    // MARK: - Message Menu

    private func menuItems(for message: Message) -> [MessageMenuItem] {
        guard message.user.isCurrentUser || message.user.id == "2",
              !viewModel.isMessageStreaming(message.id) else { return [] }
        let text = message.text
        let hasText = !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        var items: [MessageMenuItem] = []
        if hasText {
            items.append(MessageMenuItem(title: AppText.text("chat.menu.copy"), systemImage: "doc.on.doc") {
                UIPasteboard.general.string = text
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            })
        }
        if message.user.isCurrentUser {
            if viewModel.canEditAndResend(message) {
                items.append(MessageMenuItem(title: AppText.text("chat.menu.edit"), systemImage: "pencil") {
                    editingMessage = MessageSheetText(messageId: message.id, text: text)
                })
            }
        } else if viewModel.canRegenerate, let prompt = viewModel.promptMessage(for: message.id) {
            items.append(MessageMenuItem(title: AppText.text("chat.regenerate"), systemImage: "arrow.clockwise") {
                if prompt.id == viewModel.messages.last(where: { $0.user.isCurrentUser })?.id {
                    viewModel.regenerateLastResponse(userMessageId: prompt.id)
                } else {
                    pendingRegeneration = prompt
                }
            })
        }
        if hasText {
            items.append(MessageMenuItem(title: AppText.text("chat.menu.select"), systemImage: "selection.pin.in.out") {
                selectingMessage = MessageSheetText(messageId: message.id, text: text)
            })
            items.append(MessageMenuItem(title: AppText.text("chat.menu.share"), systemImage: "square.and.arrow.up") {
                sharingMessage = MessageSheetText(messageId: message.id, text: text)
            })
        }
        return items
    }

    // MARK: - Private Methods

    private func returnToModelList() {
        if let onBack {
            onBack()
        } else {
            dismiss()
        }
    }

    /// Setup callbacks for batch test functionality
    private func setupBatchTestCallbacks() {
        // Setup any additional callbacks if needed
    }
}

// MARK: - Preset Prompt Bar

struct PresetPromptBar: View {
    let presets: [PresetPrompt]
    let onSelect: (PresetPrompt) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(presets.enumerated()), id: \.offset) { _, preset in
                    Button {
                        onSelect(preset)
                    } label: {
                        HStack(spacing: 4) {
                            if preset.isASRAudio {
                                Image(systemName: "waveform")
                                    .font(.system(size: 12, weight: .semibold))
                                    .frame(width: 18, height: 18)
                                    .background(Color.customBlue.opacity(0.12))
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            } else if let path = preset.imageBundlePath, let ui = UIImage(contentsOfFile: path) {
                                Image(uiImage: ui)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 18, height: 18)
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            } else {
                                Image(systemName: preset.icon)
                                    .font(.system(size: 12))
                            }
                            Text(preset.title)
                                .font(.system(size: 13, weight: .medium))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(Capsule().fill(Color(hex: "F2F3F5")))
                        .foregroundColor(.primary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .background(Color.white.opacity(0.95))
    }
}

// MARK: - Demo Input Bar

struct DemoInputBar: View {
    let isEnabled: Bool
    var isProcessing = false
    var isStopping = false
    var onStop: () -> Void = {}
    let onSend: (String) -> Void

    @State private var text: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 8) {
            TextField(AppText.text("chat.input.placeholder"), text: $text, axis: .vertical)
                .lineLimit(1...4)
                .textFieldStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(Capsule().fill(Color(hex: "F2F3F5")))
                .focused($focused)
                .onSubmit(send)
                .disabled(!isEnabled)
                .opacity(isEnabled ? 1 : 0.5)

            if isProcessing {
                Button(action: onStop) {
                    Image(systemName: "stop.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(isStopping ? Color.secondary : Color.customBlue)
                }
                .disabled(isStopping)
                .accessibilityLabel(AppText.text(isStopping ? "chat.stopping" : "Stop generating"))
                .accessibilityIdentifier("chat.stop")
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 32))
                }
                .disabled(!isEnabled || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .accessibilityLabel(AppText.text("Send"))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    private func send() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isEnabled, !trimmed.isEmpty else { return }
        text = ""
        focused = false
        onSend(trimmed)
    }
}

struct ChatResponseFooterView: View {
    @ObservedObject var viewModel: LLMChatViewModel
    let message: Message
    @AppStorage("chat.showPrefillSpeed") private var showPrefillSpeed = true
    @AppStorage("chat.showDecodeSpeed") private var showDecodeSpeed = true

    var visiblePerformanceData: String {
        guard showPrefillSpeed || showDecodeSpeed else { return "" }
        return (message.performanceData ?? "").split(whereSeparator: \.isNewline).filter { line in
            (!line.hasPrefix("Prefill:") || showPrefillSpeed) &&
                (!line.hasPrefix("Decode:") || showDecodeSpeed)
        }.joined(separator: "\n")
    }

    private var isLatestResponse: Bool {
        guard let userIndex = viewModel.messages.lastIndex(where: { $0.user.isCurrentUser }) else { return false }
        return viewModel.messages.dropFirst(userIndex + 1).last(where: { $0.user.id == "2" })?.id == message.id
    }

    var body: some View {
        if message.user.id == "2" && !viewModel.isMessageStreaming(message.id) {
            VStack(alignment: .leading, spacing: 4) {
                if !visiblePerformanceData.isEmpty {
                    Text(visiblePerformanceData)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 4)
                        .accessibilityIdentifier("chat.performance")
                }
                if isLatestResponse && viewModel.canRegenerate {
                    ChatResponseActionsView(viewModel: viewModel)
                }
            }
        }
    }
}

struct ChatResponseActionsView: View {
    @ObservedObject var viewModel: LLMChatViewModel

    var body: some View {
        HStack(spacing: 18) {
            Button { viewModel.regenerateLastResponse() } label: {
                Label(AppText.text("chat.regenerate"), systemImage: "arrow.clockwise")
            }
            .disabled(!viewModel.canRegenerate)
            .accessibilityIdentifier("chat.regenerate")
            if viewModel.canGenerateDirectAnswer {
                Button { viewModel.regenerateLastResponse(directAnswer: true) } label: {
                    Label(AppText.text("chat.directAnswer"), systemImage: "bolt")
                }
                .accessibilityIdentifier("chat.directAnswer")
            }
            Spacer(minLength: 0)
            if viewModel.historySaveFailed {
                Button { viewModel.saveHistoryImmediately() } label: {
                    Label(AppText.text("chat.retrySave"), systemImage: "exclamationmark.circle")
                }
                .foregroundStyle(.orange)
            }
        }
        .font(.footnote)
        .buttonStyle(.plain)
        .padding(.vertical, 10)
        .foregroundStyle(.secondary)
    }
}

struct ModelLoadFailureView: View {
    let modelName: String
    let onRetry: () -> Void
    let onChooseModel: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.orange)
            Text(AppText.text("chat.loadFailed.title")).font(.headline)
            Text(modelName).font(.subheadline).foregroundStyle(.secondary)
            Text(AppText.text("chat.loadFailed.detail"))
                .font(.subheadline)
                .multilineTextAlignment(.center)
            Button(action: onRetry) {
                Label(AppText.text("chat.retryLoad"), systemImage: "arrow.clockwise")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier("chat.retryLoad")
            Button(AppText.text("chat.chooseModel"), action: onChooseModel)
                .accessibilityIdentifier("chat.chooseModel")
        }
        .padding(24)
        .frame(maxWidth: 340)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
        .padding(24)
    }
}
