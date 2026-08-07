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

    /// View model for LLM chat functionality
    @StateObject private var viewModel: LLMChatViewModel

    /// Environment variable for presentation mode
    @Environment(\.presentationMode) private var presentationMode

    // MARK: - Properties

    /// Title of the chat interface
    private let title: String

    /// Path to the model file
    private let modelPath: String

    /// Audio recorder settings configuration
    private let recorderSettings = RecorderSettings(audioFormatID: kAudioFormatLinearPCM,
                                                    sampleRate: 44100, numberOfChannels: 2,
                                                    linearPCMBitDepth: 16)

    private var supportsAudioInput: Bool {
        if ModelUtils.isAudioModel(viewModel.modelInfo.modelName) {
            return true
        }
        return viewModel.modelInfo.tags.contains { $0.localizedCaseInsensitiveContains("audio") }
    }

    private var supportsVisualInput: Bool {
        if ModelUtils.isVisualModel(viewModel.modelInfo.modelName) {
            return true
        }
        
        if viewModel.isSanaDiffusionModel {
            return true
        }
            
        let tagMatches = viewModel.modelInfo.tags.contains { tag in
            tag.localizedCaseInsensitiveContains("image") || tag.localizedCaseInsensitiveContains("video")
        }
        let categoryMatches = (viewModel.modelInfo.categories ?? []).contains { category in
            category.localizedCaseInsensitiveContains("image") || category.localizedCaseInsensitiveContains("video")
        }
        return tagMatches || categoryMatches
    }

    private var supportsVideoInput: Bool {
        guard supportsVisualInput else { return false }
        let nameContainsVideo = viewModel.modelInfo.modelName.localizedCaseInsensitiveContains("video")
        let tagsContainVideo = viewModel.modelInfo.tags.contains { $0.localizedCaseInsensitiveContains("video") }
        let categoriesContainVideo = (viewModel.modelInfo.categories ?? []).contains { $0.localizedCaseInsensitiveContains("video") }
        return nameContainsVideo || tagsContainVideo || categoriesContainVideo || ModelUtils.isOmni(viewModel.modelInfo.modelName)
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
    init(modelInfo: ModelInfo, history: ChatHistory? = nil) {
        title = modelInfo.modelName
        modelPath = modelInfo.localPath
        let viewModel = LLMChatViewModel(modelInfo: modelInfo, history: history)
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            ChatView(messages: viewModel.messages, chatType: .conversation) { draft in
                viewModel.sendToLLM(draft: draft)
            }
            .setStreamingMessageProvider(viewModel)
            .setDefaultInputText($viewModel.defaultInputText)
            .setAvailableInput(
                resolvedAvailableInput
//                viewModel.isSanaDiffusionModel ? .textAndMedia :
//                    self.title.lowercased().contains("omni") ? .full :
//                    self.title.lowercased().contains("vl") ? .textAndMedia :
//                    self.title.lowercased().contains("audio") ? .textAndAudio :
//                    (self.title.isEmpty ? .textOnly : .textOnly)
            )
            .messageUseMarkdown(true)
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
            .chatTheme(
                ChatTheme(
                    colors: .init(
                        messageMyBG: .customBlue.opacity(0.2),
                        messageFriendBG: .clear
                    ),
                    images: .init(
                        attach: Image(systemName: "photo"),
                        attachCamera: Image("attachCamera", bundle: .current)
                    )
                )
            )
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
            .navigationBarTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden()
            .disabled(!viewModel.isModelLoaded)
            .overlay(alignment: .top) {
                PresetPromptBar(presets: PresetPrompts.all()) { preset in
                    viewModel.sendPreset(preset)
                }
                .disabled(viewModel.chatInputUnavilable)
            }
            .overlay(alignment: .bottom) {
                DemoInputBar(isEnabled: !viewModel.chatInputUnavilable) { text in
                    viewModel.sendToLLM(draft: DraftMessage(
                        text: text,
                        thinkText: nil,
                        medias: [],
                        recording: nil,
                        replyMessage: nil,
                        createdAt: Date()
                    ))
                }
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        presentationMode.wrappedValue.dismiss()
                    } label: {
                        Image("backArrow", bundle: .current)
                    }
                }

                ToolbarItem(placement: .principal) {
                    HStack {
                        VStack(alignment: .leading, spacing: 0) {
                            Text(title)
                                .fontWeight(.semibold)
                                .font(.headline)
                                .foregroundColor(.black)

                            Text(viewModel.chatStatus)
                                .font(.footnote)
                                .foregroundColor(Color(hex: "AFB3B8"))
                        }
                        Spacer()
                    }
                    .padding(.leading, 10)
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 8) {
                        // Settings Button
                        Button(action: { showSettings.toggle() }) {
                            Image(systemName: "gear")
                        }
                        .sheet(isPresented: $showSettings) {
                            ModelSettingsView(showSettings: $showSettings, viewModel: viewModel)
                        }

                        // Three-dot menu with batch testing options
                        ChatMenuView(
                            showBatchFileTest: $showBatchFileTest
                        )
                    }
                }
            }
            .onAppear {
                viewModel.onStart()
                setupBatchTestCallbacks()
            }
            .onDisappear(perform: viewModel.onStop)
            .onReceive(NotificationCenter.default.publisher(for: .dismissKeyboard)) { _ in
                // Hidden keyboard
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
            // Batch File Test Sheet
            .sheet(isPresented: $showBatchFileTest) {
                BatchFileTestView(chatViewModel: viewModel)
            }

            // Loading overlay
            if !viewModel.isModelLoaded {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .overlay(
                        VStack(spacing: 20) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(1.5)

                            Text(NSLocalizedString("Model is loading...", comment: ""))
                                .font(.system(size: 15, weight: .regular))
                                .foregroundColor(.white)
                                .font(.headline)
                        }
                    )
            }
        }
    }

    // MARK: - Private Methods

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
                            if let path = preset.imageBundlePath, let ui = UIImage(contentsOfFile: path) {
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
                        .foregroundColor(.black)
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
    let onSend: (String) -> Void

    @State private var text: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 8) {
            TextField("输入消息…", text: $text, axis: .vertical)
                .lineLimit(1...4)
                .textFieldStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(Capsule().fill(Color(hex: "F2F3F5")))
                .focused($focused)
                .onSubmit(send)

            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 28))
            }
            .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.5)
    }

    private func send() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isEnabled, !trimmed.isEmpty else { return }
        text = ""
        focused = false
        onSend(trimmed)
    }
}
