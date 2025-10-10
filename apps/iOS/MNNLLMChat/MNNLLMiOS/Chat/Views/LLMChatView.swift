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
    @State private var showSettings = false
    @State private var showBatchFileTest = false
    @StateObject private var viewModel: LLMChatViewModel
    @Environment(\.presentationMode) private var presentationMode

    private let title: String
    private let modelPath: String
    private let recorderSettings = RecorderSettings(audioFormatID: kAudioFormatLinearPCM,
                                                    sampleRate: 44100, numberOfChannels: 2,
                                                    linearPCMBitDepth: 16)

    init(modelInfo: ModelInfo, history: ChatHistory? = nil) {
        title = modelInfo.modelName
        modelPath = modelInfo.localPath
        let viewModel = LLMChatViewModel(modelInfo: modelInfo, history: history)
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        ZStack {
            ChatView(messages: viewModel.messages, chatType: .conversation) { draft in
                viewModel.sendToLLM(draft: draft)
            }
            .setStreamingMessageProvider(viewModel)
            .setAvailableInput(
                self.title.lowercased().contains("omni") ? .full :
                    self.title.lowercased().contains("vl") ? .textAndMedia :
                    self.title.lowercased().contains("audio") ? .textAndAudio :
                    (self.title.isEmpty ? .textOnly : .textOnly)
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
                MediaPickerParameters(mediaType: .photo,
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
            .disabled(viewModel.chatInputUnavilable)
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
                        
                        // Three-dot menu
                        ChatMenuView(showBatchFileTest: $showBatchFileTest)
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
