//
//  LLMChatView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/8.
//

import Foundation
import SwiftUI
import ExyteChat
import AVFoundation

struct LLMChatView: View {
    @StateObject private var viewModel: LLMChatViewModel
    @Environment(\.presentationMode) private var presentationMode
    
    private let title: String
    private let modelPath: String

    private let recorderSettings = RecorderSettings(audioFormatID: kAudioFormatLinearPCM,
                                                    sampleRate: 44100, numberOfChannels: 2,
                                                    linearPCMBitDepth: 16)

    @State private var showSettings = false

    init(modelInfo: ModelInfo, history: ChatHistory? = nil) {
        self.title = modelInfo.modelName
        self.modelPath = modelInfo.localPath
        let viewModel = LLMChatViewModel(modelInfo: modelInfo, history: history)
        _viewModel = StateObject(wrappedValue: viewModel)
    }
    
    var body: some View {
        ChatView(messages: viewModel.messages, chatType: .conversation) { draft in
            viewModel.sendToLLM(draft: draft)
        }
        .setStreamingMessageProvider {
            viewModel.currentStreamingMessageId
        }
        .setAvailableInput(
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
                }
            }
        }
        
        .onAppear {
            viewModel.onStart()
        }
        .onDisappear(perform: viewModel.onStop)
        .onReceive(NotificationCenter.default.publisher(for: .dismissKeyboard)) { _ in
            // 隐藏键盘
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }
    
    // MARK: - LLM Chat Message Builder
    @ViewBuilder
    private func LLMChatMessageView(
        message: Message,
        positionInGroup: PositionInUserGroup,
        showContextMenuClosure: @escaping () -> Void,
        messageActionClosure: @escaping (Message, DefaultMessageMenuAction) -> Void,
        showAttachmentClosure: @escaping (Attachment) -> Void
    ) -> some View {
        LLMMessageView(
            message: message,
            positionInGroup: positionInGroup,
            isAssistantMessage: !message.user.isCurrentUser,
            isStreamingMessage: viewModel.currentStreamingMessageId == message.id,
            showContextMenuClosure: {
                if !viewModel.isProcessing {
                    showContextMenuClosure()
                }
            },
            messageActionClosure: messageActionClosure,
            showAttachmentClosure: showAttachmentClosure
        )
    }
}
