//
//  LLMChatView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/8.
//

import Foundation
import SwiftUI
import ExyteChat
import ExyteMediaPicker
import AVFoundation
import MessageUI
import EventKit

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
        ZStack {
            ChatView(messages: viewModel.messages, chatType: .conversation) { draft in
                viewModel.sendToLLM(draft: draft)
            }
            .setStreamingMessageProvider {
                viewModel.currentStreamingMessageId
            }
            .setAvailableInput(
                self.title.lowercased().contains("omni") ? .full:
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
                        // System Integration Buttons
                        Button(action: {
                            // Web Search
                            if let url = URL(string: "https://www.google.com/search?q=ARIA") {
                                UIApplication.shared.open(url)
                            }
                        }) {
                            Image(systemName: "magnifyingglass")
                        }

                        @State private var showMailView = false
                        @State private var showAlert = false
                        @State private var alertMessage = ""

                        Button(action: {
                            // Email
                            if MFMailComposeViewController.canSendMail() {
                                showMailView = true
                            } else {
                                alertMessage = "Mail services are not available."
                                showAlert = true
                            }
                        }) {
                            Image(systemName: "envelope")
                        }
                        .sheet(isPresented: $showMailView) {
                            MailView(recipients: ["test@example.com"], subject: "ARIA Test")
                        }
                        .alert(isPresented: $showAlert) {
                            Alert(title: Text("Error"), message: Text(alertMessage), dismissButton: .default(Text("OK")))
                        }

                        Button(action: {
                            // Calendar
                            let eventStore = EKEventStore()
                            eventStore.requestAccess(to: .event) { (granted, error) in
                                if granted && error == nil {
                                    DispatchQueue.main.async {
                                        let event = EKEvent(eventStore: eventStore)
                                        event.title = "ARIA Event"
                                        event.startDate = Date()
                                        event.endDate = Date().addingTimeInterval(60 * 60)
                                        event.location = "ARIA Office"
                                        do {
                                            try eventStore.save(event, span: .thisEvent)
                                            alertMessage = "Event saved successfully."
                                            showAlert = true
                                        } catch {
                                            alertMessage = "Failed to save event."
                                            showAlert = true
                                        }
                                    }
                                } else {
                                    alertMessage = "Access to calendar was denied."
                                    showAlert = true
                                }
                            }
                        }) {
                            Image(systemName: "calendar")
                        }

                        Button(action: {
                            // Share
                            let activityViewController = UIActivityViewController(activityItems: ["Check out ARIA!"], applicationActivities: nil)
                            UIApplication.shared.windows.first?.rootViewController?.present(activityViewController, animated: true)
                        }) {
                            Image(systemName: "square.and.arrow.up")
                        }

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
                // Hidden keyboard
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
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
