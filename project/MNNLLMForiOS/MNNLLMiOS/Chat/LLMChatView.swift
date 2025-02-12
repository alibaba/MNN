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

    private let recorderSettings = RecorderSettings(audioFormatID: kAudioFormatLinearPCM, sampleRate: 44100, numberOfChannels: 2, linearPCMBitDepth: 16)

    init(modelInfo: ModelInfo, history: ChatHistory? = nil) {
        self.title = modelInfo.name
        self.modelPath = modelInfo.localPath
        let viewModel = LLMChatViewModel(modelInfo: modelInfo, history: history)
        _viewModel = StateObject(wrappedValue: viewModel)
    }
    
    var body: some View {
        ChatView(messages: viewModel.messages, chatType: .conversation) { draft in
            viewModel.sendToLLM(draft: draft)
        }
        .setAvailableInput(
            self.title.lowercased().contains("vl") ? .textAndMedia :
            self.title.lowercased().contains("audio") ? .textAndAudio :
            (self.title.isEmpty ? .textOnly : .textOnly)
        )
        .messageUseMarkdown(true)
        .setRecorderSettings(recorderSettings)
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
                fullscreenPhotoBackground: .customPickerBg
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
                Button { presentationMode.wrappedValue.dismiss() } label: {
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
        }
        
        .onAppear {
            viewModel.onStart()
        }
        .onDisappear(perform: viewModel.onStop)
    }
}
