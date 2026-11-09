//
//  ConversationView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 13.06.2023.
//

import SwiftUI
import ExyteChat

struct ConversationView: View {

    @Environment(\.presentationMode) var presentationMode

    @StateObject var viewModel: ConversationViewModel

    var body: some View {
        ChatView(messages: viewModel.messages) { draft in
            viewModel.sendMessage(draft)
        }
        .orientationHandler { mode in
            switch mode {
            case .lock: AppDelegate.lockOrientationToPortrait()
            case .unlock: AppDelegate.unlockOrientation()
            }
        }
        .mediaPickerTheme(
            main: .init(
                text: .white,
                albumSelectionBackground: .examplePickerBg,
                fullscreenPhotoBackground: .examplePickerBg
            ),
            selection: .init(
                emptyTint: .white,
                emptyBackground: .black.opacity(0.25),
                selectedTint: .exampleBlue,
                fullscreenTint: .white
            )
        )
        .onDisappear {
            viewModel.resetUnreadCounter()
        }
        .navigationBarBackButtonHidden()
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    presentationMode.wrappedValue.dismiss()
                } label: {
                    Image(.navigateBack)
                }
            }
            ToolbarItem(placement: .navigation) {
                HStack {
                    if let conversation = viewModel.conversation, conversation.isGroup {
                        AvatarView(url: conversation.pictureURL, size: 44)
                        Text(conversation.title)
                    } else if let user = viewModel.users.first {
                        AvatarView(url: user.avatarURL, size: 44)
                        Text(user.name)
                    }
                }
            }
        }
    }
}
