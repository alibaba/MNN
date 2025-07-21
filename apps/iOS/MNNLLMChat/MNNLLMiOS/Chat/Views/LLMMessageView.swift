//
//  LLMMessageView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/7.
//

import SwiftUI
import Foundation
import SwiftUI
import ExyteChat

// MARK: - Custom Message View
struct LLMMessageView: View {
    let message: Message
    let positionInGroup: PositionInUserGroup
    let isAssistantMessage: Bool
    let isStreamingMessage: Bool
    let showContextMenuClosure: () -> Void
    let messageActionClosure: (Message, DefaultMessageMenuAction) -> Void
    let showAttachmentClosure: (Attachment) -> Void
    
    let theme = ChatTheme(
        colors: .init(
            messageMyBG: .customBlue.opacity(0.2),
            messageFriendBG: .clear
        ),
        images: .init(
            attach: Image(systemName: "photo"),
            attachCamera: Image("attachCamera", bundle: .current)
        )
    )
    
    @State var avatarViewSize: CGSize = .zero
    @State var timeSize: CGSize = .zero
    
    static let widthWithMedia: CGFloat = 204
    static let horizontalNoAvatarPadding: CGFloat = 8
    static let horizontalAvatarPadding: CGFloat = 8
    static let horizontalTextPadding: CGFloat = 12
    static let horizontalAttachmentPadding: CGFloat = 1
    static let horizontalBubblePadding: CGFloat = 70
    
    var additionalMediaInset: CGFloat {
        message.attachments.count > 1 ? LLMMessageView.horizontalAttachmentPadding * 2 : 0
    }
    
    var showAvatar: Bool {
        positionInGroup == .single
        || positionInGroup == .last
    }
    
    var topPadding: CGFloat {
        positionInGroup == .single || positionInGroup == .first ? 8 : 4
    }
    
    var bottomPadding: CGFloat {
        0
    }
    
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if !message.user.isCurrentUser {
                avatarView
            }
            
            VStack(alignment: message.user.isCurrentUser ? .trailing : .leading, spacing: 2) {
                bubbleView(message)
            }
        }
        .padding(.top, topPadding)
        .padding(.bottom, bottomPadding)
        .padding(.trailing, message.user.isCurrentUser ? LLMMessageView.horizontalNoAvatarPadding : 0)
        .padding(message.user.isCurrentUser ? .leading : .trailing, message.user.isCurrentUser ? LLMMessageView.horizontalBubblePadding : 0)
        .frame(maxWidth: UIScreen.main.bounds.width, alignment: message.user.isCurrentUser ? .trailing : .leading)
        .contentShape(Rectangle())
        .onLongPressGesture {
            showContextMenuClosure()
        }
    }
    
    @ViewBuilder
    func bubbleView(_ message: Message) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if !message.attachments.isEmpty {
                attachmentsView(message)
            }
            
            if !message.text.isEmpty {
                textWithTimeView(message)
            }
            
            if let recording = message.recording {
                VStack(alignment: .trailing, spacing: 8) {
                    recordingView(recording)
                    messageTimeView()
                        .padding(.bottom, 8)
                        .padding(.trailing, 12)
                }
            }
        }
        .bubbleBackground(message, theme: theme)
    }
    
    @ViewBuilder
    var avatarView: some View {
        Group {
            if showAvatar {
                AsyncImage(url: message.user.avatarURL) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    Circle()
                        .fill(Color.gray.opacity(0.3))
                }
                .frame(width: 32, height: 32)
                .clipShape(Circle())
                .contentShape(Circle())
            } else {
                Color.clear.frame(width: 32, height: 32)
            }
        }
        .padding(.horizontal, LLMMessageView.horizontalAvatarPadding)
        .sizeGetter($avatarViewSize)
    }
    
    @ViewBuilder
    func attachmentsView(_ message: Message) -> some View {
        ForEach(message.attachments, id: \.id) { attachment in
            AsyncImage(url: attachment.thumbnail) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } placeholder: {
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
            }
            .frame(maxWidth: LLMMessageView.widthWithMedia, maxHeight: 200)
            .cornerRadius(12)
            .onTapGesture {
                showAttachmentClosure(attachment)
            }
        }
        .applyIf(message.attachments.count > 1) {
            $0
                .padding(.top, LLMMessageView.horizontalAttachmentPadding)
                .padding(.horizontal, LLMMessageView.horizontalAttachmentPadding)
        }
        .overlay(alignment: .bottomTrailing) {
            if message.text.isEmpty {
                messageTimeView(needsCapsule: true)
                    .padding(4)
            }
        }
        .contentShape(Rectangle())
    }
    
    @ViewBuilder
    func textWithTimeView(_ message: Message) -> some View {
        // Message View with Type Writer Animation
        let messageView = LLMMessageTextView(
            text: message.text,
            messageUseMarkdown: true,
            messageId: message.id,
            isAssistantMessage: isAssistantMessage,
            isStreamingMessage: isStreamingMessage
        )
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, LLMMessageView.horizontalTextPadding)
        
        HStack(alignment: .lastTextBaseline, spacing: 12) {
            messageView
            if !message.attachments.isEmpty {
                Spacer()
            }
        }
        .padding(.vertical, 8)
    }
    
    @ViewBuilder
    func recordingView(_ recording: Recording) -> some View {
        HStack {
            Image(systemName: "mic.fill")
                .foregroundColor(.blue)
            Text("Audio Message")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, LLMMessageView.horizontalTextPadding)
        .padding(.top, 8)
    }
    
    func messageTimeView(needsCapsule: Bool = false) -> some View {
        Group {
            if needsCapsule {
                Text(DateFormatter.timeFormatter.string(from: message.createdAt))
                    .font(.caption)
                    .foregroundColor(.white)
                    .opacity(0.8)
                    .padding(.top, 4)
                    .padding(.bottom, 4)
                    .padding(.horizontal, 8)
                    .background {
                        Capsule()
                            .foregroundColor(.black.opacity(0.4))
                    }
            } else {
                Text(DateFormatter.timeFormatter.string(from: message.createdAt))
                    .font(.caption)
                    .foregroundColor(message.user.isCurrentUser ? theme.colors.messageMyTimeText : theme.colors.messageFriendTimeText)
            }
        }
        .sizeGetter($timeSize)
    }
}

// MARK: - View Extensions
extension View {
    @ViewBuilder
    func sizeGetter(_ size: Binding<CGSize>) -> some View {
        self.background(
            GeometryReader { geometry in
                Color.clear
                    .preference(key: SizePreferenceKey.self, value: geometry.size)
            }
        )
        .onPreferenceChange(SizePreferenceKey.self) { newSize in
            size.wrappedValue = newSize
        }
    }
    
    @ViewBuilder
    func applyIf<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}

// MARK: - Preference Key
struct SizePreferenceKey: PreferenceKey {
    static var defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) {}
}

// MARK: - Date Formatter Extension
extension DateFormatter {
    static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }()
}

extension View {
    @ViewBuilder
    func bubbleBackground(_ message: Message, theme: ChatTheme, isReply: Bool = false) -> some View {
        let radius: CGFloat = !message.attachments.isEmpty ? 12 : 20
        let additionalMediaInset: CGFloat = message.attachments.count > 1 ? 2 : 0
        self
            .frame(width: message.attachments.isEmpty ? nil : LLMMessageView.widthWithMedia + additionalMediaInset)
            .foregroundColor(message.user.isCurrentUser ? theme.colors.messageMyText : theme.colors.messageFriendText)
            .background {
                if isReply || !message.text.isEmpty || message.recording != nil {
                    RoundedRectangle(cornerRadius: radius)
                        .foregroundColor(message.user.isCurrentUser ? theme.colors.messageMyBG : theme.colors.messageFriendBG)
                        .opacity(isReply ? 0.5 : 1)
                }
            }
            .cornerRadius(radius)
    }
}
