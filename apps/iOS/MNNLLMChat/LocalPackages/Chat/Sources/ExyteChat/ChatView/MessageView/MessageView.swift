//
//  MessageView.swift
//  Chat
//
//  Created by Alex.M on 23.05.2022.
//

import SwiftUI



struct MessageView: View {
    @Environment(\.chatTheme) private var theme

    @Environment(\.streamingMessageProvider) private var streamingMessageProvider

    @ObservedObject var viewModel: ChatViewModel

    let message: Message
    let positionInUserGroup: PositionInUserGroup
    let chatType: ChatType
    let avatarSize: CGFloat
    let tapAvatarClosure: ChatView.TapAvatarClosure?
    let messageUseMarkdown: Bool
    let isDisplayingMessageMenu: Bool
    let showMessageTimeView: Bool

    @State var avatarViewSize: CGSize = .zero
    @State private var isPresetCollapsed = true
    @State var statusSize: CGSize = .zero
    @State var timeSize: CGSize = .zero

    static let widthWithMedia: CGFloat = 204
    static let horizontalNoAvatarPadding: CGFloat = 8
    static let horizontalAvatarPadding: CGFloat = 8
    static let horizontalTextPadding: CGFloat = 12
    static let horizontalAttachmentPadding: CGFloat = 1 // for multiple attachments
    static let statusViewSize: CGFloat = 14
    static let horizontalStatusPadding: CGFloat = 8
    static let horizontalBubblePadding: CGFloat = 70

    var font: UIFont

    private var isStreamingMessage: Bool {
        // Use injected StreamingMessageProvider protocol to determine streaming state
        // This avoids direct dependency on specific LLMChatViewModel type
        return streamingMessageProvider?.isMessageStreaming(message.id) ?? false
    }

    private var isOutputComplete: Bool {
        // Determine if output is complete based on outputCompleteAnimating state
        let streamingState = streamingMessageProvider?.getStreamingState(message.id)
        return streamingState == .outputCompleteAnimating
    }

    enum DateArrangement {
        case hstack, vstack, overlay
    }

    var additionalMediaInset: CGFloat {
        message.attachments.count > 1 ? MessageView.horizontalAttachmentPadding * 2 : 0
    }

    var dateArrangement: DateArrangement {
        let timeWidth = timeSize.width + 10
        let textPaddings = MessageView.horizontalTextPadding * 2
        let widthWithoutMedia = UIScreen.main.bounds.width
            - (message.user.isCurrentUser ? MessageView.horizontalNoAvatarPadding : avatarViewSize.width)
            - statusSize.width
            - MessageView.horizontalBubblePadding
            - textPaddings

        let maxWidth = message.attachments.isEmpty ? widthWithoutMedia : MessageView.widthWithMedia - textPaddings
        let finalWidth = message.text.width(withConstrainedWidth: maxWidth, font: font, messageUseMarkdown: messageUseMarkdown)
        let lastLineWidth = message.text.lastLineWidth(labelWidth: maxWidth, font: font, messageUseMarkdown: messageUseMarkdown)
        let numberOfLines = message.text.numberOfLines(labelWidth: maxWidth, font: font, messageUseMarkdown: messageUseMarkdown)

        if numberOfLines == 1, finalWidth + CGFloat(timeWidth) < maxWidth {
            return .hstack
        }
        if lastLineWidth + CGFloat(timeWidth) < finalWidth {
            return .overlay
        }
        return .vstack
    }

    var showAvatar: Bool {
        positionInUserGroup == .single
            || (chatType == .conversation && positionInUserGroup == .last)
            || (chatType == .comments && positionInUserGroup == .first)
    }

    var topPadding: CGFloat {
        if chatType == .comments { return 0 }
        return positionInUserGroup == .single || positionInUserGroup == .first ? 8 : 4
    }

    var bottomPadding: CGFloat {
        if chatType == .conversation { return 0 }
        return positionInUserGroup == .single || positionInUserGroup == .first ? 8 : 4
    }

    // Preset prompts can be very long; collapse them in the bubble with an expand toggle.
    private var isCollapsibleMessage: Bool {
        message.user.isCurrentUser && message.text.count > 400
    }

    private var displayedMessageText: String {
        if isCollapsibleMessage && isPresetCollapsed {
            return String(message.text.prefix(200)) + " …"
        }
        return message.text
    }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
//            if !message.user.isCurrentUser {
//                avatarView
//            }

            VStack(alignment: message.user.isCurrentUser ? .trailing : .leading, spacing: 2) {
                if !isDisplayingMessageMenu, let reply = message.replyMessage?.toMessage() {
                    replyBubbleView(reply)
                        .opacity(0.5)
                        .padding(message.user.isCurrentUser ? .trailing : .leading, 10)
                        .overlay(alignment: message.user.isCurrentUser ? .trailing : .leading) {
                            Capsule()
                                .foregroundColor(theme.colors.mainTint)
                                .frame(width: 2)
                        }
                }
                bubbleView(message)
            }

//            if message.user.isCurrentUser, let status = message.status {
//                MessageStatusView(status: status) {
//                    if case let .error(draft) = status {
//                        viewModel.sendMessage(draft)
//                    }
//                }
//                .sizeGetter($statusSize)
//            }
        }
        .padding(.top, topPadding)
        .padding(.bottom, bottomPadding)
        .padding(.trailing, message.user.isCurrentUser ? MessageView.horizontalNoAvatarPadding : MessageView.horizontalBubblePadding)
        .padding(.leading, message.user.isCurrentUser ? MessageView.horizontalBubblePadding : MessageView.horizontalNoAvatarPadding)
        .frame(maxWidth: UIScreen.main.bounds.width, alignment: message.user.isCurrentUser ? .trailing : .leading)
    }

    @ViewBuilder
    func bubbleView(_ message: Message) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if !message.attachments.isEmpty {
                attachmentsView(message)
            }

            if !message.text.isEmpty {
                textWithTimeView(message)
                    .font(Font(font))

                if isCollapsibleMessage {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isPresetCollapsed.toggle()
                        }
                    } label: {
                        Text(isPresetCollapsed ? "展开全文" : "收起")
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                    .padding(.horizontal, MessageView.horizontalTextPadding)
                    .padding(.bottom, 6)
                }
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
    func replyBubbleView(_ message: Message) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(message.user.name)
                .fontWeight(.semibold)
                .padding(.horizontal, MessageView.horizontalTextPadding)

            if !message.attachments.isEmpty {
                attachmentsView(message)
                    .padding(.top, 4)
                    .padding(.bottom, message.text.isEmpty ? 0 : 4)
            }

            if !message.text.isEmpty {
                MessageTextView(text: message.text, messageUseMarkdown: messageUseMarkdown)
                    .padding(.horizontal, MessageView.horizontalTextPadding)
            }

            if let recording = message.recording {
                recordingView(recording)
            }
        }
        .font(.caption2)
        .padding(.vertical, 8)
        .frame(width: message.attachments.isEmpty ? nil : MessageView.widthWithMedia + additionalMediaInset)
        .bubbleBackground(message, theme: theme, isReply: true)
    }

    @ViewBuilder
    var avatarView: some View {
        Group {
            if showAvatar {
                AvatarView(url: message.user.avatarURL, avatarSize: avatarSize)
                    .contentShape(Circle())
                    .onTapGesture {
                        tapAvatarClosure?(message.user, message.id)
                    }
            } else {
                Color.clear.viewSize(avatarSize)
            }
        }
        .padding(.horizontal, MessageView.horizontalAvatarPadding)
        .sizeGetter($avatarViewSize)
    }

    @ViewBuilder
    func attachmentsView(_ message: Message) -> some View {
        AttachmentsGrid(attachments: message.attachments) {
            viewModel.presentAttachmentFullScreen($0)
        }
        .applyIf(message.attachments.count > 1) {
            $0
                .padding(.top, MessageView.horizontalAttachmentPadding)
                .padding(.horizontal, MessageView.horizontalAttachmentPadding)
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
        VStack(alignment: .leading, spacing: 0) {
            // Add ThinkView for assistant messages with thinking content
            if !message.user.isCurrentUser, let thinkText = message.thinkText, !thinkText.isEmpty {
//                print("DEBUG: Displaying ThinkView with content: '\(thinkText)'")
                ThinkView(thinkContent: thinkText)
                    .padding(.horizontal, MessageView.horizontalTextPadding)
            } else if !message.user.isCurrentUser {
//                print("DEBUG: No ThinkView - thinkText is nil or empty: '\(message.thinkText ?? "nil")'")
            }
            
            if #available(iOS 17.0, *) {
                let messageView = LLMMessageTextView(text: displayedMessageText, messageUseMarkdown: message.useMarkdown, messageId: message.id, isAssistantMessage: !message.user.isCurrentUser, isStreamingMessage: isStreamingMessage, isOutputComplete: isOutputComplete, onAnimationComplete: {
                    // Post notification that animation has completed
                    NotificationCenter.default.post(name: NSNotification.Name("StreamingAnimationCompleted"), object: message.id)
                })
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, MessageView.horizontalTextPadding)
                let timeView = messageTimeView()
                    .padding(.trailing, 12)

                HStack(alignment: .lastTextBaseline, spacing: 12) {
                    messageView
                    if !message.attachments.isEmpty {
                        Spacer()
                    }
                    // timeView
                }
                .padding(.vertical, 8)
            } else {
                let messageView = MessageTextView(text: displayedMessageText, messageUseMarkdown: messageUseMarkdown)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, MessageView.horizontalTextPadding)
                let timeView = messageTimeView()
                    .padding(.trailing, 12)

                HStack(alignment: .lastTextBaseline, spacing: 12) {
                    messageView
                    if !message.attachments.isEmpty {
                        Spacer()
                    }
                    // timeView
                }
                .padding(.vertical, 8)
                // Fallback on earlier versions
            }
            
            // Add PerformanceView for assistant messages with performance data
            if !message.user.isCurrentUser, let performanceData = message.performanceData, !performanceData.isEmpty {
                PerformanceView(performanceData: performanceData)
                    .padding(.horizontal, MessageView.horizontalTextPadding)
            }
        }

        /*
         Group {
             switch dateArrangement {
             case .hstack:
                 HStack(alignment: .lastTextBaseline, spacing: 12) {
                     messageView
                     if !message.attachments.isEmpty {
                         Spacer()
                     }
                     timeView
                 }
                 .padding(.vertical, 8)
             case .vstack, .overlay:
                 VStack(alignment: .leading, spacing: 4) {
                     messageView
                     HStack(spacing: 0) {
                         Spacer()
                         timeView
                     }
                 }
                 .padding(.vertical, 8)
             case .overlay:
                 messageView
                     .padding(.vertical, 8)
                     .overlay(alignment: .bottomTrailing) {
                         timeView
                             .padding(.vertical, 8)
                     }
             }
         }
         */
    }

    @ViewBuilder
    func recordingView(_ recording: Recording) -> some View {
        RecordWaveformWithButtons(
            recording: recording,
            colorButton: message.user.isCurrentUser ? theme.colors.messageMyBG : theme.colors.mainBG,
            colorButtonBg: message.user.isCurrentUser ? theme.colors.mainBG : theme.colors.messageMyBG,
            colorWaveform: message.user.isCurrentUser ? theme.colors.messageMyText : theme.colors.messageFriendText
        )
        .padding(.horizontal, MessageView.horizontalTextPadding)
        .padding(.top, 8)
    }

    func messageTimeView(needsCapsule: Bool = false) -> some View {
        Group {
            if showMessageTimeView {
                if needsCapsule {
                    MessageTimeWithCapsuleView(text: message.time, isCurrentUser: message.user.isCurrentUser, chatTheme: theme)
                } else {
                    MessageTimeView(text: message.time, isCurrentUser: message.user.isCurrentUser, chatTheme: theme)
                }
            }
        }
        .sizeGetter($timeSize)
    }
}

extension View {
    @ViewBuilder
    func bubbleBackground(_ message: Message, theme: ChatTheme, isReply: Bool = false) -> some View {
        let radius: CGFloat = !message.attachments.isEmpty ? 12 : 20
        let additionalMediaInset: CGFloat = message.attachments.count > 1 ? 2 : 0
        frame(width: message.attachments.isEmpty ? nil : MessageView.widthWithMedia + additionalMediaInset)
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

#if DEBUG
    struct MessageView_Preview: PreviewProvider {
        static let stan = User(id: "stan", name: "Stan", avatarURL: nil, isCurrentUser: false)
        static let john = User(id: "john", name: "John", avatarURL: nil, isCurrentUser: true)

        private static var shortMessage = "Hi, buddy!"
        private static var longMessage = "Hello hello hello hello hello hello hello hello hello hello hello hello hello\n hello hello hello hello d d d d d d d d"

        private static var replyedMessage = Message(
            id: UUID().uuidString,
            user: stan,
            status: .read,
            text: longMessage,
            attachments: [
                Attachment.randomImage(),
                Attachment.randomImage(),
                Attachment.randomImage(),
                Attachment.randomImage(),
                Attachment.randomImage(),
            ]
        )

        private static var message = Message(
            id: UUID().uuidString,
            user: stan,
            status: .read,
            text: shortMessage,
            replyMessage: replyedMessage.toReplyMessage()
        )

        static var previews: some View {
            ZStack {
                Color.yellow.ignoresSafeArea()

                MessageView(
                    viewModel: ChatViewModel(),
                    message: replyedMessage,
                    positionInUserGroup: .single,
                    chatType: .conversation,
                    avatarSize: 32,
                    tapAvatarClosure: nil,
                    messageUseMarkdown: false,
                    isDisplayingMessageMenu: false,
                    showMessageTimeView: true,
                    font: UIFontMetrics.default.scaledFont(for: UIFont.systemFont(ofSize: 15))
                )
            }
        }
    }
#endif
