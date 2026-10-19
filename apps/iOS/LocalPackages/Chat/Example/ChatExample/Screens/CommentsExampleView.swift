//
//  CommentsExampleView.swift
//  ChatExample
//
//  Created by Alisa Mylnikova on 28.06.2024.
//

import SwiftUI
import ExyteChat

enum Action: MessageMenuAction {
    case reply, edit, delete, print

    func title() -> String {
        switch self {
        case .reply:
            "Reply"
        case .edit:
            "Edit"
        case .delete:
            "Delete"
        case .print:
            "Print"
        }
    }
    
    func icon() -> Image {
        switch self {
        case .reply:
            Image(systemName: "arrowshape.turn.up.left")
        case .edit:
            Image(systemName: "square.and.pencil")
        case .delete:
            Image(systemName: "xmark.bin")
        case .print:
            Image(systemName: "printer")
        }
    }
}

struct CommentsExampleView: View {

    @StateObject var viewModel = CommentsExampleViewModel()

    var body: some View {
        VStack {
            ZStack {
                Color.blue.opacity(0.2)
                Text("An interesting post for people to comment on")
                    .font(.system(size: 30))
                    .fontWeight(.bold)
                    .frame(maxWidth: .infinity)
                    .padding(30)
            }
            .fixedSize(horizontal: false, vertical: true)

            ChatView(messages: viewModel.messages, chatType: .comments, replyMode: .answer) { draft in
                viewModel.send(draft: draft)
            } messageBuilder: { message, positionInGroup, positionInCommentsGroup, showContextMenuClosure, messageActionClosure, showAttachmentClosure in
                messageCell(message, positionInCommentsGroup, showMenuClosure: showContextMenuClosure, actionClosure: messageActionClosure, attachmentClosure: showAttachmentClosure)
            } messageMenuAction: { (action: Action, defaultActionClosure, message) in
                switch action {
                case .reply:
                    defaultActionClosure(message, .reply)
                case .edit:
                    defaultActionClosure(message, .edit { editedText in
                        // update this message's text on your BE
                        print(editedText)
                    })
                case .delete:
                    print("deleted")
                case .print:
                    print(message.text)
                }
            }
            .showDateHeaders(false)
        }
        .navigationTitle("Comments example")
        .onAppear(perform: viewModel.onStart)
        .onDisappear(perform: viewModel.onStop)
    }

    @ViewBuilder
    func messageCell(_ message: Message, _ commentsPosition: CommentsPosition?, showMenuClosure: @escaping ()->(), actionClosure: @escaping (Message, DefaultMessageMenuAction) -> Void, attachmentClosure: @escaping (Attachment) -> Void) -> some View {
        VStack {
            HStack(alignment: .top, spacing: 12) {
                CachedAsyncImage(url: message.user.avatarURL) { image in
                    image
                        .resizable()
                        .scaledToFill()
                } placeholder: {
                    Rectangle().fill(Color.gray)
                }
                .frame(width: 40, height: 40)
                .clipShape(Circle())

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(message.user.name)
                            .font(.system(size: 14)).fontWeight(.semibold)
                        Spacer()
                        Text(message.createdAt.formatAgo())
                            .font(.system(size: 12)).fontWeight(.medium)
                    }

                    if !message.text.isEmpty {
                        Text(message.text)
                            .font(.system(size: 12)).fontWeight(.medium)
                            .foregroundStyle(.gray)
                    }

                    if !message.attachments.isEmpty {
                        LazyVGrid(columns: Array(repeating: GridItem(), count: 2), spacing: 8) {
                            ForEach(message.attachments) { attachment in
                                CachedAsyncImage(url: attachment.thumbnail) { image in
                                    image
                                        .resizable()
                                        .aspectRatio(1, contentMode: .fill)
                                        .cornerRadius(12)
                                } placeholder: {
                                    Rectangle().fill(Color.gray)
                                        .aspectRatio(1, contentMode: .fill)
                                        .cornerRadius(12)
                                }
                                .onTapGesture {
                                    attachmentClosure(attachment)
                                }
                            }
                        }
                    }

                    HStack {
                        if message.replyMessage == nil {
                            Group {
                                Image(systemName: "bubble")
                                    .padding(.top, 4)
                                Text("Reply")
                                    .font(.system(size: 14)).fontWeight(.medium)
                            }
                            .onTapGesture {
                                actionClosure(message, .reply)
                            }
                        }

                        Spacer()
                    }
                }
            }
            .padding(.leading, message.replyMessage != nil ? 40 : 0)

            if let commentsPosition {
                if commentsPosition.isLastInCommentsGroup {
                    Color.gray.frame(height: 0.5)
                        .padding(.vertical, 10)
                } else if commentsPosition.isLastInChat {
                    Color.clear.frame(height: 5)
                } else {
                    Color.clear.frame(height: 10)
                }
            }
        }
        .padding(.horizontal, 18)
    }
}
