//
//  SwiftUIView.swift
//  
//
//  Created by Alisa Mylnikova on 06.12.2023.
//

import SwiftUI

public extension ChatView where MessageContent == EmptyView {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void,
         inputViewBuilder: @escaping InputViewBuilderClosure,
         messageMenuAction: MessageMenuActionClosure?) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
        self.inputViewBuilder = inputViewBuilder
        self.messageMenuAction = messageMenuAction
    }
}

public extension ChatView where InputViewContent == EmptyView {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void,
         messageBuilder: @escaping MessageBuilderClosure,
         messageMenuAction: MessageMenuActionClosure?) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
        self.messageBuilder = messageBuilder
        self.messageMenuAction = messageMenuAction
    }
}

public extension ChatView where MenuAction == DefaultMessageMenuAction {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void,
         messageBuilder: @escaping MessageBuilderClosure,
         inputViewBuilder: @escaping InputViewBuilderClosure) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
        self.messageBuilder = messageBuilder
        self.inputViewBuilder = inputViewBuilder
    }
}

public extension ChatView where MessageContent == EmptyView, InputViewContent == EmptyView {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void,
         messageMenuAction: MessageMenuActionClosure?) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
        self.messageMenuAction = messageMenuAction
    }
}

public extension ChatView where InputViewContent == EmptyView, MenuAction == DefaultMessageMenuAction {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void,
         messageBuilder: @escaping MessageBuilderClosure) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
        self.messageBuilder = messageBuilder
    }
}

public extension ChatView where MessageContent == EmptyView, MenuAction == DefaultMessageMenuAction {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void,
         inputViewBuilder: @escaping InputViewBuilderClosure) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
        self.inputViewBuilder = inputViewBuilder
    }
}

public extension ChatView where MessageContent == EmptyView, InputViewContent == EmptyView, MenuAction == DefaultMessageMenuAction {

    init(messages: [Message],
         chatType: ChatType = .conversation,
         replyMode: ReplyMode = .quote,
         didSendMessage: @escaping (DraftMessage) -> Void) {
        self.type = chatType
        self.didSendMessage = didSendMessage
        self.sections = ChatView.mapMessages(messages, chatType: chatType, replyMode: replyMode)
        self.ids = messages.map { $0.id }
    }
}
