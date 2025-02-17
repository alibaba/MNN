//
//  LLMChatMessage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import SwiftUI
import ExyteChat

struct LLMChatMessage {
    var uid: String
    var sender: LLMChatUser
    let createdAt: Date
    var status: Message.Status?

    var text: String
    let images: [LLMChatImage]
    let videos: [LLMChatVideo]
    let recording: Recording?
    let replyMessage: ReplyMessage?
}

extension LLMChatMessage {
    func toChatMessage() -> ExyteChat.Message {
        ExyteChat.Message(
            id: uid,
            user: sender.toChatUser(),
            status: status,
            createdAt: createdAt,
            text: text,
            attachments: images.map { $0.toChatAttachment() } + videos.map { $0.toChatAttachment() },
            recording: recording,
            replyMessage: replyMessage
        )
    }
}
