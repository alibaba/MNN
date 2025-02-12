
//
//  ChatHistory.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import ExyteChat

struct ChatHistory: Codable, Identifiable {
    let id: String
    let modelId: String
    let modelName: String
    var messages: [HistoryMessage]
    let createdAt: Date
    var updatedAt: Date
}

struct HistoryMessage: Codable {
    let id: String
    let content: String
    let images: [LLMChatImage]?
    let audio: Recording?
    let isUser: Bool
    let createdAt: Date
}
