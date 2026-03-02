
//
//  ChatHistory.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import ExyteChat

struct ChatHistory: Codable, Identifiable, Hashable {
    let id: String
    let modelInfo: ModelInfo
    var messages: [HistoryMessage]
    let createdAt: Date
    var updatedAt: Date
    
    // For backward compatibility, provide convenient properties
    var modelId: String {
        return modelInfo.id
    }
    
    var modelName: String {
        return modelInfo.modelName
    }
}

struct HistoryMessage: Codable, Hashable {
    let id: String
    let content: String
    let images: [LLMChatImage]?
    let audio: Recording?
    let isUser: Bool
    let createdAt: Date
}
