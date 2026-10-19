
//
//  ChatHistory.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import Foundation
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
        return modelInfo.name
    }
}

struct HistoryMessage: Codable, Hashable {
    let id: String
    let content: String
    var images: [LLMChatImage]?
    let audio: Recording?
    let isUser: Bool
    let createdAt: Date
    // Optional fields keep older saved conversations readable.
    var thinkText: String? = nil
    var role: String? = nil
    var performanceData: String? = nil
    var useMarkdown: Bool? = nil
    var toolSteps: [ToolStep]? = nil
}
