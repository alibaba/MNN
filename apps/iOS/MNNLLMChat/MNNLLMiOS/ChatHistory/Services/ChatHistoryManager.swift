//
//  ChatHistoryManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import ExyteChat

class ChatHistoryManager {
    static let shared = ChatHistoryManager()
    
    private init() {}
    
    func saveChat(historyId: String, modelInfo: ModelInfo, messages: [Message]) {
        ChatHistoryDatabase.shared?.saveChat(
            historyId: historyId,
            modelInfo: modelInfo,
            messages: messages
        )
    }
    
    // For backward compatibility
    func saveChat(historyId: String, modelId: String, modelName: String, messages: [Message]) {
        let modelInfo = ModelInfo(modelId: modelId, isDownloaded: true)
        saveChat(historyId: historyId, modelInfo: modelInfo, messages: messages)
    }
    
    func getAllHistory() -> [ChatHistory] {
        return ChatHistoryDatabase.shared?.getAllHistory() ?? []
    }
    
    func getHistory(for modelId: String) -> [ChatHistory] {
        let allHistories = getAllHistory()
        let filtered = allHistories.filter { $0.modelId == modelId }
        print("Found \(filtered.count) histories for modelId: \(modelId)")
        return filtered
    }
    
    func deleteHistory(_ history: ChatHistory) {
        ChatHistoryDatabase.shared?.deleteHistory(history)
    }
}
