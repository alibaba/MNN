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
    
    func saveChat(historyId: String, modelId: String, modelName: String, messages: [Message]) {
        ChatHistoryDatabase.shared?.saveChat(
            historyId: historyId,
            modelId: modelId,
            modelName: modelName,
            messages: messages
        )
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
