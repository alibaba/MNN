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
    func saveChat(historyId: String, modelId: String, modelName _: String, messages: [Message]) {
        let modelInfo = createFallbackModelInfo(for: modelId)
        saveChat(historyId: historyId, modelInfo: modelInfo, messages: messages)
    }

    /// Create fallback ModelInfo that correctly identifies local models
    private func createFallbackModelInfo(for modelId: String) -> ModelInfo {
        // Check if this modelId corresponds to a local model
        let localModels = ModelInfo.getAvailableLocalModels()

        // Try to find matching local model by name or id
        if let localModel = localModels.first(where: { $0.modelName == modelId || $0.id.contains(modelId) }) {
            return localModel
        }

        // Fallback to downloaded model
        return ModelInfo(modelId: modelId, isDownloaded: true)
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
