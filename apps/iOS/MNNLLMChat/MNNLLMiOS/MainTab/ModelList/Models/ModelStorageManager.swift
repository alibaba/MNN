//
//  ModelStorageManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/10.
//

import Foundation

class ModelStorageManager {
    static let shared = ModelStorageManager()
    
    private let userDefaults = UserDefaults.standard
    private let downloadedModelsKey = "com.mnnllm.downloadedModels"
    private let lastUsedModelKey = "com.mnnllm.lastUsedModels"
    
    private init() {}
    
    var lastUsedModels: [String: Date] {
        get {
            userDefaults.dictionary(forKey: lastUsedModelKey) as? [String: Date] ?? [:]
        }
        set {
            userDefaults.set(newValue, forKey: lastUsedModelKey)
        }
    }
    
    func updateLastUsed(for modelId: String) {
        var models = lastUsedModels
        models[modelId] = Date()
        lastUsedModels = models
    }
    
    func getLastUsed(for modelId: String) -> Date? {
        return lastUsedModels[modelId]
    }
    
    var downloadedModels: [String] {
        get {
            userDefaults.array(forKey: downloadedModelsKey) as? [String] ?? []
        }
        set {
            userDefaults.set(newValue, forKey: downloadedModelsKey)
        }
    }
    
    func clearDownloadStatus(for modelId: String) {
        var models = downloadedModels
        models.removeAll { $0 == modelId }
        downloadedModels = models
    }
    
    func isModelDownloaded(_ modelId: String) -> Bool {
        downloadedModels.contains(modelId)
    }
    
    func markModelAsDownloaded(_ modelId: String) {
        var models = downloadedModels
        if !models.contains(modelId) {
            models.append(modelId)
            downloadedModels = models
        }
    }
}
