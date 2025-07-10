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
    
    func updateLastUsed(for modelName: String) {
        var models = lastUsedModels
        models[modelName] = Date()
        lastUsedModels = models
    }
    
    func getLastUsed(for modelName: String) -> Date? {
        return lastUsedModels[modelName]
    }
    
    var downloadedModels: [String] {
        get {
            userDefaults.array(forKey: downloadedModelsKey) as? [String] ?? []
        }
        set {
            userDefaults.set(newValue, forKey: downloadedModelsKey)
        }
    }
    
    func clearDownloadStatus(for modelName: String) {
        var models = downloadedModels
        models.removeAll { $0 == modelName }
        downloadedModels = models
    }
    
    func isModelDownloaded(_ modelName: String) -> Bool {
        downloadedModels.contains(modelName)
    }
    
    func markModelAsDownloaded(_ modelName: String) {
        var models = downloadedModels
        if !models.contains(modelName) {
            models.append(modelName)
            downloadedModels = models
        }
    }
}