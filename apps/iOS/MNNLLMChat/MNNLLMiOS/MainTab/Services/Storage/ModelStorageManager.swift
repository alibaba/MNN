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
    private let cachedSizesKey = "com.mnnllm.cachedSizes"
    
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
    
    func markModelAsNotDownloaded(_ modelName: String) {
        var models = downloadedModels
        models.removeAll { $0 == modelName }
        downloadedModels = models
        
        // Also clear cached size when model is marked as not downloaded
        clearCachedSize(for: modelName)
    }
    
    // MARK: - Cached Size Management
    
    var cachedSizes: [String: Int64] {
        get {
            userDefaults.dictionary(forKey: cachedSizesKey) as? [String: Int64] ?? [:]
        }
        set {
            userDefaults.set(newValue, forKey: cachedSizesKey)
        }
    }
    
    func getCachedSize(for modelName: String) -> Int64? {
        return cachedSizes[modelName]
    }
    
    func setCachedSize(_ size: Int64, for modelName: String) {
        var sizes = cachedSizes
        sizes[modelName] = size
        cachedSizes = sizes
    }
    
    func clearCachedSize(for modelName: String) {
        var sizes = cachedSizes
        sizes.removeValue(forKey: modelName)
        cachedSizes = sizes
    }
}