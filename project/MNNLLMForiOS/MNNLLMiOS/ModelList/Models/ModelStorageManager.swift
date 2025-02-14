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
    
    private init() {}
    
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
