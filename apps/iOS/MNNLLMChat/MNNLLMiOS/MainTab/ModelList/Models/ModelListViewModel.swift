//
//  ModelListViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import Foundation
import SwiftUI

@MainActor
class ModelListViewModel: ObservableObject {
    @Published var models: [ModelInfo] = []
    @Published private(set) var downloadProgress: [String: Double] = [:]
    @Published private(set) var currentlyDownloading: String?
    @Published var showError = false
    @Published var errorMessage = ""
    @Published var searchText = ""
    
    @Published var selectedModel: ModelInfo?
    
    private let modelClient = ModelClient()
    private let pinnedModelKey = "com.mnnllm.pinnedModelIds"
    
    public var pinnedModelIds: [String] {
        get { UserDefaults.standard.stringArray(forKey: pinnedModelKey) ?? [] }
        set { UserDefaults.standard.setValue(newValue, forKey: pinnedModelKey) }
    }
    
    var filteredModels: [ModelInfo] {
        
        let filteredModels = searchText.isEmpty ? models : models.filter { model in
            model.modelId.localizedCaseInsensitiveContains(searchText) ||
            model.tags.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }
        
        let downloadedModels = filteredModels.filter { $0.isDownloaded }
        let notDownloadedModels = filteredModels.filter { !$0.isDownloaded }
        
        return downloadedModels + notDownloadedModels
    }
    
    init() {
        Task {
            await fetchModels()
        }
    }
    
    func fetchModels() async {
        do {
            var fetchedModels = try await modelClient.getModelList()
            
            let hasDiffusionModels = fetchedModels.contains { 
                $0.name.lowercased().contains("diffusion") 
            }
            
            if hasDiffusionModels {
                fetchedModels = fetchedModels.filter { model in
                    let name = model.name.lowercased()
                    let tags = model.tags.map { $0.lowercased() }
                    
                    // only show gpu diffusion
                    if name.contains("diffusion") {
                        return name.contains("gpu") || tags.contains { $0.contains("gpu") }
                    }
                    
                    return true
                }
            }
            
            for i in 0..<fetchedModels.count {
                let modelId = fetchedModels[i].modelId
                fetchedModels[i].isDownloaded = ModelStorageManager.shared.isModelDownloaded(modelId)
                fetchedModels[i].lastUsedAt = ModelStorageManager.shared.getLastUsed(for: modelId)
            }
            
            // Sort models
            sortModels(fetchedModels: &fetchedModels)
            
            // 异步获取未下载模型的大小信息
            Task {
                await fetchModelSizes(for: fetchedModels)
            }
            
        } catch {
            showError = true
            errorMessage = "Error: \(error.localizedDescription)"
        }
    }
    
    private func fetchModelSizes(for models: [ModelInfo]) async {
        await withTaskGroup(of: Void.self) { group in
            for (_, model) in models.enumerated() {
                if !model.isDownloaded && model.cachedSize == nil {
                    group.addTask {
                        if let size = await model.fetchRemoteSize() {
                            await MainActor.run {
                                // 查找当前模型在实际数组中的索引
                                if let modelIndex = self.models.firstIndex(where: { $0.modelId == model.modelId }) {
                                    self.models[modelIndex].cachedSize = size
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    func recordModelUsage(modelId: String) {
        ModelStorageManager.shared.updateLastUsed(for: modelId)
        if let index = models.firstIndex(where: { $0.modelId == modelId }) {
            models[index].lastUsedAt = Date()
            sortModels(fetchedModels: &models)
        }
    }
    
    private func sortModels(fetchedModels: inout [ModelInfo]) {
        let pinned = pinnedModelIds
        
        fetchedModels.sort { (model1, model2) -> Bool in
            let isPinned1 = pinned.contains(model1.modelId)
            let isPinned2 = pinned.contains(model2.modelId)
            
            if isPinned1 != isPinned2 {
                return isPinned1
            }
            
            if isPinned1 && isPinned2 {
                let index1 = pinned.firstIndex(of: model1.modelId)!
                let index2 = pinned.firstIndex(of: model2.modelId)!
                return index1 > index2 // Pinned later comes first
            }
            
            // Non-pinned models
            if model1.isDownloaded != model2.isDownloaded {
                return model1.isDownloaded
            }
            
            if model1.isDownloaded {
                let date1 = model1.lastUsedAt ?? .distantPast
                let date2 = model2.lastUsedAt ?? .distantPast
                return date1 > date2
            }
            
            return false // Keep original order for not-downloaded
        }
        
        models = fetchedModels
    }
    
    func selectModel(_ model: ModelInfo) {
        if model.isDownloaded {
            selectedModel = model
        } else {
            Task {
                await downloadModel(model)
            }
        }
    }
    
    func downloadModel(_ model: ModelInfo) async {
        guard currentlyDownloading == nil else { return }
        
        currentlyDownloading = model.modelId
        downloadProgress[model.modelId] = 0
        
        do {
            try await modelClient.downloadModel(model: model) { progress in
                Task { @MainActor in
                    self.downloadProgress[model.modelId] = progress
                }
            }
            
            if let index = models.firstIndex(where: { $0.modelId == model.modelId }) {
                models[index].isDownloaded = true
                ModelStorageManager.shared.markModelAsDownloaded(model.modelId)
            }
            
        } catch {
            
            if case ModelScopeError.downloadCancelled = error {
                print("Download was cancelled")
            } else {
                showError = true
                errorMessage = "Failed to download model: \(error.localizedDescription)"
            }
        }
        
        currentlyDownloading = nil
        downloadProgress.removeValue(forKey: model.modelId)
    }
    
    func cancelDownload() async {
        if let modelId = currentlyDownloading {
            await modelClient.cancelDownload()
            
            downloadProgress.removeValue(forKey: modelId)
            currentlyDownloading = nil
            
            print("Download cancelled for model: \(modelId)")
        }
    }

    func pinModel(_ model: ModelInfo) {
        guard let index = models.firstIndex(where: { $0.modelId == model.modelId }) else { return }
        let pinned = models.remove(at: index)
        models.insert(pinned, at: 0)
        var ids = pinnedModelIds.filter { $0 != model.modelId }
        ids.append(model.modelId)
        pinnedModelIds = ids
    }
    
    func unpinModel(_ model: ModelInfo) {
        guard let index = models.firstIndex(where: { $0.modelId == model.modelId }) else { return }
        let unpinned = models.remove(at: index)
        let insertIndex = models.count // 取消置顶后放到未置顶最后
        models.insert(unpinned, at: insertIndex)
        pinnedModelIds = pinnedModelIds.filter { $0 != model.modelId }
    }
    
    func deleteModel(_ model: ModelInfo) async {
        do {
            let fileManager = FileManager.default
            let modelPath = URL.init(filePath: model.localPath)
            
            if let files = try? fileManager.contentsOfDirectory(
                at: modelPath,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) {
                let storage = ModelDownloadStorage()
                for file in files {
                    storage.clearFileStatus(at: file.path)
                }
            }
            
            if fileManager.fileExists(atPath: modelPath.path) {
                try fileManager.removeItem(at: modelPath)
            }
            
            await MainActor.run {
                if let index = models.firstIndex(where: { $0.modelId == model.modelId }) {
                    models[index].isDownloaded = false
                    ModelStorageManager.shared.clearDownloadStatus(for: model.modelId)
                }
                if selectedModel?.modelId == model.modelId {
                    selectedModel = nil
                }
            }
            
        } catch {
            print("Error deleting model: \(error)")
            await MainActor.run {
                self.errorMessage = "Failed to delete model: \(error.localizedDescription)"
                self.showError = true
            }
        }
    }
}
