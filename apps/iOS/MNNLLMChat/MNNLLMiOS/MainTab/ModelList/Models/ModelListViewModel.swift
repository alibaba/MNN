//
//  ModelListViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Foundation
import SwiftUI

@MainActor
class ModelListViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var models: [ModelInfo] = []
    @Published var searchText = ""
    @Published var quickFilterTags: [String] = []
    @Published var selectedModel: ModelInfo?
    @Published var showError = false
    @Published var errorMessage = ""
    
    // Download state
    @Published private(set) var downloadProgress: [String: Double] = [:]
    @Published private(set) var currentlyDownloading: String?
    
    // MARK: - Private Properties
    private let modelClient = ModelClient()
    private let pinnedModelKey = "com.mnnllm.pinnedModelIds"
    
    // MARK: - Model Data Access
    
    public var pinnedModelIds: [String] {
        get { UserDefaults.standard.stringArray(forKey: pinnedModelKey) ?? [] }
        set { UserDefaults.standard.setValue(newValue, forKey: pinnedModelKey) }
    }
    
    var allTags: [String] {
        Array(Set(models.flatMap { $0.tags }))
    }
    
    var allCategories: [String] {
        Array(Set(models.compactMap { $0.categories }.flatMap { $0 }))
    }
    
    var allVendors: [String] {
        Array(Set(models.compactMap { $0.vendor }))
    }
    
    var filteredModels: [ModelInfo] {
        let filtered = searchText.isEmpty ? models : models.filter { model in
            model.id.localizedCaseInsensitiveContains(searchText) ||
            model.modelName.localizedCaseInsensitiveContains(searchText) ||
            model.localizedTags.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }
        
        let downloaded = filtered.filter { $0.isDownloaded }
        let notDownloaded = filtered.filter { !$0.isDownloaded }
        
        return downloaded + notDownloaded
    }
    
    // MARK: - Initialization
    
    init() {
        Task {
            await fetchModels()
        }
    }
    
    // MARK: - Model Data Management
    
    func fetchModels() async {
        do {
            let info = try await modelClient.getModelInfo()
            
            self.quickFilterTags = info.quickFilterTags ?? []
            TagTranslationManager.shared.loadTagTranslations(info.tagTranslations)
            
            var fetchedModels = info.models
            
            filterDiffusionModels(fetchedModels: &fetchedModels)
            sortModels(fetchedModels: &fetchedModels)
            self.models = fetchedModels
            
            // Asynchronously fetch size info for both downloaded and undownloaded models
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
                // Handle undownloaded models - fetch remote size
                if !model.isDownloaded && model.cachedSize == nil && model.size_gb == nil {
                    group.addTask {
                        if let size = await model.fetchRemoteSize() {
                            await MainActor.run {
                                if let modelIndex = self.models.firstIndex(where: { $0.id == model.id }) {
                                    self.models[modelIndex].cachedSize = size
                                }
                            }
                        }
                    }
                }
                
                // Handle downloaded models - calculate and cache local directory size
                if model.isDownloaded && model.cachedSize == nil {
                    group.addTask {
                        do {
                            let localSize = try FileOperationManager.shared.calculateDirectorySize(at: model.localPath)
                            await MainActor.run {
                                if let modelIndex = self.models.firstIndex(where: { $0.id == model.id }) {
                                    self.models[modelIndex].cachedSize = localSize
                                }
                            }
                        } catch {
                            print("Error calculating local directory size for \(model.modelName): \(error)")
                        }
                    }
                }
            }
        }
    }
    
    private func filterDiffusionModels(fetchedModels: inout [ModelInfo]) {
        let hasDiffusionModels = fetchedModels.contains {
            $0.modelName.lowercased().contains("diffusion")
        }
        
        if hasDiffusionModels {
            fetchedModels = fetchedModels.filter { model in
                let name = model.modelName.lowercased()
                let tags = model.tags.map { $0.lowercased() }
                
                // Only show GPU diffusion models
                if name.contains("diffusion") {
                    return name.contains("gpu") || tags.contains { $0.contains("gpu") }
                }
                
                return true
            }
        }
        
        for i in 0..<fetchedModels.count {
            let model = fetchedModels[i]
            fetchedModels[i].isDownloaded = ModelStorageManager.shared.isModelDownloaded(model.modelName)
            fetchedModels[i].lastUsedAt = ModelStorageManager.shared.getLastUsed(for: model.modelName)
        }
    }
    
    private func sortModels(fetchedModels: inout [ModelInfo]) {
        let pinned = pinnedModelIds
        
        fetchedModels.sort { (model1, model2) -> Bool in
            let isPinned1 = pinned.contains(model1.id)
            let isPinned2 = pinned.contains(model2.id)
            let isDownloading1 = currentlyDownloading == model1.id
            let isDownloading2 = currentlyDownloading == model2.id
            
            // 1. Currently downloading models have highest priority
            if isDownloading1 != isDownloading2 {
                return isDownloading1
            }
            
            // 2. Pinned models have second priority
            if isPinned1 != isPinned2 {
                return isPinned1
            }
            
            // 3. If both are pinned, sort by pin time
            if isPinned1 && isPinned2 {
                let index1 = pinned.firstIndex(of: model1.id)!
                let index2 = pinned.firstIndex(of: model2.id)!
                return index1 > index2 // Pinned later comes first
            }
            
            // 4. Non-pinned models sorted by download status
            if model1.isDownloaded != model2.isDownloaded {
                return model1.isDownloaded
            }
            
            // 5. If both downloaded, sort by last used time
            if model1.isDownloaded {
                let date1 = model1.lastUsedAt ?? .distantPast
                let date2 = model2.lastUsedAt ?? .distantPast
                return date1 > date2
            }
            
            return false // Keep original order for not-downloaded
        }
    }
    
    // MARK: - Model Selection & Usage
    
    func selectModel(_ model: ModelInfo) {
        if model.isDownloaded {
            selectedModel = model
        } else {
            Task {
                await downloadModel(model)
            }
        }
    }
    
    func recordModelUsage(modelName: String) {
        ModelStorageManager.shared.updateLastUsed(for: modelName)
        if let index = models.firstIndex(where: { $0.modelName == modelName }) {
            models[index].lastUsedAt = Date()
            sortModels(fetchedModels: &models)
        }
    }
    
    // MARK: - Download Management
    
    func downloadModel(_ model: ModelInfo) async {
        guard currentlyDownloading == nil else { return }
        
        currentlyDownloading = model.id
        downloadProgress[model.id] = 0
        
        do {
            try await modelClient.downloadModel(model: model) { progress in
                Task { @MainActor in
                    self.downloadProgress[model.id] = progress
                }
            }
            
            if let index = models.firstIndex(where: { $0.id == model.id }) {
                models[index].isDownloaded = true
                ModelStorageManager.shared.markModelAsDownloaded(model.modelName)
                
                // Calculate and cache size for newly downloaded model
                Task {
                    do {
                        let localSize = try FileOperationManager.shared.calculateDirectorySize(at: model.localPath)
                        await MainActor.run {
                            self.models[index].cachedSize = localSize
                        }
                    } catch {
                        print("Error calculating size for newly downloaded model \(model.modelName): \(error)")
                    }
                }
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
        downloadProgress.removeValue(forKey: model.id)
    }
    
    func cancelDownload() async {
        if let modelId = currentlyDownloading {
            await modelClient.cancelDownload()
            
            downloadProgress.removeValue(forKey: modelId)
            currentlyDownloading = nil
            
            print("Download cancelled for model: \(modelId)")
        }
    }
    
    // MARK: - Pin Management
    
    func pinModel(_ model: ModelInfo) {
        guard let index = models.firstIndex(where: { $0.id == model.id }) else { return }
        let pinned = models.remove(at: index)
        models.insert(pinned, at: 0)
        
        var pinnedIds = pinnedModelIds
        if let existingIndex = pinnedIds.firstIndex(of: model.id) {
            pinnedIds.remove(at: existingIndex)
        }
        pinnedIds.insert(model.id, at: 0)
        pinnedModelIds = pinnedIds
    }
    
    func unpinModel(_ model: ModelInfo) {
        var pinnedIds = pinnedModelIds
        if let index = pinnedIds.firstIndex(of: model.id) {
            pinnedIds.remove(at: index)
            pinnedModelIds = pinnedIds
            
            // Re-sort models after unpinning
            sortModels(fetchedModels: &models)
        }
    }
    
    // MARK: - Model Deletion
    
    func deleteModel(_ model: ModelInfo) async {
        guard model.isDownloaded else { return }
        
        do {
            // Delete local files
            let fileManager = FileManager.default
            let modelPath = model.localPath
            
            if fileManager.fileExists(atPath: modelPath) {
                try fileManager.removeItem(atPath: modelPath)
            }
            
            // Update model state
            if let index = models.firstIndex(where: { $0.id == model.id }) {
                models[index].isDownloaded = false
                models[index].cachedSize = nil // Clear cached size since model is deleted
                ModelStorageManager.shared.markModelAsDownloaded(model.modelName)
            }
            
            // Re-sort models after deletion
            sortModels(fetchedModels: &models)
            
        } catch {
            showError = true
            errorMessage = "Failed to delete model: \(error.localizedDescription)"
        }
    }
}
