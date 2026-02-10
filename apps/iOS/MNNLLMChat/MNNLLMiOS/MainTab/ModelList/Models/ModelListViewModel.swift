//
//  ModelListViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Foundation
import SwiftUI
import Combine

class ModelListViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var models: [ModelInfo] = []
    @Published var quickFilterTags: [String] = []
    @Published var selectedModel: ModelInfo?
    @Published var showError = false
    @Published var errorMessage = ""
    
    // Download state
    @Published private(set) var downloadProgress: [String: Double] = [:]
    @Published private(set) var currentlyDownloading: String?
    
    // MARK: - Private Properties
    private let modelClient = ModelClient.shared
    private let pinnedModelKey = "com.mnnllm.pinnedModelIds"
    private var cancellables = Set<AnyCancellable>()
    
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
    
    // MARK: - Initialization
    
    init() {
        Task { @MainActor in
            await fetchModels()
        }
        
        NotificationCenter.default
            .publisher(for: .modelUsageUpdated)
            .sink { [weak self] notification in
                if let modelName = notification.userInfo?["modelName"] as? String {
                    Task { @MainActor in
                        self?.updateModelLastUsed(modelName: modelName)
                    }
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Model Data Management
    
    /// Load models from Bundle root directory (LocalModel folder structure flattened)
    private func loadLocalModels() async -> [ModelInfo] {
        let fileManager = FileManager.default
        var localModels: [ModelInfo] = []
        
        guard let resourcePath = Bundle.main.resourcePath else {
            return localModels
        }
        
        do {
            let contents = try fileManager.contentsOfDirectory(atPath: resourcePath)
            
            // Check if we have model files directly in Bundle root
            let modelFiles = ["config.json", "llm_config.json", "llm.mnn", "tokenizer.txt"]
            let foundModelFiles = contents.filter { modelFiles.contains($0) }
            
            if !foundModelFiles.isEmpty {
                // Check if we have a complete model (at least config.json)
                if foundModelFiles.contains("llm.mnn") {
                    // MARK: Config the Local Model here
                    let modelName = "Qwen3-0.6B-MNN-Inside"
                    let localModel = ModelInfo(
                        modelName: modelName,
                        tags: [
                            // MARK: if you know that model support think, uncomment the line
                             NSLocalizedString("tag.deepThinking", comment: "Deep thinking tag for local model"),
                               NSLocalizedString("tag.localModel", comment: "Local model inside the app")],
                        categories: ["Local Models"],
                        vendor: "Local",
                        sources: ["local": "bundle_root/\(modelName)"],
                        isDownloaded: true
                    )
                    localModels.append(localModel)
                    
                    ModelStorageManager.shared.markModelAsDownloaded(modelName)
                }
            } else {
                // Fallback: try to find LocalModel folder
                let localModelPath = (resourcePath as NSString).appendingPathComponent("LocalModel")
                var isDirectory: ObjCBool = false
                
                if fileManager.fileExists(atPath: localModelPath, isDirectory: &isDirectory), isDirectory.boolValue {
                    localModels.append(contentsOf: await processLocalModelFolder(at: localModelPath))
                }
            }
            
        } catch {
            // Silently handle error
        }
        
        return localModels
    }
    
    /// Process LocalModel folder (fallback for non-flattened structure)
    private func processLocalModelFolder(at validPath: String) async -> [ModelInfo] {
        let fileManager = FileManager.default
        var localModels: [ModelInfo] = []
        
        // Check if this is a valid model directory (contains config.json)
        let configPath = (validPath as NSString).appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: configPath) {
            let modelName = "LocalModel"
            let localModel = ModelInfo(
                modelName: modelName,
                tags: ["local", "bundled"],
                categories: ["Local Models"],
                vendor: "Local",
                sources: ["local": "local/\(modelName)"],
                isDownloaded: true
            )
            localModels.append(localModel)
            
            ModelStorageManager.shared.markModelAsDownloaded(modelName)
        } else {
            // Check for subdirectories that might contain models
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: validPath)
                
                for item in contents {
                    // Skip hidden files and common non-model files
                    if item.hasPrefix(".") || item == "bench.txt" {
                        continue
                    }
                    
                    let itemPath = (validPath as NSString).appendingPathComponent(item)
                    var isItemDirectory: ObjCBool = false
                    
                    if fileManager.fileExists(atPath: itemPath, isDirectory: &isItemDirectory),
                       isItemDirectory.boolValue {
                        
                        let itemConfigPath = (itemPath as NSString).appendingPathComponent("config.json")
                        
                        if fileManager.fileExists(atPath: itemConfigPath) {
                            // Use custom name for Qwen3-0.6B-MNN to avoid conflicts
                            let modelName = item == "Qwen3-0.6B-MNN" ? "Qwen3-0.6B-MNN-Inside" : item
                            let localModel = ModelInfo(
                                modelName: modelName,
                                tags: ["local", "bundled"],
                                categories: ["Local Models"],
                                vendor: "Local",
                                sources: ["local": "local/\(item)"],
                                isDownloaded: true
                            )
                            localModels.append(localModel)
                            
                            ModelStorageManager.shared.markModelAsDownloaded(modelName)
                        }
                    }
                }
            } catch {
                // Silently handle error
            }
        }
        
        return localModels
    }
    
    @MainActor
    func fetchModels() async {
        do {
            let info = try await modelClient.getModelInfo()
            
            self.quickFilterTags = info.quickFilterTags ?? []
            TagTranslationManager.shared.loadTagTranslations(info.tagTranslations)
            
            var fetchedModels = info.models
            
            // Add LocalModel folder models, avoiding duplicates
            let localModels = await loadLocalModels()
            let existingModelNames = Set(fetchedModels.map { $0.modelName })
            let uniqueLocalModels = localModels.filter { !existingModelNames.contains($0.modelName) }
            fetchedModels.append(contentsOf: uniqueLocalModels)
            
            filterDiffusionModels(fetchedModels: &fetchedModels)
            filterModelsForRelease(fetchedModels: &fetchedModels)
            loadCachedSizes(for: &fetchedModels)
            syncDownloadStatus(for: &fetchedModels)
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
    
    private func loadCachedSizes(for models: inout [ModelInfo]) {
        for i in 0..<models.count {
            if let cachedSize = ModelStorageManager.shared.getCachedSize(for: models[i].modelName) {
                models[i].cachedSize = cachedSize
            }
        }
    }
    
    private func syncDownloadStatus(for models: inout [ModelInfo]) {
        for i in 0..<models.count {
            let isDownloaded = ModelStorageManager.shared.isModelDownloaded(models[i].modelName)
            models[i].isDownloaded = isDownloaded
            
            // Also sync last used date
            if let lastUsed = ModelStorageManager.shared.getLastUsed(for: models[i].modelName) {
                models[i].lastUsedAt = lastUsed
            }
        }
    }
    
    private func fetchModelSizes(for models: [ModelInfo]) async {
        await withTaskGroup(of: Void.self) { group in
            for (_, model) in models.enumerated() {
                // Handle undownloaded models - fetch remote size
                if !model.isDownloaded && model.cachedSize == nil && model.file_size == nil {
                    group.addTask {
                        if let size = await model.fetchRemoteSize() {
                            await MainActor.run {
                                if let modelIndex = self.models.firstIndex(where: { $0.id == model.id }) {
                                    self.models[modelIndex].cachedSize = size
                                    ModelStorageManager.shared.setCachedSize(size, for: model.modelName)
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
                                    ModelStorageManager.shared.setCachedSize(localSize, for: model.modelName)
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
    
    private func filterModelsForRelease(fetchedModels: inout [ModelInfo]) {
        #if !DEBUG
        fetchedModels = fetchedModels.filter { model in
            // Filter out models with "MiniCPM" in the name
            if model.modelName.lowercased().contains("minicpm") {
                return false
            }
            
            // Filter out models with size_gb > 8
            if let sizeGB = model.size_gb, sizeGB > 8.0 {
                return false
            }
            
            return true
        }
        #endif
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
    
    @MainActor
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
        Task { @MainActor in
            if let index = self.models.firstIndex(where: { $0.modelName == modelName }) {
                self.models[index].lastUsedAt = Date()
                self.sortModels(fetchedModels: &self.models)
            }
        }
    }
    
    // MARK: - Download Management
    
    func downloadModel(_ model: ModelInfo) async {
        await MainActor.run {
            guard currentlyDownloading == nil else { return }
            currentlyDownloading = model.id
            
            if downloadProgress[model.id] == nil {
                downloadProgress[model.id] = 0
            }
        }
        
        do {
            try await modelClient.downloadModel(model: model) { progress in
                Task { @MainActor in
                    self.downloadProgress[model.id] = progress
                }
            }
            
            await MainActor.run {
                if let index = self.models.firstIndex(where: { $0.id == model.id }) {
                    self.models[index].isDownloaded = true
                    ModelStorageManager.shared.markModelAsDownloaded(model.modelName)
                }
                
                self.downloadProgress.removeValue(forKey: model.id)
                self.currentlyDownloading = nil
            }
            
            // Calculate and cache size for newly downloaded model
            do {
                let localSize = try FileOperationManager.shared.calculateDirectorySize(at: model.localPath)
                await MainActor.run {
                    if let index = self.models.firstIndex(where: { $0.id == model.id }) {
                        self.models[index].cachedSize = localSize
                        ModelStorageManager.shared.setCachedSize(localSize, for: model.modelName)
                    }
                }
            } catch {
                print("Error calculating size for newly downloaded model \(model.modelName): \(error)")
            }
            
        } catch {
            await MainActor.run {
                if case ModelScopeError.downloadCancelled = error {
                    print("Download was cancelled")
                } else {
                    self.downloadProgress.removeValue(forKey: model.id)
                    self.showError = true
                    self.errorMessage = "Failed to download model: \(error.localizedDescription)"
                }
                self.currentlyDownloading = nil
            }
        }
    }
    
    func cancelDownload() async {
        let modelId = await MainActor.run { currentlyDownloading }
        
        if let modelId = modelId {
            await modelClient.cancelDownload(for: modelId)
            
            await MainActor.run {
                self.currentlyDownloading = nil
            }
            
            print("Download cancelled for model: \(modelId)")
        }
    }
    
    // MARK: - Pin Management
    
    @MainActor
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
    
    @MainActor
    func unpinModel(_ model: ModelInfo) {
        var pinnedIds = pinnedModelIds
        if let index = pinnedIds.firstIndex(of: model.id) {
            pinnedIds.remove(at: index)
            pinnedModelIds = pinnedIds
            
            // Re-sort models after unpinning
            sortModels(fetchedModels: &models)
        }
    }
    
    // MARK: - Error Management
    
    @MainActor
    func dismissError() {
        showError = false
        errorMessage = ""
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
            await MainActor.run {
                if let index = self.models.firstIndex(where: { $0.id == model.id }) {
                    self.models[index].isDownloaded = false
                    self.models[index].cachedSize = nil
                    ModelStorageManager.shared.markModelAsNotDownloaded(model.modelName)
                }
                
                // Re-sort models after deletion
                self.sortModels(fetchedModels: &self.models)
            }
            
        } catch {
            await MainActor.run {
                self.showError = true
                self.errorMessage = "Failed to delete model: \(error.localizedDescription)"
            }
        }
    }
    
    @MainActor
    private func updateModelLastUsed(modelName: String) {
        if let index = models.firstIndex(where: { $0.modelName == modelName }) {
            if let lastUsed = ModelStorageManager.shared.getLastUsed(for: modelName) {
                models[index].lastUsedAt = lastUsed
                sortModels(fetchedModels: &models)
            }
        }
    }
}

// MARK: - Notification Names

extension Notification.Name {
    static let modelUsageUpdated = Notification.Name("modelUsageUpdated")
}
