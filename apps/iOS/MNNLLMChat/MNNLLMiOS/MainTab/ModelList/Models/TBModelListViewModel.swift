//
//  TBModelListViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Foundation
import SwiftUI

@MainActor
class TBModelListViewModel: ObservableObject {
    @Published var models: [TBModelInfo] = []
    @Published private(set) var downloadProgress: [String: Double] = [:]
    @Published private(set) var currentlyDownloading: String?
    @Published var showError = false
    @Published var errorMessage = ""
    @Published var searchText = ""
    @Published var quickFilterTags: [String] = []
    
    @Published var selectedModel: TBModelInfo?
    
    private let modelClient = TBModelClient()
    private let pinnedModelKey = "com.mnnllm.pinnedModelIds"
    
    public var pinnedModelIds: [String] {
        get { UserDefaults.standard.stringArray(forKey: pinnedModelKey) ?? [] }
        set { UserDefaults.standard.setValue(newValue, forKey: pinnedModelKey) }
    }
    
    // 获取所有可用的标签
    var allTags: [String] {
        let allTags = Set(models.flatMap { $0.tags })
        return Array(allTags)
    }
    
    // 获取所有可用的分类
    var allCategories: [String] {
        let allCategories = Set(models.compactMap { $0.categories }.flatMap { $0 })
        return Array(allCategories)
    }
    
    // 获取所有可用的厂商
    var allVendors: [String] {
        let allVendors = Set(models.compactMap { $0.vendor })
        return Array(allVendors)
    }
    
    var filteredModels: [TBModelInfo] {
        let filteredModels = searchText.isEmpty ? models : models.filter { model in
            model.id.localizedCaseInsensitiveContains(searchText) ||
            model.modelName.localizedCaseInsensitiveContains(searchText) ||
            model.localizedTags.contains { $0.localizedCaseInsensitiveContains(searchText) }
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
                $0.modelName.lowercased().contains("diffusion") 
            }
            
            if hasDiffusionModels {
                fetchedModels = fetchedModels.filter { model in
                    let name = model.modelName.lowercased()
                    let tags = model.tags.map { $0.lowercased() }
                    
                    // only show gpu diffusion
                    if name.contains("diffusion") {
                        return name.contains("gpu") || tags.contains { $0.contains("gpu") }
                    }
                    
                    return true
                }
            }
            
            for i in 0..<fetchedModels.count {
                let modelId = fetchedModels[i].id
                fetchedModels[i].isDownloaded = ModelStorageManager.shared.isModelDownloaded(modelId)
                fetchedModels[i].lastUsedAt = ModelStorageManager.shared.getLastUsed(for: modelId)
            }
            
            // Sort models
            sortModels(fetchedModels: &fetchedModels)
            
            // 加载 quickFilterTags 和 tagTranslations
            if let mockResponse = try? await loadMockResponse() {
                quickFilterTags = mockResponse.quickFilterTags ?? []
                TagTranslationManager.shared.loadTagTranslations(mockResponse.tagTranslations)
            }
            
            // 异步获取未下载模型的大小信息
            Task {
                await fetchModelSizes(for: fetchedModels)
            }
            
        } catch {
            showError = true
            errorMessage = "Error: \(error.localizedDescription)"
        }
    }
    
    private func loadMockResponse() async throws -> TBMockDataResponse? {
        guard let url = Bundle.main.url(forResource: "mock", withExtension: "json") else {
            return nil
        }
        
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TBMockDataResponse.self, from: data)
    }
    
    private func fetchModelSizes(for models: [TBModelInfo]) async {
        await withTaskGroup(of: Void.self) { group in
            for (_, model) in models.enumerated() {
                if !model.isDownloaded && model.cachedSize == nil && model.size_gb == nil {
                    group.addTask {
                        if let size = await model.fetchRemoteSize() {
                            await MainActor.run {
                                // 查找当前模型在实际数组中的索引
                                if let modelIndex = self.models.firstIndex(where: { $0.id == model.id }) {
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
        if let index = models.firstIndex(where: { $0.id == modelId }) {
            models[index].lastUsedAt = Date()
            sortModels(fetchedModels: &models)
        }
    }
    
    private func sortModels(fetchedModels: inout [TBModelInfo]) {
        let pinned = pinnedModelIds
        
        fetchedModels.sort { (model1, model2) -> Bool in
            let isPinned1 = pinned.contains(model1.id)
            let isPinned2 = pinned.contains(model2.id)
            let isDownloading1 = currentlyDownloading == model1.id
            let isDownloading2 = currentlyDownloading == model2.id
            
            // 1. 正在下载的模型优先级最高
            if isDownloading1 != isDownloading2 {
                return isDownloading1
            }
            
            // 2. 置顶的模型次优先级
            if isPinned1 != isPinned2 {
                return isPinned1
            }
            
            // 3. 如果都是置顶的，按置顶时间排序
            if isPinned1 && isPinned2 {
                let index1 = pinned.firstIndex(of: model1.id)!
                let index2 = pinned.firstIndex(of: model2.id)!
                return index1 > index2 // Pinned later comes first
            }
            
            // 4. 非置顶模型按下载状态排序
            if model1.isDownloaded != model2.isDownloaded {
                return model1.isDownloaded
            }
            
            // 5. 如果都已下载，按最后使用时间排序
            if model1.isDownloaded {
                let date1 = model1.lastUsedAt ?? .distantPast
                let date2 = model2.lastUsedAt ?? .distantPast
                return date1 > date2
            }
            
            return false // Keep original order for not-downloaded
        }
        
        models = fetchedModels
    }
    
    func selectModel(_ model: TBModelInfo) {
        if model.isDownloaded {
            selectedModel = model
        } else {
            Task {
                await downloadModel(model)
            }
        }
    }
    
    func downloadModel(_ model: TBModelInfo) async {
        guard currentlyDownloading == nil else { return }
        
        currentlyDownloading = model.id
        downloadProgress[model.id] = 0
        
        do {
            try await modelClient.downloadModel(model: model) { progress in
                self.downloadProgress[model.id] = progress
            }
            
            if let index = models.firstIndex(where: { $0.id == model.id }) {
                models[index].isDownloaded = true
                ModelStorageManager.shared.markModelAsDownloaded(model.id)
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

    func pinModel(_ model: TBModelInfo) {
        guard let index = models.firstIndex(where: { $0.id == model.id }) else { return }
        let pinned = models.remove(at: index)
        models.insert(pinned, at: 0)
        var ids = pinnedModelIds.filter { $0 != model.id }
        ids.append(model.id)
        pinnedModelIds = ids
    }
    
    func unpinModel(_ model: TBModelInfo) {
        guard let index = models.firstIndex(where: { $0.id == model.id }) else { return }
        let unpinned = models.remove(at: index)
        let insertIndex = models.count // 取消置顶后放到未置顶最后
        models.insert(unpinned, at: insertIndex)
        pinnedModelIds = pinnedModelIds.filter { $0 != model.id }
    }
    
    func deleteModel(_ model: TBModelInfo) async {
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
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].isDownloaded = false
                    ModelStorageManager.shared.clearDownloadStatus(for: model.id)
                }
                if selectedModel?.id == model.id {
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