//
//  ModelListViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import Foundation

@MainActor
class ModelListViewModel: ObservableObject {
    @Published private(set) var models: [ModelInfo] = []
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
                fetchedModels[i].isDownloaded = ModelStorageManager.shared.isModelDownloaded(fetchedModels[i].modelId)
            }
            // 多模型置顶顺序，最新置顶的在最顶部
            let pinned = pinnedModelIds.reversed().compactMap { id in
                fetchedModels.first(where: { $0.modelId == id })
            }
            let rest = fetchedModels.filter { !pinnedModelIds.contains($0.modelId) }
            models = pinned + rest
        } catch {
            showError = true
            errorMessage = "Error: \(error.localizedDescription)"
        }
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
        Task(priority: .background) {
            do {
                try await modelClient.downloadModel(model: model) { progress in
                    Task { @MainActor in
                        DispatchQueue.main.async {
                            self.downloadProgress[model.modelId] = progress
                        }
                    }
                }
                
                if let index = models.firstIndex(where: { $0.modelId == model.modelId }) {
                    models[index].isDownloaded = true
                    DispatchQueue.main.async {
                        ModelStorageManager.shared.markModelAsDownloaded(model.modelId)
                    }
                }
            } catch {
                showError = true
                errorMessage = "Failed to download model: \(error.localizedDescription)"
            }
        
            currentlyDownloading = nil
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
            
            // 获取模型目录下的所有文件
            if let files = try? fileManager.contentsOfDirectory(
                at: modelPath,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) {
                // 清理每个文件的下载状态
                let storage = ModelDownloadStorage()
                for file in files {
                    storage.clearFileStatus(at: file.path)
                }
            }
            
            // 删除文件
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
