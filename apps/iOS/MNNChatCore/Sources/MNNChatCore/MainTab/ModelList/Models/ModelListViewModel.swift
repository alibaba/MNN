//
//  ModelListViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Combine
import Foundation
import SwiftUI

/// Download state of one catalog entry, as shown by the row controls.
enum ModelDownloadState: Equatable {
    case idle
    case downloading(Double)
    case paused(Double)
    case failed(String, progress: Double)
    case downloaded
}

/// Stopped-download bookkeeping; a running download is `currentlyDownloading`.
enum DownloadPhase: Equatable {
    case paused
    case failed(String)
}

class ModelListViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var models: [ModelInfo] = []
    @Published var quickFilterTags: [String] = []
    @Published var selectedModel: ModelInfo?
    @Published var showError = false
    @Published var errorMessage = ""
    @Published private(set) var isLoadingModels = true

    // Download state
    @Published private(set) var downloadProgress: [String: Double] = [:]
    @Published private(set) var currentlyDownloading: String?
    @Published private(set) var downloadPhases: [String: DownloadPhase] = [:]
    /// Smoothed bytes per second of the running download.
    @Published private(set) var downloadSpeed: Double?

    // MARK: - Private Properties

    // Temporarily hide the SentencePiece market packages audited on 2026-10-12.
    private static let hiddenSentencePieceModels: Set<String> = [
        "Baichuan2-7B-Chat-MNN",
        "ERNIE-4.5-0.3B-PT-MNN",
        "Llama-2-7b-chat-ms-MNN",
        "MiniCPM-V-4-MNN",
        "MiniCPM4-0.5B-MNN",
        "MiniCPM4-8B-MNN",
        "MobileLLM-125M-MNN",
        "MobileLLM-1B-MNN",
        "MobileLLM-350M-MNN",
        "MobileLLM-600M-MNN",
        "Nanbeige4-3B-Thinking-2511-MNN",
        "TinyLlama-1.1B-Chat-MNN",
        "TinyLlama-1.1B-Chat-v1.0-MNN",
        "Yi-6B-Chat-MNN",
        "chatglm-6b-MNN",
        "chatglm2-6b-MNN",
        "chatglm3-6b-MNN",
        "codegeex2-6b-MNN",
        "gemma-2-2b-it-MNN",
        "gemma-2-9b-it-MNN",
        "gemma-3-1b-it-qat-q4_0-gguf-MNN",
        "gemma-3-4b-it-q4_0-mnn",
        "gemma-7b-it-MNN",
        "internlm-chat-7b-MNN",
    ]

    private let modelClient: ModelClient
    private let pinnedModelKey = "com.mnnllm.pinnedModelIds"
    private var cancellables = Set<AnyCancellable>()
    private var isFetchingModels = false
    private var downloadGeneration = UUID()
    /// Lets the chat unload a model before its files go away. Returning
    /// false (e.g. a reply is still generating) cancels the deletion.
    var prepareModelDeletion: (@MainActor (ModelInfo) -> Bool)?
    private var downloadStartedAt: Date?
    private var speedSample: (time: Date, progress: Double)?

    // MARK: - Model Data Access

    var pinnedModelIds: [String] {
        get { UserDefaults.standard.stringArray(forKey: pinnedModelKey) ?? [] }
        set { UserDefaults.standard.setValue(newValue, forKey: pinnedModelKey) }
    }

    var allTags: [String] {
        Array(Set(models.flatMap(\.tags)))
    }

    var allCategories: [String] {
        Array(Set(models.compactMap(\.categories).flatMap { $0 }))
    }

    var allVendors: [String] {
        Array(Set(models.compactMap(\.vendor)))
    }

    // MARK: - Initialization

    init(modelClient: ModelClient = .shared, fetchOnInit: Bool = true) {
        self.modelClient = modelClient
        if fetchOnInit {
            Task { @MainActor in
                await fetchModels()
            }
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

    /// Load models from Bundle root directory and LocalModel folder
    private func loadLocalModels() async -> [ModelInfo] {
        var localModels: [ModelInfo] = []

        // First, try to load models from ModelIndex.json
        let indexModels = ModelInfo.getAvailableLocalModels()
        localModels.append(contentsOf: indexModels)

        // Mark all index models as downloaded
        for model in indexModels {
            ModelStorageManager.shared.markModelAsDownloaded(model.modelName)
        }

        // Fallback: Load models from Bundle root directory (legacy support)
        let fileManager = FileManager.default
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
                    let modelName = "Qwen3-0.6B-MNN-Built-In"
                    let localModel = ModelInfo(
                        modelName: modelName,
                        tags: [
                            // MARK: if you know that model support think, uncomment the line

                            AppText.text("tag.deepThinking", comment: "Deep thinking tag for local model"),
                            AppText.text("tag.localModel", comment: "Local model inside the app"),
                        ],
                        categories: ["Local Models"],
                        vendor: "Local",
                        sources: ["local": "bundle_root/\(modelName)"],
                        isDownloaded: true
                    )
                    localModels.append(localModel)

                    ModelStorageManager.shared.markModelAsDownloaded(modelName)
                }
            }
            
            // Always check LocalModel folder for additional models (e.g., Sana Diffusion)
            let localModelPath = (resourcePath as NSString).appendingPathComponent("LocalModel")
            var isDirectory: ObjCBool = false

            if fileManager.fileExists(atPath: localModelPath, isDirectory: &isDirectory), isDirectory.boolValue {
                localModels.append(contentsOf: await processLocalModelFolder(at: localModelPath))
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

        // Get list of folders already handled by ModelIndex.json to avoid duplicates
        let indexModels = ModelInfo.getAvailableLocalModels()
        let indexModelFolders = Set(indexModels.compactMap { model in
            if let sources = model.sources,
               let localSource = sources["local"],
               localSource.hasPrefix("LocalModel/")
            {
                return String(localSource.dropFirst("LocalModel/".count))
            }
            return nil
        })

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
                    // Skip hidden files, common non-model files, and folders already handled by ModelIndex.json
                    if item.hasPrefix(".") || item == "bench.txt" || indexModelFolders.contains(item) {
                        continue
                    }

                    let itemPath = (validPath as NSString).appendingPathComponent(item)
                    var isItemDirectory: ObjCBool = false

                    if fileManager.fileExists(atPath: itemPath, isDirectory: &isItemDirectory),
                       isItemDirectory.boolValue
                    {
                        // Check if this is a Sana Diffusion model
                        if ModelUtils.isSanaDiffusionModel(atPath: itemPath) {
                            let modelName = item.contains("ghibli") ? "Sana-Ghibli-Style-Transfer" : "Sana-Diffusion-\(item)"
                            let localModel = ModelInfo(
                                modelName: modelName,
                                tags: ["local", "bundled", "sana", "diffusion", "style-transfer"],
                                categories: ["Local Models", "Diffusion"],
                                vendor: "Local",
                                sources: ["local": "local/\(item)"],
                                isDownloaded: true
                            )
                            localModels.append(localModel)
                            ModelStorageManager.shared.markModelAsDownloaded(modelName)
                            continue
                        }

                        let itemConfigPath = (itemPath as NSString).appendingPathComponent("config.json")

                        if fileManager.fileExists(atPath: itemConfigPath) {
                            let modelName = ModelInfo.bundledModelName(for: item)
                            let localModel = ModelInfo(
                                modelName: modelName,
                                tags: ["local", "Build-In", "Think"],
                                categories: ["Local Models"],
                                vendor: "Local",
                                sources: ["local": "LocalModel/\(item)"],
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
        guard !isFetchingModels else { return }
        isFetchingModels = true
        isLoadingModels = true
        defer {
            isLoadingModels = false
            isFetchingModels = false
        }
        // Local models remain usable even when the catalog cannot be reached.
        var localModels = await loadLocalModels()
        for name in ModelStorageManager.shared.downloadedModels {
            guard !localModels.contains(where: { $0.modelName == name }) else { continue }
            let model = ModelInfo(modelId: "taobao-mnn/\(name)")
            if FileManager.default.fileExists(atPath: model.localPath + "/config.json") {
                localModels.append(model)
            }
        }
        localModels.removeAll { Self.hiddenSentencePieceModels.contains($0.modelName) }
        if models.isEmpty {
            var offlineModels = localModels
            loadCachedSizes(for: &offlineModels)
            sortModels(fetchedModels: &offlineModels)
            models = offlineModels
        }
        do {
            let info = try await modelClient.getModelInfo()

            quickFilterTags = info.quickFilterTags ?? []
            TagTranslationManager.shared.loadTagTranslations(info.tagTranslations)

            var fetchedModels = info.models.filter {
                !Self.hiddenSentencePieceModels.contains($0.modelName)
            }

            // Bundled models take precedence; downloaded models keep catalog metadata.
            fetchedModels.removeAll { remoteModel in
                localModels.contains { $0.vendor == "Local" && $0.modelName == remoteModel.modelName }
            }

            let catalogNames = Set(fetchedModels.map(\.modelName))
            fetchedModels.append(contentsOf: localModels.filter { !catalogNames.contains($0.modelName) })

            filterDiffusionModels(fetchedModels: &fetchedModels)
            filterModelsForRelease(fetchedModels: &fetchedModels)
            loadCachedSizes(for: &fetchedModels)
            syncDownloadStatus(for: &fetchedModels)
            sortModels(fetchedModels: &fetchedModels)
            models = fetchedModels

            // Asynchronously fetch size info for both downloaded and undownloaded models
            Task {
                await fetchModelSizes(for: fetchedModels)
            }

        } catch {
            guard !models.contains(where: { $0.isDownloaded }) else { return }
            showError = true
            errorMessage = String(format: AppText.text("error.modelCatalog"), self.modelServiceErrorDescription(error))
        }
    }

    private func modelServiceErrorDescription(_ error: Error) -> String {
        if let serviceError = error as? ModelScopeError {
            switch serviceError {
            case .invalidURL, .invalidResponse, .invalidData:
                return AppText.text("error.modelResponse")
            case .downloadFailed(let underlying):
                return modelServiceErrorDescription(underlying)
            case .fileSystemError:
                return AppText.text("error.modelStorage")
            case .downloadCancelled:
                return AppText.text("error.modelService")
            }
        }
        let nativeError = error as NSError
        if nativeError.domain == NSURLErrorDomain { return AppText.text("error.modelNetwork") }
        if nativeError.domain == NSCocoaErrorDomain && nativeError.code == NSFileWriteOutOfSpaceError {
            return AppText.text("error.modelStorage")
        }
        return AppText.text("error.modelService")
    }

    private func loadCachedSizes(for models: inout [ModelInfo]) {
        for i in 0 ..< models.count {
            if let cachedSize = ModelStorageManager.shared.getCachedSize(for: models[i].modelName) {
                models[i].cachedSize = cachedSize
            }
        }
    }

    private func syncDownloadStatus(for models: inout [ModelInfo]) {
        for i in 0 ..< models.count {
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
                if !model.isDownloaded, model.cachedSize == nil, model.file_size == nil {
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
                if model.isDownloaded, model.cachedSize == nil {
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

        for i in 0 ..< fetchedModels.count {
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

        fetchedModels.sort { model1, model2 -> Bool in
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
            if isPinned1, isPinned2 {
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

    @MainActor
    func renameModel(_ model: ModelInfo, to name: String) {
        ModelNameStore.shared.setName(name, for: model)
    }

    // MARK: - Download Management

    func downloadState(for model: ModelInfo) -> ModelDownloadState {
        if model.isDownloaded { return .downloaded }
        let progress = downloadProgress[model.id] ?? 0
        if currentlyDownloading == model.id { return .downloading(progress) }
        switch downloadPhases[model.id] {
        case .paused: return .paused(progress)
        case let .failed(message): return .failed(message, progress: progress)
        case nil: return .idle
        }
    }

    /// The catalog entry may learn its size after a download started.
    func totalBytes(for model: ModelInfo) -> Int64? {
        models.first { $0.id == model.id }?.expectedBytes ?? model.expectedBytes
    }

    /// Starts, resumes or retries a download. Already-downloaded chunks are
    /// kept on disk, so resuming continues where the last attempt stopped.
    @MainActor
    func downloadModel(_ model: ModelInfo) async {
        guard currentlyDownloading == nil else { return }
        if let message = insufficientSpaceMessage(for: model) {
            downloadPhases[model.id] = .failed(message)
            return
        }
        let generation = UUID()
        downloadGeneration = generation
        currentlyDownloading = model.id
        downloadPhases.removeValue(forKey: model.id)
        downloadSpeed = nil
        speedSample = nil
        downloadStartedAt = Date()

        if downloadProgress[model.id] == nil {
            downloadProgress[model.id] = 0
        }

        do {
            try await modelClient.downloadModel(model: model) { progress in
                Task { @MainActor in
                    guard self.downloadGeneration == generation else { return }
                    self.recordProgress(progress, for: model)
                }
            }
            guard downloadGeneration == generation else { return }

            try ModelConfigManager(modelPath: model.localPath).applyDownloadedModelDefaults()

            await MainActor.run {
                guard self.downloadGeneration == generation else { return }
                if let index = self.models.firstIndex(where: { $0.id == model.id }) {
                    self.models[index].isDownloaded = true
                    ModelStorageManager.shared.markModelAsDownloaded(model.modelName)
                }

                self.downloadProgress.removeValue(forKey: model.id)
                self.downloadPhases.removeValue(forKey: model.id)
                self.currentlyDownloading = nil
                self.downloadSpeed = nil
            }

            // Calculate and cache size for newly downloaded model
            do {
                let modelPath = model.localPath
                let localSize = try await Task.detached(priority: .utility) {
                    try FileOperationManager.shared.calculateDirectorySize(at: modelPath)
                }.value
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
                guard self.downloadGeneration == generation else { return }
                if case ModelScopeError.downloadCancelled = error {
                    self.downloadPhases[model.id] = .paused
                } else {
                    // Shown inline next to the model with a retry button.
                    self.downloadPhases[model.id] = .failed(
                        String(format: AppText.text("error.modelDownload"), self.modelServiceErrorDescription(error)))
                }
                self.currentlyDownloading = nil
                self.downloadSpeed = nil
            }
        }
    }

    /// Stops the current download; its chunks stay on disk for `downloadModel`.
    @MainActor
    func pauseDownload() async {
        guard let modelId = currentlyDownloading else { return }
        downloadGeneration = UUID()
        currentlyDownloading = nil
        downloadSpeed = nil
        downloadPhases[modelId] = .paused
        await modelClient.cancelDownload(for: modelId)
    }

    /// Stops the download (if running) and deletes the partial files.
    @MainActor
    func discardDownload(_ model: ModelInfo) async {
        if currentlyDownloading == model.id {
            downloadGeneration = UUID()
            currentlyDownloading = nil
            downloadSpeed = nil
        }
        downloadPhases.removeValue(forKey: model.id)
        downloadProgress.removeValue(forKey: model.id)
        await modelClient.discardPartialDownload(for: model)
    }

    @MainActor
    private func recordProgress(_ progress: Double, for model: ModelInfo) {
        downloadProgress[model.id] = progress
        guard let total = totalBytes(for: model) else { return }
        let now = Date()
        // Files finished in an earlier attempt are counted in a burst right
        // after (re)starting; keep moving the baseline until that settles.
        guard let startedAt = downloadStartedAt, now.timeIntervalSince(startedAt) >= 1.5,
              let sample = speedSample else {
            speedSample = (now, progress)
            return
        }
        let elapsed = now.timeIntervalSince(sample.time)
        guard elapsed >= 1 else { return }
        let current = max(0, (progress - sample.progress) * Double(total) / elapsed)
        downloadSpeed = downloadSpeed.map { $0 * 0.6 + current * 0.4 } ?? current
        speedSample = (now, progress)
    }

    @MainActor
    private func insufficientSpaceMessage(for model: ModelInfo) -> String? {
        guard let total = totalBytes(for: model), let available = StorageSpace.availableBytes() else { return nil }
        let remaining = Int64(Double(total) * (1 - (downloadProgress[model.id] ?? 0)))
        let needed = remaining + 200 * 1024 * 1024 // headroom for config edits and the OS
        guard available < needed else { return nil }
        return String(format: AppText.text("download.error.space"), ByteText.file(needed), ByteText.file(available))
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
        if let prepare = prepareModelDeletion {
            guard await MainActor.run(body: { prepare(model) }) else { return }
        }

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
                self.errorMessage = String(format: AppText.text("error.modelDelete"), error.localizedDescription)
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
