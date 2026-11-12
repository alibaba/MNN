//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Foundation
import Hub

/// ModelClient - Unified model download and management client
///
/// This class provides a centralized interface for downloading models from multiple sources
/// including ModelScope, Modeler, and HuggingFace platforms. It handles source-specific
/// download logic, progress tracking, and error handling.
///
/// Key Features:
/// - Multi-platform support (ModelScope, Modeler, HuggingFace)
/// - Progress tracking with throttling to prevent UI stuttering
/// - Cached ModelScope catalog for offline use
/// - Dependency injection for download managers
/// - Cancellation support for ongoing downloads
///
/// Architecture:
/// - Uses factory pattern for download manager creation
/// - Implements strategy pattern for different download sources
/// - Provides async/await interface for modern Swift concurrency
///
/// Usage:
/// ```swift
/// let client = ModelClient.shared
/// let models = try await client.getModelInfo()
/// try await client.downloadModel(model: selectedModel) { progress in
///     print("Download progress: \(progress * 100)%")
/// }
/// ```
class ModelClient {
    // MARK: - Singleton

    static let shared = ModelClient()

    // MARK: - Properties

    private let maxRetries = 5

    private let baseMirrorURL = "https://hf-mirror.com"
    private let baseURL = "https://huggingface.co"
    private let AliCDNURL = "https://meta.alicdn.com/data/mnn/apis/model_market.json"


    private var downloadManagers: [String: ModelDownloadManagerProtocol] = [:]
    private let downloadManagerFactory: ModelDownloadManagerFactory
    private let downloadManagerQueue = DispatchQueue(label: "com.mnn.downloadManager", attributes: .concurrent)
    private let inFlightLock = NSLock()
    private var inFlightDownloads: [String: Task<Void, Error>] = [:]

    private lazy var baseURLString: String = {
        switch ModelSourceManager.shared.selectedSource {
        case .huggingFace:
            return baseURL
        default:
            return baseMirrorURL
        }
    }()

    /// The shared instance serves the app; an injected factory supports isolated tests.
    ///
    /// - Parameter downloadManagerFactory: Factory for creating download managers.
    ///                                      Defaults to LegacyModelDownloadManagerFactory
    init(downloadManagerFactory: ModelDownloadManagerFactory = LegacyModelDownloadManagerFactory()) {
        print("ModelClient singleton initialized")
        self.downloadManagerFactory = downloadManagerFactory
    }

    deinit {
        print("ModelClient deinit")
        downloadManagers.removeAll()
    }

    /// ModelScope determines membership and dates. The older index supplies optional
    /// translations, capability tags and verified repository IDs for other download sources.
    func getModelInfo() async throws -> TBDataResponse {
        let cache = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
            .appendingPathComponent("MNNModelMarket", isDirectory: true)
        let market = ModelScopeMarket(session: .shared,
                                      seedURL: Bundle.module.url(forResource: "ModelScopeFormats", withExtension: "json"),
                                      cacheDirectory: cache)
        async let labels = catalogLabels(cache: cache)
        let repositories = try await market.models()
        let curated = await labels
        var byID: [String: ModelInfo] = [:]
        for model in curated?.models ?? [] {
            if let id = model.sources?["ModelScope"] { byID[id] = model }
        }
        let models = repositories.map { repo -> ModelInfo in
            let label = byID[repo.id]
            var sources = label?.sources ?? [:]
            sources["ModelScope"] = repo.id
            let tags = (repo.tasks ?? []) + (repo.tags ?? []).compactMap {
                $0.hasPrefix("custom_tag:") ? String($0.dropFirst("custom_tag:".count)) : nil
            }
            var model = ModelInfo(modelName: repo.name, tags: label?.tags ?? tags,
                                  categories: label?.categories ?? repo.tasks, size_gb: label?.size_gb,
                                  file_size: repo.file_size.map(Double.init) ?? label?.file_size,
                                  vendor: label?.vendor, sources: sources, tagTranslations: label?.tagTranslations)
            model.catalogUpdatedAt = repo.updatedAt
            return model
        }
        return TBDataResponse(tagTranslations: curated?.tagTranslations ?? [:], quickFilterTags: curated?.quickFilterTags,
                              models: models, metadata: nil)
    }

    private func catalogLabels(cache: URL) async -> TBDataResponse? {
        let file = cache.appendingPathComponent("labels.json")
        let request = URLRequest(url: URL(string: AliCDNURL)!, timeoutInterval: 6)
        if let (data, response) = try? await URLSession.shared.data(for: request),
           (response as? HTTPURLResponse)?.statusCode == 200,
           let labels = try? JSONDecoder().decode(TBDataResponse.self, from: data) {
            try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            try? data.write(to: file, options: .atomic)
            return labels
        }
        guard let data = try? Data(contentsOf: file) else { return nil }
        return try? JSONDecoder().decode(TBDataResponse.self, from: data)
    }

    /// Downloads a model from the selected source with progress tracking
    ///
    /// A new download of the same model waits for the previous (paused or
    /// cancelled) attempt to unwind, so two attempts never write the same
    /// temporary files.
    ///
    /// - Parameters:
    ///   - model: The ModelInfo object containing model details
    ///   - progress: Progress callback that receives download progress (0.0 to 1.0)
    /// - Throws: `ModelScopeError.downloadCancelled` when paused/cancelled,
    ///           otherwise the underlying network or file system error
    func downloadModel(model: ModelInfo,
                       progress: @escaping (Double) -> Void) async throws
    {
        let modelId = model.id
        let previous = inFlightLock.withLock { inFlightDownloads[modelId] }
        _ = await previous?.result

        let source = model.downloadSource
        let task = Task<Void, Error> {
            switch source {
            case .modelScope, .modeler:
                try await self.downloadFromModelScope(model, source: source, progress: progress)
            case .huggingFace:
                try await self.downloadFromHuggingFace(model, progress: progress)
            }
        }
        inFlightLock.withLock { inFlightDownloads[modelId] = task }
        defer {
            inFlightLock.withLock {
                if inFlightDownloads[modelId] == task { inFlightDownloads.removeValue(forKey: modelId) }
            }
        }

        do {
            try await task.value
        } catch {
            if task.isCancelled || Self.isCancellation(error) { throw ModelScopeError.downloadCancelled }
            throw error
        }
    }

    /// Stops the download of a model, keeping its temporary files for resume.
    /// - Parameter modelId: The ID of the model to cancel download for
    func cancelDownload(for modelId: String) async {
        let manager = downloadManagerQueue.sync { downloadManagers[modelId] }
        let task = inFlightLock.withLock { inFlightDownloads[modelId] }
        await manager?.cancelDownload()
        task?.cancel()
    }

    /// Stops the download and deletes everything it wrote: temporary chunks
    /// and the partially populated model folder.
    func discardPartialDownload(for model: ModelInfo) async {
        await cancelDownload(for: model.id)
        let task = inFlightLock.withLock { inFlightDownloads[model.id] }
        _ = await task?.result
        await cleanupDownloadManager(for: model.id)

        let fileManager = FileManager.default
        guard let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let downloads = documents.appendingPathComponent(".downloads", isDirectory: true)
        let prefix = "model_\(model.id.stableHash)_"
        for name in (try? fileManager.contentsOfDirectory(atPath: downloads.path)) ?? [] where name.hasPrefix(prefix) {
            try? fileManager.removeItem(at: downloads.appendingPathComponent(name))
        }
        guard model.vendor != "Local", !ModelStorageManager.shared.isModelDownloaded(model.modelName) else { return }
        try? fileManager.removeItem(atPath: model.localPath)
    }

    /// Cancels all active downloads
    func cancelAllDownloads() async {
        let managers = downloadManagerQueue.sync {
            Array(downloadManagers.values)
        }

        await withTaskGroup(of: Void.self) { group in
            for manager in managers {
                group.addTask {
                    await manager.cancelDownload()
                }
            }
        }
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if case ModelScopeError.downloadCancelled = error { return true }
        return (error as? URLError)?.code == .cancelled
    }

    /// Downloads model from ModelScope/Modelers platform
    ///
    /// - Parameters:
    ///   - model: The ModelInfo object to download
    ///   - progress: Progress callback for download updates
    /// - Throws: Download or network related errors
    private func downloadFromModelScope(_ model: ModelInfo,
                                        source _: ModelSource,
                                        progress: @escaping (Double) -> Void) async throws
    {
        let downloadManager = downloadManagerQueue.sync(flags: .barrier) {
            if let existingManager = downloadManagers[model.id] {
                return existingManager
            } else {
                let newManager = downloadManagerFactory.createDownloadManager(
                    repoPath: model.id,
                    source: .modelScope
                )
                downloadManagers[model.id] = newManager
                return newManager
            }
        }

        do {
            try await downloadManager.downloadModel(
                to: "huggingface/models/taobao-mnn",
                modelId: model.id,
                modelName: model.modelName
            ) { fileProgress in
                Task { @MainActor in
                    progress(fileProgress)
                }
            }

            await cleanupDownloadManager(for: model.id)

        } catch {
            // Keep the manager after a pause so resume reuses its state;
            // a failed attempt starts over with a fresh one.
            if !Self.isCancellation(error) {
                await cleanupDownloadManager(for: model.id)
            }
            throw error
        }
    }

    private func cleanupDownloadManager(for modelId: String) async {
        _ = downloadManagerQueue.sync(flags: .barrier) {
            downloadManagers.removeValue(forKey: modelId)
        }
    }

    func getActiveDownloadersCount() -> Int {
        return downloadManagerQueue.sync {
            downloadManagers.count
        }
    }

    /// Downloads model from HuggingFace platform with optimized progress updates
    ///
    /// This method implements throttling to prevent UI stuttering by limiting
    /// progress update frequency and filtering out minor progress changes.
    /// Cancelling the enclosing task stops the Hub snapshot.
    ///
    /// - Parameters:
    ///   - model: The ModelInfo object to download
    ///   - progress: Progress callback for download updates
    /// - Throws: Download or network related errors
    private func downloadFromHuggingFace(_ model: ModelInfo,
                                         progress: @escaping (Double) -> Void) async throws
    {
        let repo = Hub.Repo(id: model.id)
        let modelFiles = ["*.*"]
        let mirrorHubApi = HubApi(endpoint: baseURL)

        // Progress throttling mechanism to prevent UI stuttering
        var lastUpdateTime = Date()
        var lastProgress = 0.0
        let progressUpdateInterval: TimeInterval = 0.1 // Limit update frequency to every 100ms
        let progressThreshold = 0.001 // Progress change threshold of 0.1%

        try await mirrorHubApi.snapshot(from: repo, matching: modelFiles) { fileProgress in
            let currentProgress = fileProgress.fractionCompleted
            let currentTime = Date()

            // Check if progress should be updated
            let timeDiff = currentTime.timeIntervalSince(lastUpdateTime)
            let progressDiff = abs(currentProgress - lastProgress)

            // Update progress if any of these conditions are met:
            // 1. Time interval exceeds threshold
            // 2. Progress change exceeds threshold
            // 3. Progress reaches 100% (download complete)
            // 4. Progress is 0% (download start)
            if timeDiff >= progressUpdateInterval ||
                progressDiff >= progressThreshold ||
                currentProgress >= 1.0 ||
                currentProgress == 0.0
            {
                lastUpdateTime = currentTime
                lastProgress = currentProgress

                // Ensure progress updates are executed on the main thread
                Task { @MainActor in
                    progress(currentProgress)
                }
            }
        }
    }
}

/// NetworkError - Enumeration of network-related errors
///
/// This enum defines the various error conditions that can occur during
/// network operations and model downloads.
///
/// Error Cases:
/// - invalidResponse: HTTP response is invalid or has non-200 status code
/// - invalidData: Data received is corrupted or cannot be decoded
/// - downloadFailed: Download operation failed due to network or file system issues
/// - unknown: Unexpected error occurred during network operation
enum NetworkError: Error {
    case invalidResponse
    case invalidData
    case downloadFailed
    case unknown
}
