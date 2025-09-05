//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Hub
import Foundation

/// ModelClient - Unified model download and management client
/// 
/// This class provides a centralized interface for downloading models from multiple sources
/// including ModelScope, Modeler, and HuggingFace platforms. It handles source-specific
/// download logic, progress tracking, and error handling.
/// 
/// Key Features:
/// - Multi-platform support (ModelScope, Modeler, HuggingFace)
/// - Progress tracking with throttling to prevent UI stuttering
/// - Automatic fallback to local mock data for development
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
    
    // Debug flag to use local mock data instead of network API
    private let useLocalMockData = false
    
    private var downloadManagers: [String: ModelDownloadManagerProtocol] = [:]
    private let downloadManagerFactory: ModelDownloadManagerFactory
    private let downloadManagerQueue = DispatchQueue(label: "com.mnn.downloadManager", attributes: .concurrent)
    
    private lazy var baseURLString: String = {
        switch ModelSourceManager.shared.selectedSource {
        case .huggingFace:
            return baseURL
        default:
            return baseMirrorURL
        }
    }()
    
    /// Private initializer for singleton pattern
    /// 
    /// - Parameter downloadManagerFactory: Factory for creating download managers.
    ///                                      Defaults to LegacyModelDownloadManagerFactory
    private init(downloadManagerFactory: ModelDownloadManagerFactory = LegacyModelDownloadManagerFactory()) {
        print("ModelClient singleton initialized")
        self.downloadManagerFactory = downloadManagerFactory
    }
    
    deinit {
        print("ModelClient deinit")
        downloadManagers.removeAll()
    }
    
    /// Retrieves model information from the configured API endpoint
    /// 
    /// This method fetches the latest model catalog from the network API.
    /// In debug mode or when network fails, it falls back to local mock data.
    /// 
    /// - Returns: TBDataResponse containing the model catalog
    /// - Throws: NetworkError if both network request and local fallback fail
    func getModelInfo() async throws -> TBDataResponse {
        if useLocalMockData {
            // Debug mode: use local mock data
            guard let url = Bundle.main.url(forResource: "mock", withExtension: "json") else {
                throw NetworkError.invalidData
            }
            
            let data = try Data(contentsOf: url)
            let mockResponse = try JSONDecoder().decode(TBDataResponse.self, from: data)
            return mockResponse
        } else {
            // Production mode: fetch from network API
            return try await fetchDataFromAliAPI()
        }
    }
        
    /// Fetches data from the network API with fallback to local mock data
    ///
    /// - Throws: NetworkError if both network request and local fallback fail
    private func fetchDataFromAliAPI() async throws -> TBDataResponse {
        do {
            guard let url = URL(string: AliCDNURL) else {
                throw NetworkError.invalidData
            }
            
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw NetworkError.invalidResponse
            }
            
            let apiResponse = try JSONDecoder().decode(TBDataResponse.self, from: data)
            return apiResponse
            
        } catch {
            print("Network request failed: \(error). Falling back to local mock data.")
            
            // Fallback to local mock data if network request fails
            guard let url = Bundle.main.url(forResource: "mock", withExtension: "json") else {
                throw NetworkError.invalidData
            }
            
            let data = try Data(contentsOf: url)
            let mockResponse = try JSONDecoder().decode(TBDataResponse.self, from: data)
            return mockResponse
        }
    }
    
    /// Downloads a model from the selected source with progress tracking
    ///
    /// - Parameters:
    ///   - model: The ModelInfo object containing model details
    ///   - progress: Progress callback that receives download progress (0.0 to 1.0)
    /// - Throws: Various network or file system errors
    func downloadModel(model: ModelInfo,
                       progress: @escaping (Double) -> Void) async throws {
        switch ModelSourceManager.shared.selectedSource {
        case .modelScope:
            try await downloadFromModelScope(model, source: .modelScope , progress: progress)
        case .modeler:
            try await downloadFromModelScope(model, source: .modeler, progress: progress)
        case .huggingFace:
            try await downloadFromHuggingFace(model, progress: progress)
        }
    }
    
    /// Cancels download for a specific model
    /// - Parameter modelId: The ID of the model to cancel download for
    func cancelDownload(for modelId: String) async {
        downloadManagerQueue.sync {
            if let downloadManager = downloadManagers[modelId] {
                Task {
                    await downloadManager.cancelDownload()
                }
            }
        }
    }
    
    /// Cancels all active downloads
    func cancelAllDownloads() async {
        let managers = downloadManagerQueue.sync {
            return Array(downloadManagers.values)
        }
        
        await withTaskGroup(of: Void.self) { group in
            for manager in managers {
                group.addTask {
                    await manager.cancelDownload()
                }
            }
        }
    }
    
    /// Downloads model from ModelScope/Modler platform
    ///
    /// - Parameters:
    ///   - model: The ModelInfo object to download
    ///   - progress: Progress callback for download updates
    /// - Throws: Download or network related errors
    private func downloadFromModelScope(_ model: ModelInfo,
                                        source: ModelSource,
                                        progress: @escaping (Double) -> Void) async throws {
        let downloadManager = downloadManagerQueue.sync {
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
            if case ModelScopeError.downloadCancelled = error {
                throw ModelScopeError.downloadCancelled
            } else {
                await cleanupDownloadManager(for: model.id)
                throw NetworkError.downloadFailed
            }
        }
    }
    
    private func cleanupDownloadManager(for modelId: String) async {
        _ = downloadManagerQueue.sync {
            downloadManagers.removeValue(forKey: modelId)
        }
    }
    
    func getActiveDownloadersCount() -> Int {
        return downloadManagerQueue.sync {
            return downloadManagers.count
        }
    }

    /// Downloads model from HuggingFace platform with optimized progress updates
    ///
    /// This method implements throttling to prevent UI stuttering by limiting
    /// progress update frequency and filtering out minor progress changes.
    ///
    /// - Parameters:
    ///   - model: The ModelInfo object to download
    ///   - progress: Progress callback for download updates
    /// - Throws: Download or network related errors
    private func downloadFromHuggingFace(_ model: ModelInfo,
                                         progress: @escaping (Double) -> Void) async throws {
        let repo = Hub.Repo(id: model.id)
        let modelFiles = ["*.*"]
        let mirrorHubApi = HubApi(endpoint: baseURL)
        
        // Progress throttling mechanism to prevent UI stuttering
        var lastUpdateTime = Date()
        var lastProgress: Double = 0.0
        let progressUpdateInterval: TimeInterval = 0.1 // Limit update frequency to every 100ms
        let progressThreshold: Double = 0.001 // Progress change threshold of 0.1%
        
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
               currentProgress == 0.0 {
                
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
