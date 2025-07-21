//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Hub
import Foundation

class ModelClient {
    private let maxRetries = 5
    
    private let baseMirrorURL = "https://hf-mirror.com"
    private let baseURL = "https://huggingface.co"
    private let AliCDNURL = "https://meta.alicdn.com/data/mnn/apis/model_market.json"
    
    // Debug flag to use local mock data instead of network API
    private let useLocalMockData = false
    
    private var currentDownloadManager: ModelScopeDownloadManager?
    
    private lazy var baseURLString: String = {
        switch ModelSourceManager.shared.selectedSource {
        case .huggingFace:
            return baseURL
        default:
            return baseMirrorURL
        }
    }()
    
    init() {}
    
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
        
    /**
     * Fetches data from the network API with fallback to local mock data
     *
     * @throws NetworkError if both network request and local fallback fail
     */
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
    
    /**
     * Downloads a model from the selected source with progress tracking
     *
     * @param model The ModelInfo object containing model details
     * @param progress Progress callback that receives download progress (0.0 to 1.0)
     * @throws Various network or file system errors
     */
    func downloadModel(model: ModelInfo,
                       progress: @escaping (Double) -> Void) async throws {
        switch ModelSourceManager.shared.selectedSource {
        case .modelScope, .modeler:
            try await downloadFromModelScope(model, progress: progress)
        case .huggingFace:
            try await downloadFromHuggingFace(model, progress: progress)
        }
    }
    
    /**
     * Cancels the current download operation
     */
    func cancelDownload() async {
        if let manager = currentDownloadManager {
            await manager.cancelDownload()
            currentDownloadManager = nil
            print("Download cancelled")
        }
    }
    /**
     * Downloads model from ModelScope platform
     *
     * @param model The ModelInfo object to download
     * @param progress Progress callback for download updates
     * @throws Download or network related errors
     */
    private func downloadFromModelScope(_ model: ModelInfo,
                                        progress: @escaping (Double) -> Void) async throws {
        let ModelScopeId = model.id
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        
        let manager = ModelScopeDownloadManager.init(repoPath: ModelScopeId, config: config, enableLogging: true, source: ModelSourceManager.shared.selectedSource)
        currentDownloadManager = manager
        
        try await manager.downloadModel(to:"huggingface/models/taobao-mnn", modelId: ModelScopeId, modelName: model.modelName) { fileProgress in
            Task { @MainActor in
                progress(fileProgress)
            }
        }
        
        currentDownloadManager = nil
    }

    /**
     * Downloads model from HuggingFace platform with optimized progress updates
     *
     * This method implements throttling to prevent UI stuttering by limiting
     * progress update frequency and filtering out minor progress changes.
     *
     * @param model The ModelInfo object to download
     * @param progress Progress callback for download updates
     * @throws Download or network related errors
     */
    private func downloadFromHuggingFace(_ model: ModelInfo,
                                         progress: @escaping (Double) -> Void) async throws {
        let repo = Hub.Repo(id: model.id)
        let modelFiles = ["*.*"]
        let mirrorHubApi = HubApi(endpoint: baseURL)
        
        // Progress throttling mechanism to prevent UI stuttering
        var lastUpdateTime = Date()
        var lastProgress: Double = 0.0
        let progressUpdateInterval: TimeInterval = 0.1 // Limit update frequency to every 100ms
        let progressThreshold: Double = 0.01 // Progress change threshold of 1%
        
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


enum NetworkError: Error {
    case invalidResponse
    case invalidData
    case downloadFailed
    case unknown
}
