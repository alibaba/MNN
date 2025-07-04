//
//  TBModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Hub
import Foundation

class TBModelClient {
    private let baseMirrorURL = "https://hf-mirror.com"
    private let baseURL = "https://huggingface.co"
    private let maxRetries = 5
    
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
    
    func getModelList() async throws -> [TBModelInfo] {
        guard let url = Bundle.main.url(forResource: "mock", withExtension: "json") else {
            throw NetworkError.invalidData
        }
        
        let data = try Data(contentsOf: url)
        let mockResponse = try JSONDecoder().decode(TBMockDataResponse.self, from: data)
        
        // 加载全局标签翻译
        TagTranslationManager.shared.loadTagTranslations(mockResponse.tagTranslations)
        
        return mockResponse.models
    }
    
    func downloadModel(model: TBModelInfo,
                       progress: @escaping (Double) -> Void) async throws {
        switch ModelSourceManager.shared.selectedSource {
        case .modelScope, .modeler:
            try await downloadFromModelScope(model, progress: progress)
        case .huggingFace:
            try await downloadFromHuggingFace(model, progress: progress)
        }
    }
    
    func cancelDownload() async {
        if let manager = currentDownloadManager {
            await manager.cancelDownload()
            currentDownloadManager = nil
            print("Download cancelled")
        }
    }
    
    private func downloadFromModelScope(_ model: TBModelInfo,
                                        progress: @escaping (Double) -> Void) async throws {
        let ModelScopeId = model.id.replacingOccurrences(of: "taobao-mnn", with: "MNN")
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        
        let manager = ModelScopeDownloadManager.init(repoPath: ModelScopeId, config: config, enableLogging: true, source: ModelSourceManager.shared.selectedSource)
        currentDownloadManager = manager
        
        try await manager.downloadModel(to:"huggingface/models/taobao-mnn", modelId: ModelScopeId, modelName: model.modelName) { fileProgress in
            progress(fileProgress)
        }
        
        currentDownloadManager = nil
    }

    private func downloadFromHuggingFace(_ model: TBModelInfo,
                                         progress: @escaping (Double) -> Void) async throws {
        let repo = Hub.Repo(id: model.id)
        let modelFiles = ["*.*"]
        let mirrorHubApi = HubApi(endpoint: baseURL)
        try await mirrorHubApi.snapshot(from: repo, matching: modelFiles) { fileProgress in
            progress(fileProgress.fractionCompleted)
        }
    }
}

struct TBMockDataResponse: Codable {
    let tagTranslations: [String: String]
    let quickFilterTags: [String]?
    let models: [TBModelInfo]
    let metadata: MockMetadata?
    
    struct MockMetadata: Codable {
        let version: String
        let lastUpdated: String
        let schemaVersion: String
        let totalModels: Int
        let supportedPlatforms: [String]
        let minAppVersion: String
    }
}
