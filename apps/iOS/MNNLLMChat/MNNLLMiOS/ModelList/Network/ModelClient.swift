//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import Hub
import Foundation

class ModelClient {
    private let baseMirrorURL = "https://hf-mirror.com"
    private let baseURL = "https://huggingface.co"
    private let maxRetries = 5
    
    private lazy var baseURLString: String = {
        switch ModelSourceManager.shared.selectedSource {
        case .huggingFace:
            return baseURL
        default:
            return baseMirrorURL
        }
    }()
    
    init() {}
    
    func getModelList() async throws -> [ModelInfo] {
        let url = URL(string: "\(baseURLString)/api/models?author=taobao-mnn&limit=100")!
        return try await performRequest(url: url, retries: maxRetries)
    }
    
    func getRepoInfo(repoName: String, revision: String) async throws -> RepoInfo {
        let url = URL(string: "\(baseURLString)/api/models/\(repoName)")!
        return try await performRequest(url: url, retries: maxRetries)
    }

    @MainActor
    func downloadModel(model: ModelInfo,
                       progress: @escaping (Double) -> Void) async throws {
        switch ModelSourceManager.shared.selectedSource {
        case .modelScope, .modeler:
            try await downloadFromModelScope(model, progress: progress)
        case .huggingFace:
            try await downloadFromHuggingFace(model, progress: progress)
        }
    }

    private func downloadFromModelScope(_ model: ModelInfo,
                                        progress: @escaping (Double) -> Void) async throws {
        let ModelScopeId = model.modelId.replacingOccurrences(of: "taobao-mnn", with: "MNN")
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        
        let manager = ModelScopeDownloadManager.init(repoPath: ModelScopeId, config: config, enableLogging: true, source: ModelSourceManager.shared.selectedSource)
        
        try await manager.downloadModel(to:"huggingface/models/taobao-mnn", modelId: ModelScopeId, modelName: model.name) { fileProgress in
            progress(fileProgress)
        }
    }

    private func downloadFromHuggingFace(_ model: ModelInfo,
                                         progress: @escaping (Double) -> Void) async throws {
        let repo = Hub.Repo(id: model.modelId)
        let modelFiles = ["*.*"]
        let mirrorHubApi = HubApi(endpoint: baseURL)
        try await mirrorHubApi.snapshot(from: repo, matching: modelFiles) { fileProgress in
            progress(fileProgress.fractionCompleted)
        }
    }
    
    private func performRequest<T: Decodable>(url: URL, retries: Int = 3) async throws -> T {
        var lastError: Error?
        
        for attempt in 1...retries {
            do {
                var request = URLRequest(url: url)
                request.setValue("application/json", forHTTPHeaderField: "Accept")
                
                let (data, response) = try await URLSession.shared.data(for: request)
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw NetworkError.invalidResponse
                }
                
                if httpResponse.statusCode == 200 {
                    return try JSONDecoder().decode(T.self, from: data)
                }
                
                throw NetworkError.invalidResponse
                
            } catch {
                lastError = error
                if attempt < retries {
                    try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt)) * 1_000_000_000))
                    continue
                }
            }
        }
        
        throw lastError ?? NetworkError.unknown
    }
}

enum NetworkError: Error {
    case invalidResponse
    case invalidData
    case downloadFailed
    case unknown
}
