//
//  TBModelInfo.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Hub
import Foundation

struct ModelInfo: Codable {
    // MARK: - Properties
    let modelName: String
    let tags: [String]
    let categories: [String]?
    let size_gb: Double?
    let vendor: String?
    let sources: [String: String]?
    let tagTranslations: [String: [String]]?
    
    // Runtime properties
    var isDownloaded: Bool = false
    var lastUsedAt: Date?
    var cachedSize: Int64? = nil
    
    // MARK: - Initialization
    
    init(modelName: String = "",
         tags: [String] = [],
         categories: [String]? = nil,
         size_gb: Double? = nil,
         vendor: String? = nil,
         sources: [String: String]? = nil,
         tagTranslations: [String: [String]]? = nil,
         isDownloaded: Bool = false,
         lastUsedAt: Date? = nil,
         cachedSize: Int64? = nil) {
        
        self.modelName = modelName
        self.tags = tags
        self.categories = categories
        self.size_gb = size_gb
        self.vendor = vendor
        self.sources = sources
        self.tagTranslations = tagTranslations
        self.isDownloaded = isDownloaded
        self.lastUsedAt = lastUsedAt
        self.cachedSize = cachedSize
    }
    
    init(modelId: String, isDownloaded: Bool = true) {
        let modelName = modelId.components(separatedBy: "/").last ?? modelId
        
        self.init(
            modelName: modelName,
            tags: [],
            sources: ["huggingface": modelId],
            isDownloaded: isDownloaded
        )
    }
    
    // MARK: - Model Identity & Localization
    
    var id: String {
        guard let sources = sources else {
            return "taobao-mnn/\(modelName)"
        }
        
        let sourceKey = ModelSourceManager.shared.selectedSource.rawValue
        return sources[sourceKey] ?? "taobao-mnn/\(modelName)"
    }
    
    var localizedTags: [String] {
        let currentLanguage = LanguageManager.shared.currentLanguage
        let isChineseLanguage = currentLanguage == "简体中文"
        
        if isChineseLanguage, let translations = tagTranslations {
            let languageCode = "zh-Hans"
            return translations[languageCode] ?? tags
        } else {
            return tags
        }
    }
    
    // MARK: - File System & Path Management
    
    var localPath: String {
        let modelScopeId = "taobao-mnn/\(modelName)"
        return HubApi.shared.localRepoLocation(HubApi.Repo.init(id: modelScopeId)).path
    }
    
    // MARK: - Size Calculation & Formatting
    
    var formattedSize: String {
        if let cached = cachedSize {
            return FileOperationManager.shared.formatBytes(cached)
        } else if isDownloaded {
            return FileOperationManager.shared.formatLocalDirectorySize(at: localPath)
        } else if let sizeGb = size_gb {
            return String(format: "%.1f GB", sizeGb)
        } else {
            return "None"
        }
    }
    
    /// Calculates and caches the local directory size
    /// - Returns: The formatted size string and updates cachedSize property
    mutating func calculateAndCacheSize() -> String {
        if let cached = cachedSize {
            return FileOperationManager.shared.formatBytes(cached)
        }
        
        if isDownloaded {
            do {
                let sizeInBytes = try FileOperationManager.shared.calculateDirectorySize(at: localPath)
                self.cachedSize = sizeInBytes
                return FileOperationManager.shared.formatBytes(sizeInBytes)
            } catch {
                print("Error calculating directory size: \(error)")
                return "Unknown"
            }
        } else if let sizeGb = size_gb {
            return String(format: "%.1f GB", sizeGb)
        } else {
            return "None"
        }
    }
    
    // MARK: - Remote Size Calculation
    
    func fetchRemoteSize() async -> Int64? {
        let modelScopeId = "taobao-mnn/\(modelName)"

        do {
            let files = try await fetchFileList(repoPath: modelScopeId, root: "", revision: "")
            let totalSize = try await calculateTotalSize(files: files, repoPath: modelScopeId)
            return totalSize
        } catch {
            print("Error fetching remote size for \(id): \(error)")
            return nil
        }
    }
    
    private func fetchFileList(repoPath: String, root: String, revision: String) async throws -> [ModelFile] {
        let url = try buildURL(
            repoPath: repoPath,
            path: "/repo/files",
            queryItems: [
                URLQueryItem(name: "Root", value: root),
                URLQueryItem(name: "Revision", value: revision)
            ]
        )
        
        let (data, response) = try await URLSession.shared.data(from: url)
        try validateResponse(response)
        
        let modelResponse = try JSONDecoder().decode(ModelResponse.self, from: data)
        return modelResponse.data.files
    }
    
    private func calculateTotalSize(files: [ModelFile], repoPath: String) async throws -> Int64 {
        var totalSize: Int64 = 0
        
        for file in files {
            if file.type == "tree" {
                let subFiles = try await fetchFileList(repoPath: repoPath, root: file.path, revision: "")
                totalSize += try await calculateTotalSize(files: subFiles, repoPath: repoPath)
            } else if file.type == "blob" {
                totalSize += Int64(file.size)
            }
        }
        
        return totalSize
    }
    
    // MARK: - Network Utilities
    
    private func buildURL(repoPath: String, path: String, queryItems: [URLQueryItem]) throws -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "modelscope.cn"
        components.path = "/api/v1/models/\(repoPath)\(path)"
        components.queryItems = queryItems
        
        guard let url = components.url else {
            throw ModelScopeError.invalidURL
        }
        return url
    }
    
    private func validateResponse(_ response: URLResponse) throws {
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw ModelScopeError.invalidResponse
        }
    }
    
    // MARK: - Codable
    
    private enum CodingKeys: String, CodingKey {
        case modelName, tags, categories, size_gb, vendor, sources, tagTranslations, cachedSize
    }
}
