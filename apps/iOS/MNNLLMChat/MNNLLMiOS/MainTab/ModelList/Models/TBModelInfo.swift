//
//  TBModelInfo.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Hub
import Foundation

struct TBModelInfo: Codable {
    let modelName: String
    let tags: [String]
    let categories: [String]?
    let size_gb: Double?
    let vendor: String?
    let sources: [String: String]?
    let tagTranslations: [String: [String]]?
    
    // 运行时属性
    var isDownloaded: Bool = false
    var lastUsedAt: Date?
    var cachedSize: Int64? = nil
    
    // 统一的ID属性，根据当前选择的源获取对应的modelId
    var id: String {
        guard let sources = sources else {
            return "taobao-mnn/\(modelName)"
        }
        
        let sourceKey = ModelSourceManager.shared.selectedSource.rawValue
        return sources[sourceKey] ?? "taobao-mnn/\(modelName)"
    }
    
    // 本地化的标签 - 使用模型自己的tagTranslations
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
    
    var localPath: String {
        return HubApi.shared.localRepoLocation(HubApi.Repo.init(id: id)).path
    }
    
    var formattedSize: String {
        if isDownloaded {
            return formatLocalSize()
        } else if let cached = cachedSize {
            return formatBytes(cached)
        } else if let sizeGb = size_gb {
            return String(format: "%.1f GB", sizeGb)
        } else {
            return "计算中..."
        }
    }
    
    func fetchRemoteSize() async -> Int64? {
        let modelScopeId = id.replacingOccurrences(of: "taobao-mnn", with: "MNN")
        
        do {
            let files = try await fetchFileList(repoPath: modelScopeId, root: "", revision: "")
            let totalSize = try await calculateTotalSize(files: files, repoPath: modelScopeId)
            return totalSize
        } catch {
            print("Error fetching remote size for \(id): \(error)")
            return nil
        }
    }
    
    private func formatLocalSize() -> String {
        let path = localPath
        guard FileManager.default.fileExists(atPath: path) else { return "未知" }
        
        do {
            let totalSize = try calculateDirectorySize(at: path)
            return formatBytes(totalSize)
        } catch {
            return "未知"
        }
    }
    
    private func calculateDirectorySize(at path: String) throws -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        let enumerator = fileManager.enumerator(atPath: path)
        while let fileName = enumerator?.nextObject() as? String {
            let filePath = (path as NSString).appendingPathComponent(fileName)
            let attributes = try fileManager.attributesOfItem(atPath: filePath)
            if let fileSize = attributes[.size] as? Int64 {
                totalSize += fileSize
            }
        }
        
        return totalSize
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
    
    // MARK: - 云端文件大小计算方法
    
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
    
    private enum CodingKeys: String, CodingKey {
        case modelName
        case tags
        case categories
        case size_gb
        case vendor
        case sources
        case tagTranslations
        case cachedSize
    }
}

// 更新TagTranslationManager以支持单个标签翻译
//extension TagTranslationManager {
//    func getTranslation(for tag: String) -> String? {
//        return globalTagTranslations[tag]
//    }
//}
