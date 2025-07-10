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
            return formatBytes(cached)
        } else if isDownloaded {
            return formatLocalSize()
        } else if let sizeGb = size_gb {
            return String(format: "%.1f GB", sizeGb)
        } else {
            return "Calculating..."
        }
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
    
    // MARK: - Local Size Calculation
    
    private func formatLocalSize() -> String {
        let path = localPath
        guard FileManager.default.fileExists(atPath: path) else { return "Unknown" }
        
        do {
            let totalSize = try calculateDirectorySize(at: path)
            return formatBytes(totalSize)
        } catch {
            return "Unknown"
        }
    }
    
    private func calculateDirectorySize(at path: String) throws -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        print("Calculating directory size for path: \(path)")
        
        let directoryURL = URL(fileURLWithPath: path)
        
        guard fileManager.fileExists(atPath: path) else {
            print("Path does not exist: \(path)")
            return 0
        }
        
        let resourceKeys: [URLResourceKey] = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileSizeKey, .nameKey]
        let enumerator = fileManager.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: resourceKeys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants],
            errorHandler: { (url, error) -> Bool in
                print("Error accessing \(url): \(error)")
                return true
            }
        )
        
        guard let fileEnumerator = enumerator else {
            throw NSError(domain: "FileEnumerationError", code: -1, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create file enumerator"])
        }
        
        var fileCount = 0
        for case let fileURL as URL in fileEnumerator {
            do {
                let resourceValues = try fileURL.resourceValues(forKeys: Set(resourceKeys))
                
                guard let isRegularFile = resourceValues.isRegularFile, isRegularFile else { continue }
                
                let fileName = resourceValues.name ?? "Unknown"
                fileCount += 1
                
                // Use actual disk allocated size, fallback to logical size if not available
                if let actualSize = resourceValues.totalFileAllocatedSize {
                    totalSize += Int64(actualSize)
                    
                    if fileCount <= 10 {
                        let actualSizeGB = Double(actualSize) / (1024 * 1024 * 1024)
                        let logicalSizeGB = Double(resourceValues.fileSize ?? 0) / (1024 * 1024 * 1024)
                        print("File \(fileCount): \(fileName) - Logical: \(String(format: "%.3f", logicalSizeGB)) GB, Actual: \(String(format: "%.3f", actualSizeGB)) GB")
                    }
                } else if let logicalSize = resourceValues.fileSize {
                    totalSize += Int64(logicalSize)
                    
                    if fileCount <= 10 {
                        let logicalSizeGB = Double(logicalSize) / (1024 * 1024 * 1024)
                        print("File \(fileCount): \(fileName) - Size: \(String(format: "%.3f", logicalSizeGB)) GB (fallback)")
                    }
                }
            } catch {
                print("Error getting resource values for \(fileURL): \(error)")
                continue
            }
        }
        
        let totalSizeGB = Double(totalSize) / (1024 * 1024 * 1024)
        print("Total files: \(fileCount), Total actual disk usage: \(String(format: "%.2f", totalSizeGB)) GB")
        
        return totalSize
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
