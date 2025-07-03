//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import Hub
import Foundation

struct ModelInfo: Codable {
    let modelId: String
    let createdAt: String
    let downloads: Int
    let tags: [String]
    
    var name: String {
        modelId.removingTaobaoPrefix()
    }
    
    var isDownloaded: Bool = false
    var lastUsedAt: Date?
    
    var cachedSize: Int64? = nil
    
    var localPath: String {
        return HubApi.shared.localRepoLocation(HubApi.Repo.init(id: modelId)).path
    }
    
    var formattedSize: String {
        if isDownloaded {
            return formatLocalSize()
        } else if let cached = cachedSize {
            return formatBytes(cached)
        } else {
            return "计算中..."
        }
    }
    
    func fetchRemoteSize() async -> Int64? {
        let modelScopeId = modelId.replacingOccurrences(of: "taobao-mnn", with: "MNN")
        
        do {
            let files = try await fetchFileList(repoPath: modelScopeId, root: "", revision: "")
            let totalSize = try await calculateTotalSize(files: files, repoPath: modelScopeId)
            return totalSize
        } catch {
            print("Error fetching remote size for \(modelId): \(error)")
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
        case modelId
        case tags
        case downloads
        case createdAt
        case cachedSize
    }
}

struct RepoInfo: Codable {
    let modelId: String
    let sha: String
    let siblings: [Sibling]

    struct Sibling: Codable {
        let rfilename: String
    }
}
