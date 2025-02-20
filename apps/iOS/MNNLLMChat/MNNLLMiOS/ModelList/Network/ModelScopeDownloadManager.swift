//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

@available(iOS 13.0, macOS 10.15, *)
public actor ModelScopeDownloadManager: Sendable {
    
    private let repoPath: String
    private let baseURL = "https://modelscope.cn/api/v1/models"
    private let session: URLSession
    private let fileManager: FileManager
    private let storage: DownloadStorage
    
    private var totalFiles = 0
    private var downloadedFiles = 0
    private var totalSize: Int64 = 0
    private var downloadedSize: Int64 = 0
    
    public init(
        repoPath: String,
        inBackground: Bool = true
    ) {
        self.repoPath = repoPath
        self.fileManager = .default
        self.storage = DownloadStorage()
        
        let config = URLSessionConfiguration.default
        
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)
    }
    
    // MARK: - Public Methods
    
    public func downloadModel(
        to destinationFolder: String = "",
        modelId: String,
        modelName: String,
        progress: ((Double) -> Void)? = nil
    ) async throws {
        print("Starting download for modelId: \(modelId)")
        
        let destination = try resolveDestinationPath(base: destinationFolder, modelId: modelName)
        print("Will download to: \(destination)")
        
        let files = try await fetchFileList(root: "", revision: "")
        totalFiles = files.count
        print("Fetched \(files.count) files")
                
        try await downloadFiles(
            files: files,
            revision: "",
            destinationPath: destination,
            progress: progress ?? { _ in }
        )
    }
    
    // MARK: - Private Methods
    
    private func fetchFileList(
        root: String,
        revision: String
    ) async throws -> [ModelFile] {
        let url = try buildURL(
            path: "/repo/files",
            queryItems: [
                URLQueryItem(name: "Root", value: root),
                URLQueryItem(name: "Revision", value: revision)
            ]
        )
        let (data, response) = try await session.data(from: url)
        try validateResponse(response)
        
        let modelResponse = try JSONDecoder().decode(ModelResponse.self, from: data)
        return modelResponse.data.files
    }
    
    
    private func downloadFile(
        file: ModelFile,
        destinationPath: String,
        onProgress: @escaping (Int64) -> Void,
        maxRetries: Int = 3,
        retryDelay: TimeInterval = 2.0
    ) async throws {
        var lastError: Error?
        
        for attempt in 1...maxRetries {
            do {
                print("Attempt \(attempt) of \(maxRetries) for file: \(file.name)")
                try await downloadFileWithRetry(
                    file: file,
                    destinationPath: destinationPath,
                    onProgress: onProgress
                )
                return // 下载成功，直接返回
            } catch {
                lastError = error
                print("Download failed (attempt \(attempt)): \(error.localizedDescription)")
                
                if attempt < maxRetries {
                    print("Retrying in \(retryDelay) seconds...")
                    try await Task.sleep(nanoseconds: UInt64(retryDelay * 1_000_000_000))
                }
            }
        }
        
        // 所有重试都失败了，抛出最后一个错误
        throw lastError ?? ModelScopeError.downloadFailed(NSError(
            domain: "ModelScope",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "All download attempts failed"]
        ))
    }

    private func downloadFileWithRetry(
        file: ModelFile,
        destinationPath: String,
        onProgress: @escaping (Int64) -> Void
    ) async throws {
        // 创建一个强引用来保持 session 存活
        let session = self.session
        
        print("Starting download for file: \(file.name)")
        print("Destination path: \(destinationPath)")
        
        let destination = URL(fileURLWithPath: destinationPath)
            .appendingPathComponent(file.name.sanitizedPath)
        
        // 使用模型ID和文件路径组合生成唯一的临时文件名
        let modelHash = repoPath.hash
        let fileHash = file.path.hash
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("model_\(modelHash)_file_\(fileHash)_\(file.name.sanitizedPath).tmp")
        
        var resumeOffset: Int64 = 0
        
        // 如果临时文件存在，获取已下载的大小
        if fileManager.fileExists(atPath: tempURL.path) {
            let attributes = try fileManager.attributesOfItem(atPath: tempURL.path)
            resumeOffset = attributes[.size] as? Int64 ?? 0
            print("Resuming download from offset: \(resumeOffset)")
        } else {
            // 创建新的临时文件
            try Data().write(to: tempURL)
        }
        
        let url = try buildURL(
            path: "/repo",
            queryItems: [
                URLQueryItem(name: "Revision", value: "master"),
                URLQueryItem(name: "FilePath", value: file.path)
            ]
        )
        
        // 创建带有 Range 头的请求
        var request = URLRequest(url: url)
        if resumeOffset > 0 {
            request.setValue("bytes=\(resumeOffset)-", forHTTPHeaderField: "Range")
        }
        
        // 使用 URLSession 的 bytes API 来获取下载进度
        print("Requesting URL: \(url)")
        
        // 使用 withCheckedThrowingContinuation 来管理异步操作的生命周期
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let (asyncBytes, response) = try await session.bytes(for: request)
                    print("Response status code: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                    try validateResponse(response)
                    
                    // 打开文件用于追加
                    let fileHandle = try FileHandle(forWritingTo: tempURL)
                    if resumeOffset > 0 {
                        try fileHandle.seek(toOffset: UInt64(resumeOffset))
                    }
                    
                    var downloadedBytes: Int64 = resumeOffset
                    
                    // 分块下载并报告进度
                    for try await byte in asyncBytes {
                        try fileHandle.write(contentsOf: [byte])
                        downloadedBytes += 1
                        if downloadedBytes % 1024 == 0 { // 每 1KB 更新一次进度
                            onProgress(downloadedBytes)
                        }
                    }
                    
                    // 确保文件已关闭
                    try fileHandle.close()
                    
                    // 验证文件大小
                    let finalSize = try FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int64 ?? 0
                    guard finalSize == file.size else {
                        throw ModelScopeError.downloadFailed(NSError(
                            domain: "ModelScope",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "Downloaded file size mismatch: expected \(file.size), got \(finalSize)"]
                        ))
                    }
                    
                    // 移动到最终位置
                    if fileManager.fileExists(atPath: destination.path) {
                        try fileManager.removeItem(at: destination)
                    }
                    try fileManager.moveItem(at: tempURL, to: destination)
                    
                    // 报告最终进度
                    onProgress(downloadedBytes)
                    
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    private func downloadFiles(
        files: [ModelFile],
        revision: String,
        destinationPath: String,
        progress: @escaping (Double) -> Void
    ) async throws {
        print("Starting downloadFiles with \(files.count) files")
        
        // 计算所有文件的总大小
        func calculateTotalSize(files: [ModelFile]) async throws -> Int64 {
            var size: Int64 = 0
            for file in files {
                if file.type == "tree" {
                    let subFiles = try await fetchFileList(
                        root: file.path,
                        revision: revision
                    )
                    size += try await calculateTotalSize(files: subFiles)
                } else if file.type == "blob" {
                    size += Int64(file.size)
                }
            }
            return size
        }
        
        // 在开始下载前计算总大小
        if totalSize == 0 {
            totalSize = try await calculateTotalSize(files: files)
            print("Total download size: \(totalSize) bytes")
        }
        
        for file in files {
            print("Processing file: \(file.name), type: \(file.type)")
            
            if file.type == "tree" {
                let newPath = (destinationPath as NSString)
                    .appendingPathComponent(file.name.sanitizedPath)
                print("Created directory at: \(newPath)")
                
                try fileManager.createDirectoryIfNeeded(at: newPath)
                
                let subFiles = try await fetchFileList(
                    root: file.path,
                    revision: revision
                )
                print("Found \(subFiles.count) subfiles in \(file.path)")
                
                try await downloadFiles(
                    files: subFiles,
                    revision: revision,
                    destinationPath: newPath,
                    progress: progress
                )
            } else if file.type == "blob" {
                print("Downloading blob: \(file.name)")
                if !storage.isFileDownloaded(file, at: destinationPath) {
                    
                    let destination = URL(fileURLWithPath: destinationPath)
                        .appendingPathComponent(file.name.sanitizedPath)
                    
                    let url = try buildURL(
                        path: "/repo",
                        queryItems: [
                            URLQueryItem(name: "Revision", value: "master"),
                            URLQueryItem(name: "FilePath", value: file.path)
                        ]
                    )

                    try await downloadFile(
                        file: file,
                        destinationPath: destinationPath,
                        onProgress: { downloadedBytes in
                            let currentProgress = Double(self.downloadedSize + downloadedBytes) / Double(self.totalSize)
                            print("currentProgress:\(currentProgress), self.downloadedSize\(self.downloadedSize), \(self.totalSize)")
                            progress(currentProgress)
                        },
                        maxRetries: 10,  // 最多重试3次
                        retryDelay: 1.0 // 每次重试间隔2秒
                    )
                    
                    downloadedSize += Int64(file.size)
                    storage.saveFileStatus(file, at: destinationPath)
                    print("Downloaded and saved: \(file.name)")
                    
                } else {
                    downloadedSize += Int64(file.size)
                    print("File already exists: \(file.name)")
                }
                
                progress(Double(downloadedSize) / Double(totalSize))
            }
        }
    }
    
    private func buildURL(
        path: String,
        queryItems: [URLQueryItem]
    ) throws -> URL {
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
    
    private func resolveDestinationPath(
        base: String,
        modelId: String
    ) throws -> String {
        guard let documentsPath = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)
            .first else {
            throw ModelScopeError.fileSystemError(
                NSError(domain: "ModelScope", code: -1, userInfo: [
                    NSLocalizedDescriptionKey: "Cannot access Documents directory"
                ])
            )
        }
        
        let modelScopePath = documentsPath
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("taobao-mnn", isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)
        
        try fileManager.createDirectoryIfNeeded(at: modelScopePath.path)
        
        return modelScopePath.path
    }
}
