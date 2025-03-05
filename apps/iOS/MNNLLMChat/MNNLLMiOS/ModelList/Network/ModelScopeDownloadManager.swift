//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

// MARK: - ModelScopeDownloadManager

/// A manager class that handles downloading models from ModelScope repository
/// Supports features like:
/// - Resume interrupted downloads
/// - Progress tracking
/// - File integrity validation
/// - Directory structure preservation
@available(iOS 13.4, macOS 10.15, *)
public actor ModelScopeDownloadManager: Sendable {
    // MARK: - Properties
    
    private let repoPath: String
    private let session: URLSession
    private let fileManager: FileManager
    private let storage: ModelDownloadStorage

    private var source: ModelSource
    private var totalFiles: Int = 0
    private var downloadedFiles: Int = 0
    private var totalSize: Int64 = 0
    private var downloadedSize: Int64 = 0
    private var lastUpdatedBytes: Int64 = 0
    
    // MARK: - Initialization
    
    /// Creates a new ModelScope download manager
    /// - Parameters:
    ///   - repoPath: The repository path in format "owner/model-name"
    ///   - config: URLSession configuration for customizing network behavior.
    ///             Use `.default` for standard downloads, `.background` for background downloads.
    ///             Defaults to `.default`
    ///   - enableLogging: Whether to enable debug logging. Defaults to true
    ///   - source: use modelScope or modeler
    /// - Note: When using background configuration, the app must handle URLSession background events
    public init(
        repoPath: String,
        config: URLSessionConfiguration = .default,
        enableLogging: Bool = true,
        source: ModelSource
    ) {
        self.repoPath = repoPath
        self.fileManager = .default
        self.storage = ModelDownloadStorage()
        self.session = URLSession(configuration: config)
        self.source = source
        ModelScopeLogger.isEnabled = enableLogging
    }
    
    // MARK: - Public Methods
    
    /// Downloads a model from ModelScope repository
    /// - Parameters:
    ///   - destinationPath: Local path where the model will be saved
    ///   - revision: Model revision/version to download (defaults to latest)
    ///   - progress: Closure called with download progress (0.0 to 1.0)
    /// - Throws: ModelScopeError for network, file system, or validation failures
    /// - Returns: Void when download completes successfully
    ///
    /// Example usage:
    /// ```swift
    /// let manager = ModelScopeDownloadManager(repoPath: "damo/Qwen-1.5B")
    /// try await manager.downloadModel(
    ///     to: "/path/to/models",
    ///     progress: { progress in
    ///         print("Download progress: \(progress * 100)%")
    ///     }
    /// )
    /// Will download to /path/to/models/Qwen-1.5B
    /// ```
    public func downloadModel(
        to destinationFolder: String = "",
        modelId: String,
        modelName: String,
        progress: ((Double) -> Void)? = nil
    ) async throws {
        ModelScopeLogger.info("Starting download for modelId: \(modelId)")
        
        let destination = try resolveDestinationPath(base: destinationFolder, modelId: modelName)
        ModelScopeLogger.info("Will download to: \(destination)")
        
        let files = try await fetchFileList(root: "", revision: "")
        totalFiles = files.count
        ModelScopeLogger.info("Fetched \(files.count) files")
                
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
                return
            } catch {
                lastError = error
                print("Download failed (attempt \(attempt)): \(error.localizedDescription)")
                
                if attempt < maxRetries {
                    print("Retrying in \(retryDelay) seconds...")
                    try await Task.sleep(nanoseconds: UInt64(retryDelay * 1_000_000_000))
                }
            }
        }
        
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
        let session = self.session
        
        ModelScopeLogger.info("Starting download for file: \(file.name)")
        ModelScopeLogger.debug("Destination path: \(destinationPath)")
        
        let destination = URL(fileURLWithPath: destinationPath)
            .appendingPathComponent(file.name.sanitizedPath)
        
        let modelHash = repoPath.hash
        let fileHash = file.path.hash
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("model_\(modelHash)_file_\(fileHash)_\(file.name.sanitizedPath).tmp")
        
        var resumeOffset: Int64 = 0
        
        if fileManager.fileExists(atPath: tempURL.path) {
            let attributes = try fileManager.attributesOfItem(atPath: tempURL.path)
            resumeOffset = attributes[.size] as? Int64 ?? 0
            ModelScopeLogger.info("Resuming download from offset: \(resumeOffset)")
        } else {
            try Data().write(to: tempURL)
        }
        
        let url: URL
        if source == .modelScope {
            url = try buildURL(
                path: "/repo",
                queryItems: [
                    URLQueryItem(name: "Revision", value: "master"),
                    URLQueryItem(name: "FilePath", value: file.path)
                ]
            )
        } else {
            url = try buildModelerURL(
                path: file.path,
                queryItems: []
            )
        }
        
        var request = URLRequest(url: url)
        if resumeOffset > 0 {
            request.setValue("bytes=\(resumeOffset)-", forHTTPHeaderField: "Range")
        }
        
        ModelScopeLogger.debug("Requesting URL: \(url)")
        
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let (asyncBytes, response) = try await session.bytes(for: request)
                    ModelScopeLogger.debug("Response status code: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                    try validateResponse(response)
                    
                    let fileHandle = try FileHandle(forWritingTo: tempURL)
                    if resumeOffset > 0 {
                        try fileHandle.seek(toOffset: UInt64(resumeOffset))
                    }
                    
                    var downloadedBytes: Int64 = resumeOffset
                    
                    for try await byte in asyncBytes {
                        try fileHandle.write(contentsOf: [byte])
                        downloadedBytes += 1
                        if downloadedBytes % 1024 == 0 {
                            onProgress(downloadedBytes)
                        }
                    }
                    
                    try fileHandle.close()
                    
                    let finalSize = try FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int64 ?? 0
                    guard finalSize == file.size else {
                        storage.clearFileStatus(at: destination.path)
                        throw ModelScopeError.downloadFailed(NSError(
                            domain: "ModelScope",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "Size mismatch: expected \(file.size), got \(finalSize)"]
                        ))
                    }
                    
                    if fileManager.fileExists(atPath: destination.path) {
                        try fileManager.removeItem(at: destination)
                    }
                    try fileManager.moveItem(at: tempURL, to: destination)
                    
                    onProgress(downloadedBytes)
                    continuation.resume()
                } catch {
                    ModelScopeLogger.error("Download failed: \(error.localizedDescription)")
                    storage.clearFileStatus(at: destination.path)
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
        ModelScopeLogger.info("Starting download with \(files.count) files")
        
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
        
        if totalSize == 0 {
            totalSize = try await calculateTotalSize(files: files)
            print("Total download size: \(totalSize) bytes")
        }
        
        for file in files {
            ModelScopeLogger.debug("Processing: \(file.name), type: \(file.type)")
            
            if file.type == "tree" {
                let newPath = (destinationPath as NSString)
                    .appendingPathComponent(file.name.sanitizedPath)
                ModelScopeLogger.debug("Creating directory: \(newPath)")
                
                try fileManager.createDirectoryIfNeeded(at: newPath)
                
                let subFiles = try await fetchFileList(
                    root: file.path,
                    revision: revision
                )
                ModelScopeLogger.debug("Found \(subFiles.count) subfiles in \(file.path)")
                
                try await downloadFiles(
                    files: subFiles,
                    revision: revision,
                    destinationPath: newPath,
                    progress: progress
                )
            } else if file.type == "blob" {
                ModelScopeLogger.debug("Downloading: \(file.name)")
                if !storage.isFileDownloaded(file, at: destinationPath) {
                    try await downloadFile(
                        file: file,
                        destinationPath: destinationPath,
                        onProgress: { downloadedBytes in
                            let currentProgress = Double(self.downloadedSize + downloadedBytes) / Double(self.totalSize)
                            progress(currentProgress)
                            // 1MB = 1,024 * 1,024
                           let bytesDelta = self.downloadedSize - self.lastUpdatedBytes
                           if bytesDelta >= 1_024 * 1_024 {
                               self.lastUpdatedBytes = self.downloadedSize
                               DispatchQueue.main.async {
                                   progress(currentProgress)
                               }
                           }
                        },
                        maxRetries: 50,
                        retryDelay: 1.0
                    )
                    
                    downloadedSize += Int64(file.size)
                    storage.saveFileStatus(file, at: destinationPath)
                    ModelScopeLogger.info("Downloaded and saved: \(file.name)")
                    
                } else {
                    downloadedSize += Int64(file.size)
                    ModelScopeLogger.debug("File exists: \(file.name)")
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
    
    private func buildModelerURL(
        path: String,
        queryItems: [URLQueryItem]
    ) throws -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "modelers.cn"
        components.path = "/coderepo/web/v1/file/\(repoPath)/main/media/\(path)"
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
        
        let components = base.components(separatedBy: "/")

        var currentURL = documentsPath
        components.forEach { component in
            currentURL = currentURL.appendingPathComponent(component, isDirectory: true)
        }
        let modelScopePath = currentURL.appendingPathComponent(modelId, isDirectory: true)
        
        try fileManager.createDirectoryIfNeeded(at: modelScopePath.path)
        
        return modelScopePath.path
    }
}
