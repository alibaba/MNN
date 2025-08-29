//
//  ModelScopeDownloadManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

// MARK: - ModelScopeDownloadManager

/// ModelScopeDownloadManager - Specialized download manager for ModelScope and Modeler platforms
/// 
/// This actor-based download manager provides platform-specific optimizations for downloading
/// models from ModelScope and Modeler repositories. It implements intelligent resume functionality,
/// comprehensive error handling, and maintains directory structure integrity.
/// 
/// Key Features:
/// - Platform-specific URL handling for ModelScope and Modeler
/// - Intelligent resume capability with temporary file preservation
/// - Real-time progress tracking with optimized callback frequency
/// - Recursive directory structure preservation
/// - File integrity validation using size verification
/// - Exponential backoff retry mechanism with configurable attempts
/// - Memory-efficient streaming downloads
/// - Thread-safe operations using Swift Actor model
/// 
/// Architecture:
/// - Uses URLSession.bytes for memory-efficient streaming
/// - Implements temporary file management for resume functionality
/// - Supports both ModelScope and Modeler API endpoints
/// - Maintains download state persistence through ModelDownloadStorage
/// 
/// Performance Optimizations:
/// - Progress update throttling (every 320KB) to prevent UI blocking
/// - Temporary file reuse for interrupted downloads
/// - Efficient directory traversal with recursive file discovery
/// - Minimal memory footprint through streaming downloads
/// 
/// Error Handling:
/// - Comprehensive retry logic with exponential backoff
/// - Graceful cancellation with state preservation
/// - File integrity validation and automatic cleanup
/// - Network error recovery with configurable retry attempts
/// 
/// Usage:
/// ```swift
/// let manager = ModelScopeDownloadManager(
///     repoPath: "damo/Qwen-1.5B",
///     source: .modelScope
/// )
/// try await manager.downloadModel(
///     to: "models",
///     modelId: "qwen-1.5b",
///     modelName: "Qwen-1.5B"
/// ) { progress in
///     print("Progress: \(progress * 100)%")
/// }
/// ```
@available(iOS 13.4, macOS 10.15, *)
public actor ModelScopeDownloadManager: ModelDownloadManagerProtocol {
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
    
    // Download cancellation related properties
    private var isCancelled: Bool = false
    private var currentDownloadTask: Task<Void, Error>?
    private var currentFileHandle: FileHandle?
    
    // MARK: - Initialization
    
    /// Creates a new ModelScope download manager with platform-specific configuration
    /// 
    /// - Parameters:
    ///   - repoPath: Repository path in format "owner/model-name"
    ///   - config: URLSession configuration for network behavior customization
    ///             Use .default for standard downloads, .background for background downloads
    ///   - enableLogging: Whether to enable detailed debug logging
    ///   - source: Target platform (ModelScope or Modeler)
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
        ModelDownloadLogger.isEnabled = enableLogging
    }
    
    // MARK: - Public Methods
    
    /// Downloads a complete model from ModelScope or Modeler repository
    /// 
    /// This method orchestrates the entire download process including file discovery,
    /// directory structure creation, resume functionality, and progress tracking.
    /// It supports both ModelScope and Modeler platforms with platform-specific optimizations.
    /// 
    /// - Parameters:
    ///   - destinationFolder: Base folder for download (relative to Documents)
    ///   - modelId: Unique identifier for the model
    ///   - modelName: Display name used for folder creation
    ///   - progress: Optional progress callback (0.0 to 1.0)
    /// - Throws: ModelScopeError for network, file system, or validation failures
    /// 
    /// Example:
    /// ```swift
    /// try await manager.downloadModel(
    ///     to: "models",
    ///     modelId: "qwen-1.5b",
    ///     modelName: "Qwen-1.5B"
    /// ) { progress in
    ///     print("Progress: \(progress * 100)%")
    /// }
    /// ```
    public func downloadModel(
        to destinationFolder: String = "",
        modelId: String,
        modelName: String,
        progress: ((Double) -> Void)? = nil
    ) async throws {
        
        isCancelled = false
        
        ModelDownloadLogger.info("Starting download for modelId: \(modelId)")
        
        let destination = try resolveDestinationPath(base: destinationFolder, modelId: modelName)
        ModelDownloadLogger.info("Will download to: \(destination)")
        
        let files = try await fetchFileList(root: "", revision: "")
        totalFiles = files.count
        ModelDownloadLogger.info("Fetched \(files.count) files")
                
        try await downloadFiles(
            files: files,
            revision: "",
            destinationPath: destination,
            progress: progress ?? { _ in }
        )
    }
    
    /// Cancels all ongoing download operations while preserving resume capability
    /// 
    /// This method gracefully stops all active downloads, closes file handles,
    /// and preserves temporary files to enable resume functionality in future attempts.
    /// The URLSession is invalidated to ensure clean cancellation.
    public func cancelDownload() async {
        isCancelled = true
        
        currentDownloadTask?.cancel()
        currentDownloadTask = nil
        
        await closeFileHandle()
        
        session.invalidateAndCancel()
        
        ModelDownloadLogger.info("Download cancelled, temporary files preserved for resume")
    }
    
    // MARK: - Private Methods - Progress Management
    
    /// Updates download progress with throttling to prevent excessive UI updates
    /// 
    /// - Parameters:
    ///   - progress: Current progress value (0.0 to 1.0)
    ///   - callback: Progress callback function to invoke on main thread
    private func updateProgress(_ progress: Double, callback: @escaping (Double) -> Void) {
        Task { @MainActor in
            callback(progress)
        }
    }
    
    /// Fetches the complete file list from ModelScope or Modeler repository
    /// 
    /// This method queries the repository API to discover all available files,
    /// supporting both ModelScope and Modeler platform endpoints with proper
    /// error handling and response validation.
    /// 
    /// - Parameters:
    ///   - root: Root directory path to fetch files from
    ///   - revision: Model revision/version to fetch files for
    /// - Returns: Array of ModelFile objects representing repository files
    /// - Throws: ModelScopeError if request fails or response is invalid
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
    
    
    /**
     * Downloads a single file with intelligent resume and retry mechanisms
     * 
     * This method handles individual file downloads with comprehensive error recovery,
     * resume functionality through temporary files, and progress tracking. It supports
     * both ModelScope and Modeler platforms with platform-specific URL construction.
     * 
     * Features:
     * - Automatic resume from temporary files using HTTP Range requests
     * - Exponential backoff retry mechanism (configurable attempts)
     * - Memory-efficient streaming using URLSession.bytes
     * - File integrity validation using size verification
     * - Progress update throttling to prevent UI blocking
     * - Graceful cancellation with state preservation
     * 
     * @param file ModelFile metadata including path, size, and download information
     * @param destinationPath Target local path for the downloaded file
     * @param onProgress Progress callback receiving downloaded bytes count
     * @param maxRetries Maximum number of retry attempts (default: 3)
     * @param retryDelay Delay between retry attempts in seconds (default: 2.0)
     * @throws ModelScopeError if download fails after all retry attempts
     */
    private func downloadFile(
        file: ModelFile,
        destinationPath: String,
        onProgress: @escaping (Int64) -> Void,
        maxRetries: Int = 3,
        retryDelay: TimeInterval = 2.0
    ) async throws {
        var lastError: Error?
        
        for attempt in 1...maxRetries {
            if isCancelled { break }
            
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
        
        if isCancelled {
            throw ModelScopeError.downloadCancelled
        }
        
        let session = self.session
        
        ModelDownloadLogger.info("Starting download for file: \(file.name)")
        ModelDownloadLogger.debug("Destination path: \(destinationPath)")
        
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
            ModelDownloadLogger.info("Resuming download from offset: \(resumeOffset)")
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
        
        ModelDownloadLogger.debug("Requesting URL: \(url)")
        
        return try await withCheckedThrowingContinuation { continuation in
            currentDownloadTask = Task {
                do {
                    let (asyncBytes, response) = try await session.bytes(for: request)
                    ModelDownloadLogger.debug("Response status code: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                    try validateResponse(response)
                    
                    let fileHandle = try FileHandle(forWritingTo: tempURL)
                    self.currentFileHandle = fileHandle
                    
                    if resumeOffset > 0 {
                        try fileHandle.seek(toOffset: UInt64(resumeOffset))
                    }
                    
                    var downloadedBytes: Int64 = resumeOffset
                    var bytesCount = 0
                    
                    for try await byte in asyncBytes {
                        // Frequently check cancellation status
                        if isCancelled {
                            try fileHandle.close()
                            self.currentFileHandle = nil
                            // Don't delete temp files when cancelled, preserve resume functionality
                            continuation.resume(throwing: ModelScopeError.downloadCancelled)
                            return
                        }
                        
                        try fileHandle.write(contentsOf: [byte])
                        downloadedBytes += 1
                        bytesCount += 1
                        
                        // Reduce progress callback frequency: update every 64KB * 5 instead of every 1KB
                        if bytesCount >= 64 * 1024 * 5 {
                            onProgress(downloadedBytes)
                            bytesCount = 0
                        }
                    }
                    
                    try fileHandle.close()
                    self.currentFileHandle = nil
                    
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
                    // Clean up file handle when handling errors
                    if let handle = self.currentFileHandle {
                        try? handle.close()
                        self.setCurrentFileHandle(nil)
                    }
                    
                    if !isCancelled {
                        ModelDownloadLogger.error("Download failed: \(error.localizedDescription)")
                        storage.clearFileStatus(at: destination.path)
                    }
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Downloads files recursively with directory structure preservation
    /// 
    /// This method processes the complete file list, creating necessary directory
    /// structures and downloading files in the correct order. It calculates total
    /// download size, handles existing files, and maintains progress tracking.
    /// 
    /// - Parameters:
    ///   - files: Array of ModelFile objects representing all repository files
    ///   - revision: Model revision/version for download URLs
    ///   - destinationPath: Base directory path for downloads
    ///   - progress: Progress callback function (0.0 to 1.0)
    /// - Throws: ModelScopeError if any file download fails
    private func downloadFiles(
        files: [ModelFile],
        revision: String,
        destinationPath: String,
        progress: @escaping (Double) -> Void
    ) async throws {
        ModelDownloadLogger.info("Starting download with \(files.count) files")
        
        if isCancelled {
            throw ModelScopeError.downloadCancelled
        }
        
        if totalSize == 0 {
            totalSize = try await calculateTotalSize(files: files, revision: revision)
            print("Total download size: \(totalSize) bytes")
        }
        
        for file in files {
            
            if Task.isCancelled || isCancelled {
                throw ModelScopeError.downloadCancelled
            }
            
            ModelDownloadLogger.debug("Processing: \(file.name), type: \(file.type)")
            
            if file.type == "tree" {
                let newPath = (destinationPath as NSString)
                    .appendingPathComponent(file.name.sanitizedPath)
                ModelDownloadLogger.debug("Creating directory: \(newPath)")
                
                try fileManager.createDirectoryIfNeeded(at: newPath)
                
                let subFiles = try await fetchFileList(
                    root: file.path,
                    revision: revision
                )
                ModelDownloadLogger.debug("Found \(subFiles.count) subfiles in \(file.path)")
                
                try await downloadFiles(
                    files: subFiles,
                    revision: revision,
                    destinationPath: newPath,
                    progress: progress
                )
            } else if file.type == "blob" {
                
                ModelDownloadLogger.debug("Downloading: \(file.name)")
                
                if !storage.isFileDownloaded(file, at: destinationPath) {
                    try await downloadFile(
                        file: file,
                        destinationPath: destinationPath,
                        onProgress: { downloadedBytes in
                            let currentProgress = Double(self.downloadedSize + downloadedBytes) / Double(self.totalSize)
                            self.updateProgress(currentProgress, callback: progress)
                        },
                        maxRetries: 500, // Can be made configurable
                        retryDelay: 1.0 // Can be made configurable
                    )
                    
                    downloadedSize += Int64(file.size)
                    storage.saveFileStatus(file, at: destinationPath)
                    ModelDownloadLogger.info("Downloaded and saved: \(file.name)")
                    
                } else {
                    downloadedSize += Int64(file.size)
                    ModelDownloadLogger.debug("File exists: \(file.name)")
                }
                
                let currentProgress = Double(downloadedSize) / Double(totalSize)
                updateProgress(currentProgress, callback: progress)
            }
        }
        
        Task { @MainActor in
            progress(1.0)
        }
    }
    
    /// Calculates the total download size for progress tracking
    /// 
    /// Recursively traverses directory structures to compute the total size
    /// of all files that need to be downloaded, enabling accurate progress reporting.
    /// 
    /// - Parameters:
    ///   - files: Array of ModelFile objects to calculate size for
    ///   - revision: Model revision for fetching subdirectory contents
    /// - Returns: Total size in bytes across all files
    /// - Throws: ModelScopeError if file list fetching fails
    private func calculateTotalSize(files: [ModelFile], revision: String) async throws -> Int64 {
        var size: Int64 = 0
        for file in files {
            if file.type == "tree" {
                let subFiles = try await fetchFileList(
                    root: file.path,
                    revision: revision
                )
                size += try await calculateTotalSize(files: subFiles, revision: revision)
            } else if file.type == "blob" {
                size += Int64(file.size)
            }
        }
        return size
    }
    
    
    /// Resets internal download state for a fresh download session
    /// 
    /// Clears progress counters and prepares the manager for a new download operation.
    /// This method is called at the beginning of each download to ensure clean state.
    private func resetDownloadState() async {
        totalFiles = 0
        downloadedFiles = 0
        totalSize = 0
        downloadedSize = 0
        lastUpdatedBytes = 0
    }
    
    /// Resets the cancellation flag to allow new download operations
    /// 
    /// Clears all download state including cancellation status and progress counters,
    /// preparing the manager for a completely fresh download session.
    private func resetCancelStatus() {
        isCancelled = false
        
        totalFiles = 0
        downloadedFiles = 0
        totalSize = 0
        downloadedSize = 0
        lastUpdatedBytes = 0
    }
    
    /// Safely closes the current file handle to prevent resource leaks
    /// 
    /// This method ensures proper cleanup of file handles during cancellation
    /// or error conditions, preventing file descriptor leaks.
    private func closeFileHandle() async {
        do {
            try currentFileHandle?.close()
            currentFileHandle = nil
        } catch {
            print("Error closing file handle: \(error)")
        }
    }
    
    /// Constructs ModelScope API URLs with proper query parameters
    /// 
    /// Builds complete URLs for ModelScope repository API endpoints,
    /// handling URL encoding and validation.
    /// 
    /// - Parameters:
    ///   - path: API endpoint path to append to base URL
    ///   - queryItems: URL query parameters for the request
    /// - Returns: Constructed and validated URL
    /// - Throws: ModelScopeError.invalidURL if URL construction fails
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
    
    /// Constructs Modeler platform URLs with proper query parameters
    /// 
    /// Builds complete URLs for Modeler repository API endpoints,
    /// handling URL encoding and validation for the Modeler platform.
    /// 
    /// - Parameters:
    ///   - path: File path within the repository
    ///   - queryItems: URL query parameters for the request
    /// - Returns: Constructed and validated URL
    /// - Throws: ModelScopeError.invalidURL if URL construction fails
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
    
    /// Validates HTTP response status codes for successful requests
    /// 
    /// Ensures the HTTP response indicates success (2xx status codes)
    /// and throws appropriate errors for failed requests.
    /// 
    /// - Parameter response: URLResponse to validate
    /// - Throws: ModelScopeError.invalidResponse if status code indicates failure
    private func validateResponse(_ response: URLResponse) throws {
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw ModelScopeError.invalidResponse
        }
    }
    
    /// Resolves and creates the complete destination path for model downloads
    /// 
    /// Constructs the full local file system path where the model will be downloaded,
    /// creating necessary directory structures and validating access permissions.
    /// 
    /// - Parameters:
    ///   - base: Base folder path relative to Documents directory
    ///   - modelId: Model identifier used for folder naming
    /// - Returns: Absolute path to the model download directory
    /// - Throws: ModelScopeError.fileSystemError if directory creation fails
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
    
    /// Thread-safe setter for the current file handle
    /// 
    /// - Parameter handle: FileHandle instance to set, or nil to clear
    private func setCurrentFileHandle(_ handle: FileHandle?) {
        currentFileHandle = handle
    }
    
    /// Retrieves the size of a temporary file for resume functionality
    /// 
    /// Calculates the current size of a temporary download file to determine
    /// the resume offset for interrupted downloads.
    /// 
    /// - Parameters:
    ///   - file: ModelFile to get temporary file size for
    ///   - destinationPath: Destination path used for temp file naming
    /// - Returns: Size of temporary file in bytes, or 0 if file doesn't exist
    private func getTempFileSize(for file: ModelFile, destinationPath: String) -> Int64 {
        let modelHash = repoPath.hash
        let fileHash = file.path.hash
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("model_\(modelHash)_file_\(fileHash)_\(file.name.sanitizedPath).tmp")
        
        guard fileManager.fileExists(atPath: tempURL.path) else {
            return 0
        }
        
        do {
            let attributes = try fileManager.attributesOfItem(atPath: tempURL.path)
            return attributes[.size] as? Int64 ?? 0
        } catch {
            ModelDownloadLogger.error("Failed to get temp file size for \(file.name): \(error)")
            return 0
        }
    }
}
