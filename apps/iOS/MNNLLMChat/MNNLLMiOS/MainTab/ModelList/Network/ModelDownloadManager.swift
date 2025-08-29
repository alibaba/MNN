//
//  ModelDownloadManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/8/27.
//

import Foundation

/// ModelDownloadManager - Advanced concurrent model download manager
/// 
/// This actor-based download manager provides high-performance, resumable downloads
/// with intelligent chunking, dynamic concurrency optimization, and comprehensive
/// error handling for model files from various sources.
/// 
/// Key Features:
/// - Concurrent downloads with dynamic concurrency adjustment
/// - Intelligent file chunking for large files (>50MB)
/// - Resume capability with partial download preservation
/// - Exponential backoff retry mechanism
/// - Real-time progress tracking with throttling
/// - Memory-efficient streaming downloads
/// - Thread-safe operations using Swift Actor model
/// 
/// Architecture:
/// - Uses URLSession for network operations
/// - Implements semaphore-based concurrency control
/// - Supports both direct and chunked download strategies
/// - Maintains download state persistence
/// 
/// Performance Optimizations:
/// - Dynamic chunk size calculation based on network conditions
/// - Optimal concurrency level determination
/// - Progress update throttling to prevent UI blocking
/// - Temporary file management for resume functionality
/// 
/// Usage:
/// ```swift
/// let manager = ModelDownloadManager(
///     repoPath: "owner/model-name",
///     source: .modelScope
/// )
/// try await manager.downloadModel(
///     to: "models",
///     modelId: "example-model",
///     modelName: "ExampleModel"
/// ) { progress in
///     print("Progress: \(progress * 100)%")
/// }
/// ```
@available(iOS 13.4, macOS 10.15, *)
public actor ModelDownloadManager: ModelDownloadManagerProtocol {
    
    // MARK: - Properties
    
    private let repoPath: String
    private var session: URLSession
    private let sessionConfig: URLSessionConfiguration
    private var isSessionInvalidated = false
    private let fileManager: FileManager
    private let storage: ModelDownloadStorage
    private let config: DownloadConfig
    private let source: ModelSource
    
    private let concurrencyManager: DynamicConcurrencyManager
    private var downloadSemaphore: AsyncSemaphore
    private var downloadChunkSemaphore: AsyncSemaphore?
    private var downloadQueue: [DownloadTask] = []
    
    private var progress = DownloadProgress()
    private var progressCallback: ((Double) -> Void)?
    
    private var isCancelled: Bool = false
    
    // MARK: - Initialization
    
    /// Initializes a new ModelDownloadManager with configurable parameters
    /// 
    /// - Parameters:
    ///   - repoPath: Repository path in format "owner/model-name"
    ///   - config: Download configuration including retry and chunk settings
    ///   - sessionConfig: URLSession configuration for network requests
    ///   - enableLogging: Whether to enable detailed logging
    ///   - source: Model source platform (ModelScope, Modeler, etc.)
    ///   - concurrencyConfig: Dynamic concurrency management configuration
    public init(
        repoPath: String,
        config: DownloadConfig = .default,
        sessionConfig: URLSessionConfiguration = .default,
        enableLogging: Bool = true,
        source: ModelSource,
        concurrencyConfig: DynamicConcurrencyConfig = .default
    ) {
        self.repoPath = repoPath
        self.config = config
        self.source = source
        self.sessionConfig = sessionConfig
        self.fileManager = .default
        self.storage = ModelDownloadStorage()
        self.session = URLSession(configuration: sessionConfig)
        self.concurrencyManager = DynamicConcurrencyManager(config: concurrencyConfig)
        self.downloadSemaphore = AsyncSemaphore(value: config.maxConcurrentDownloads)
        
        ModelDownloadLogger.isEnabled = enableLogging
    }
    
    // MARK: - Public Methods
    
    /// Downloads a complete model with all its files
    /// 
    /// This method orchestrates the entire download process including file discovery,
    /// task preparation, concurrent execution, and progress tracking. It supports
    /// resume functionality and handles various error conditions gracefully.
    /// 
    /// - Parameters:
    ///   - destinationFolder: Base folder for download (relative to Documents)
    ///   - modelId: Unique identifier for the model
    ///   - modelName: Display name used for folder creation
    ///   - progress: Optional progress callback (0.0 to 1.0)
    /// - Throws: ModelScopeError for various download failures
    public func downloadModel(
        to destinationFolder: String = "",
        modelId: String,
        modelName: String,
        progress: ((Double) -> Void)? = nil
    ) async throws {
        self.progressCallback = progress
        
        // Ensure we have a valid session and reset cancelled state
        await ensureValidSession()
        
        ModelDownloadLogger.info("Starting optimized download for modelId: \(modelId)")
        
        let destination = try resolveDestinationPath(base: destinationFolder, modelId: modelName)
        ModelDownloadLogger.info("Will download to: \(destination)")
        
        let files = try await fetchFileList(root: "", revision: "")
        
        // Calculate total size and prepare download tasks
        try await prepareDownloadTasks(files: files, destinationPath: destination)
        
        // Start concurrent downloads
        try await executeDownloads()
        
        // Check if download was cancelled during execution
        if isCancelled {
            ModelDownloadLogger.info("Download was cancelled, maintaining current progress state")
            throw ModelScopeError.downloadCancelled
        }
        
        await updateProgress(1.0)
        ModelDownloadLogger.info("Download completed successfully")
    }
    
    /// Cancels all ongoing download operations while preserving partial downloads
    /// 
    /// This method gracefully stops all active downloads and preserves temporary
    /// files to enable resume functionality in future download attempts.
    public func cancelDownload() async {
        isCancelled = true
        
        // Cancel all session tasks but don't invalidate the session
        session.getAllTasks { tasks in
            for task in tasks {
                task.cancel()
            }
        }
        
        ModelDownloadLogger.info("Download cancelled, temporary files preserved for resume")
    }
    
    // MARK: - Private Methods - Task Preparation
    
    /// Prepares download tasks by analyzing files and creating appropriate download strategies
    /// 
    /// This method processes the file list and determines the optimal download approach
    /// for each file based on size, existing partial downloads, and configuration.
    /// 
    /// - Parameters:
    ///   - files: Array of ModelFile objects representing files to download
    ///   - destinationPath: Target directory path for downloads
    /// - Throws: ModelScopeError for file system or processing errors
    private func prepareDownloadTasks(
        files: [ModelFile],
        destinationPath: String
    ) async throws {
        downloadQueue.removeAll()
        progress = DownloadProgress()
        
        try await processFiles(files, destinationPath: destinationPath)
        
        progress.totalFiles = downloadQueue.count
        ModelDownloadLogger.info("Prepared \(downloadQueue.count) download tasks, total size: \(progress.totalBytes) bytes")
    }
    
    /// Processes individual files and creates corresponding download tasks
    /// 
    /// Analyzes each file to determine if it needs chunked or direct download
    /// based on file size and configuration thresholds. Handles directory creation
    /// and recursive file processing for nested structures.
    /// 
    /// - Parameters:
    ///   - files: Array of ModelFile objects to process
    ///   - destinationPath: Base destination path for downloads
    /// - Throws: ModelScopeError for file system or network errors
    private func processFiles(
        _ files: [ModelFile],
        destinationPath: String
    ) async throws {
        for file in files {
            if file.type == "tree" {
                let newPath = (destinationPath as NSString)
                    .appendingPathComponent(file.name.sanitizedPath)
                try fileManager.createDirectoryIfNeeded(at: newPath)
                
                let subFiles = try await fetchFileList(root: file.path, revision: "")
                try await processFiles(subFiles, destinationPath: newPath)
            } else if file.type == "blob" {
                if !storage.isFileDownloaded(file, at: destinationPath) {
                    var task = DownloadTask(
                        file: file,
                        destinationPath: destinationPath,
                        priority: .medium
                    )
                    
                    // Check if file should be chunked
                    if Int64(file.size) > config.largeFileThreshold {
                        task.chunks = await createChunks(for: file)
                        ModelDownloadLogger.info("File \(file.name) will be downloaded in \(task.chunks.count) chunks")
                    }
                    
                    downloadQueue.append(task)
                    progress.totalBytes += Int64(file.size)
                } else {
                    progress.downloadedBytes += Int64(file.size)
                }
            }
        }
    }
    
    /// Creates chunked download information for large files
    /// 
    /// Divides large files into optimal chunks for concurrent downloading,
    /// calculates resume offsets for existing partial downloads, and creates
    /// chunk metadata with temporary file locations.
    /// 
    /// - Parameter file: ModelFile object representing the file to chunk
    /// - Returns: Array of ChunkInfo objects containing chunk metadata
    private func createChunks(for file: ModelFile) async -> [ChunkInfo] {
        let fileSize = Int64(file.size)
        
        let recommendedChunkSize = await concurrencyManager.recommendedChunkSize()
        let chunkSize = min(recommendedChunkSize, config.chunkSize)
        
        let chunkCount = Int(ceil(Double(fileSize) / Double(chunkSize)))
        var chunks: [ChunkInfo] = []
        
        ModelDownloadLogger.info("File \(file.name): using chunk size \(chunkSize / 1024 / 1024)MB, total \(chunkCount) chunks")
        
        for i in 0..<chunkCount {
            let startOffset = Int64(i) * chunkSize
            let endOffset = min(startOffset + chunkSize - 1, fileSize - 1)
            
            let modelHash = repoPath.hash
            let fileHash = file.path.hash
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("model_\(modelHash)_file_\(fileHash)_chunk_\(i)_\(file.name.sanitizedPath).tmp")
            
            // Check if chunk already exists and calculate downloaded bytes
            var downloadedBytes: Int64 = 0
            var isCompleted = false
            
            if fileManager.fileExists(atPath: tempURL.path) {
                do {
                    let attributes = try fileManager.attributesOfItem(atPath: tempURL.path)
                    downloadedBytes = attributes[.size] as? Int64 ?? 0
                    let expectedChunkSize = endOffset - startOffset + 1
                    isCompleted = downloadedBytes >= expectedChunkSize
                } catch {
                    ModelDownloadLogger.error("Failed to get chunk attributes: \(error)")
                }
            }
            
            chunks.append(ChunkInfo(
                index: i,
                startOffset: startOffset,
                endOffset: endOffset,
                tempURL: tempURL,
                isCompleted: isCompleted,
                downloadedBytes: downloadedBytes
            ))
        }
        
        return chunks
    }
    
    // MARK: - Download Execution
    
    /// Executes download tasks with dynamic concurrency management
    /// 
    /// Manages the concurrent execution of download tasks using semaphores
    /// and dynamic concurrency adjustment based on network performance.
    /// Handles both chunked and direct download strategies.
    /// 
    /// - Throws: ModelScopeError if downloads fail or are cancelled
    private func executeDownloads() async throws {
        await withTaskGroup(of: Void.self) { group in
            for task in downloadQueue {
                if isCancelled { break }
                
                group.addTask {
                    await self.downloadSemaphore.wait()
                    defer { Task { await self.downloadSemaphore.signal() } }
                    
                    do {
                        if task.isChunked {
                            try await self.downloadFileInChunks(task: task)
                        } else {
                            try await self.downloadFileDirect(task: task)
                        }
                        
                        // Only mark as completed if not cancelled
                        if await !self.isCancelled {
                            await self.markFileCompleted(task: task)
                        }
                    } catch {
                        if await !self.isCancelled {
                            ModelDownloadLogger.error("Failed to download \(task.file.name): \(error)")
                        }
                    }
                }
            }
        }
    }
    
    /// Downloads a file using chunked strategy with resume capability
    /// 
    /// Handles the download of individual file chunks with retry logic,
    /// progress tracking, and automatic chunk merging upon completion.
    /// Uses optimal concurrency for chunk downloads.
    /// 
    /// - Parameter task: DownloadTask containing chunk information and file metadata
    /// - Throws: ModelScopeError for network or file system errors
    private func downloadFileInChunks(task: DownloadTask) async throws {
        ModelDownloadLogger.info("Starting chunked download for: \(task.file.name)")
        
        let concurrencyCount = await concurrencyManager.calculateOptimalConcurrency(chunkCount: task.chunks.count)
        
        downloadChunkSemaphore = AsyncSemaphore(value: concurrencyCount)
        
        ModelDownloadLogger.info("Using optimal concurrency: \(concurrencyCount) for \(task.chunks.count) chunks")
        
        // Check if any chunks are already completed and update progress
        let completedBytes = task.chunks.reduce(0) { total, chunk in
            return total + (chunk.isCompleted ? (chunk.endOffset - chunk.startOffset + 1) : chunk.downloadedBytes)
        }
        if completedBytes > 0 {
            await updateDownloadProgress(bytes: completedBytes)
            ModelDownloadLogger.info("Found \(completedBytes) bytes already downloaded for \(task.file.name)")
        }
        
        try await withThrowingTaskGroup(of: Void.self) { group in
            for chunk in task.chunks {
                if isCancelled { break }
                await self.downloadChunkSemaphore?.wait()
                defer { Task { await self.downloadChunkSemaphore?.signal() } }
                
                if !chunk.isCompleted {
                    group.addTask {
                        try await self.downloadChunk(chunk: chunk, file: task.file)
                    }
                }
            }
            
            try await group.waitForAll()
        }
        
        if isCancelled {
            throw ModelScopeError.downloadCancelled
        }
        
        // Merge chunks
        try await mergeChunks(task: task)
    }
    
    /// Downloads a specific chunk with range requests and resume support
    /// 
    /// Performs HTTP range request to download a specific portion of a file,
    /// with automatic resume from existing partial downloads and exponential
    /// backoff retry logic.
    /// 
    /// - Parameters:
    ///   - chunk: ChunkInfo containing chunk metadata and temporary file location
    ///   - file: ModelFile object representing the source file
    /// - Throws: ModelScopeError for network or file system errors
    private func downloadChunk(chunk: ChunkInfo, file: ModelFile) async throws {
        
        if chunk.isCompleted {
            ModelDownloadLogger.info("Chunk \(chunk.index) already completed, skipping")
            return
        }
        
        var lastError: Error?
        
        // Retry logic with exponential backoff
        for attempt in 0..<config.maxRetries {
            do {
                let url = try buildDownloadURL(for: file)
                var request = URLRequest(url: url)
                
                // Calculate resume offset within this chunk
                let resumeOffset = chunk.startOffset + chunk.downloadedBytes
                let remainingEndOffset = chunk.endOffset
                
                // Set Range header for resumable download
                if chunk.downloadedBytes > 0 {
                    request.setValue("bytes=\(resumeOffset)-\(remainingEndOffset)", forHTTPHeaderField: "Range")
                    ModelDownloadLogger.info("Resuming chunk \(chunk.index) from offset \(chunk.downloadedBytes)")
                } else {
                    request.setValue("bytes=\(chunk.startOffset)-\(chunk.endOffset)", forHTTPHeaderField: "Range")
                }
                
                let (asyncBytes, response) = try await session.bytes(for: request)
                try validateResponse(response)
                
                // Create or open file handle for writing
                if chunk.downloadedBytes == 0 && !fileManager.fileExists(atPath: chunk.tempURL.path) {
                    fileManager.createFile(atPath: chunk.tempURL.path, contents: nil, attributes: nil)
                }
                
                let fileHandle = try FileHandle(forWritingTo: chunk.tempURL)
                defer { try? fileHandle.close() }
                
                if chunk.downloadedBytes > 0 {
                    try fileHandle.seekToEnd()
                }
                
                var bytesCount = 0
                
                for try await byte in asyncBytes {
                    if isCancelled { throw ModelScopeError.downloadCancelled }
                    
                    try fileHandle.write(contentsOf: [byte])
                    bytesCount += 1
                    
                    if bytesCount >= 64 * 1024 {
                        await updateDownloadProgress(bytes: Int64(bytesCount))
                        bytesCount = 0
                    }
                }
                
                if bytesCount > 0 {
                    await updateDownloadProgress(bytes: Int64(bytesCount))
                }

                ModelDownloadLogger.info("Chunk \(chunk.index) downloaded successfully")
                
                return // Success, exit retry loop
                
            } catch {
                lastError = error
                ModelDownloadLogger.error("Chunk \(chunk.index) download attempt \(attempt + 1) failed: \(error)")
                
                // Don't retry if cancelled
                if isCancelled {
                    throw ModelScopeError.downloadCancelled
                }
                
                // Don't wait after the last attempt
                if attempt < config.maxRetries - 1 {
                    let delay = config.retryDelay * pow(2.0, Double(attempt)) // Exponential backoff
                    ModelDownloadLogger.info("Retrying chunk \(chunk.index) in \(delay) seconds...")
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
        }
        
        // All retries failed
        throw lastError ?? ModelScopeError.downloadFailed(NSError(
            domain: "ModelScope",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "All chunk download attempts failed"]
        ))
    }
    
    /// Downloads a file directly without chunking
    /// 
    /// Performs a complete file download in a single request with resume
    /// capability and progress tracking for smaller files. Uses exponential
    /// backoff retry mechanism for failed attempts.
    /// 
    /// - Parameter task: DownloadTask containing file information and destination
    /// - Throws: ModelScopeError for network or file system errors
    private func downloadFileDirect(task: DownloadTask) async throws {
        ModelDownloadLogger.info("downloadFileDirect \(task.file.name)")
        
        let file = task.file
        let destination = URL(fileURLWithPath: task.destinationPath)
            .appendingPathComponent(file.name.sanitizedPath)
        
        let modelHash = repoPath.hash
        let fileHash = file.path.hash
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("model_\(modelHash)_file_\(fileHash)_\(file.name.sanitizedPath).tmp")
        
        var lastError: Error?
        
        // Retry logic with exponential backoff
        for attempt in 0..<config.maxRetries {
            do {
                // Check for resume
                var resumeOffset: Int64 = 0
                if fileManager.fileExists(atPath: tempURL.path) {
                    let attributes = try fileManager.attributesOfItem(atPath: tempURL.path)
                    resumeOffset = attributes[.size] as? Int64 ?? 0
                } else {
                    try Data().write(to: tempURL)
                }
                
                let url = try buildDownloadURL(for: file)
                var request = URLRequest(url: url)
                if resumeOffset > 0 {
                    request.setValue("bytes=\(resumeOffset)-", forHTTPHeaderField: "Range")
                }
                
                let (asyncBytes, response) = try await session.bytes(for: request)
                try validateResponse(response)
                
                let fileHandle = try FileHandle(forWritingTo: tempURL)
                defer { try? fileHandle.close() }
                
                if resumeOffset > 0 {
                    try fileHandle.seek(toOffset: UInt64(resumeOffset))
                }
                
                var downloadedBytes: Int64 = resumeOffset
                var bytesCount = 0
                
                for try await byte in asyncBytes {
                    if isCancelled { throw ModelScopeError.downloadCancelled }
                    
                    try fileHandle.write(contentsOf: [byte])
                    downloadedBytes += 1
                    bytesCount += 1
                    
                    // Update progress less frequently
                    if bytesCount >= 64 * 1024 {
                        await updateDownloadProgress(bytes: Int64(bytesCount))
                        bytesCount = 0
                    }
                }
                
                // Final progress update
                if bytesCount > 0 {
                    await updateDownloadProgress(bytes: Int64(bytesCount))
                }
                
                // Move to final destination
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.moveItem(at: tempURL, to: destination)
                
                ModelDownloadLogger.info("File \(file.name) downloaded successfully")
                return // Success, exit retry loop
                
            } catch {
                lastError = error
                ModelDownloadLogger.error("File \(file.name) download attempt \(attempt + 1) failed: \(error)")
                
                // Don't retry if cancelled
                if isCancelled {
                    throw ModelScopeError.downloadCancelled
                }
                
                // Don't wait after the last attempt
                if attempt < config.maxRetries - 1 {
                    let delay = config.retryDelay * pow(2.0, Double(attempt)) // Exponential backoff
                    ModelDownloadLogger.info("Retrying file \(file.name) in \(delay) seconds...")
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
        }
        
        // All retries failed
        throw lastError ?? ModelScopeError.downloadFailed(NSError(
            domain: "ModelScope",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "All file download attempts failed"]
        ))
    }
    
    /// Merges downloaded chunks into the final file
    /// 
    /// Combines all downloaded chunks in the correct order to create the
    /// final file, then cleans up temporary chunk files.
    /// 
    /// - Parameter task: DownloadTask containing chunk information and destination
    /// - Throws: ModelScopeError for file system errors during merging
    private func mergeChunks(task: DownloadTask) async throws {
        let destination = URL(fileURLWithPath: task.destinationPath)
            .appendingPathComponent(task.file.name.sanitizedPath)
        
        // Create destination file if it doesn't exist
        if !fileManager.fileExists(atPath: destination.path) {
            fileManager.createFile(atPath: destination.path, contents: nil, attributes: nil)
        }
        
        let finalFileHandle = try FileHandle(forWritingTo: destination)
        defer { try? finalFileHandle.close() }
        
        // Sort chunks by index and merge
        let sortedChunks = task.chunks.sorted { $0.index < $1.index }
        
        for chunk in sortedChunks {
            if isCancelled { throw ModelScopeError.downloadCancelled }
            
            let chunkData = try Data(contentsOf: chunk.tempURL)
            try finalFileHandle.write(contentsOf: chunkData)
            
            // Clean up chunk file
            try? fileManager.removeItem(at: chunk.tempURL)
        }
        
        ModelDownloadLogger.info("Successfully merged \(sortedChunks.count) chunks for \(task.file.name)")
    }
    
    // MARK: - Helper Methods
    
    private func ensureValidSession() async {
        if isSessionInvalidated {
            // Create a new session if the previous one was invalidated
            session = URLSession(configuration: sessionConfig)
            isSessionInvalidated = false
            ModelDownloadLogger.info("Created new URLSession after previous invalidation")
        }
        
        // Always reset the cancelled flag when ensuring valid session
        isCancelled = false
    }
    
    private func buildDownloadURL(for file: ModelFile) throws -> URL {
        if source == .modelScope {
            return try buildURL(
                path: "/repo",
                queryItems: [
                    URLQueryItem(name: "Revision", value: "master"),
                    URLQueryItem(name: "FilePath", value: file.path)
                ]
            )
        } else {
            return try buildModelerURL(
                path: file.path,
                queryItems: []
            )
        }
    }
    
    private func markFileCompleted(task: DownloadTask) async {
        progress.completedFiles += 1
        storage.saveFileStatus(task.file, at: task.destinationPath)
        ModelDownloadLogger.info("Completed: \(task.file.name) (\(progress.completedFiles)/\(progress.totalFiles))")
    }
    
    private func updateDownloadProgress(bytes: Int64) async {
        progress.downloadedBytes += bytes
        await updateProgress(progress.progress)
    }
    
    private func updateProgress(_ value: Double) async {
        guard let callback = progressCallback else { return }
        
        Task { @MainActor in
            callback(value)
        }
    }
    
    // MARK: - Network Methods
    
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
