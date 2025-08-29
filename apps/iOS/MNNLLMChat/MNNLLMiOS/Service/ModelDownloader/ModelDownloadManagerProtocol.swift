//
//  ModelDownloadManagerProtocol.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/8/27.
//

import Foundation

/// Protocol defining the interface for model download managers
/// 
/// This protocol enables dependency injection and provides a common contract for different download implementations.
/// It supports concurrent downloads, progress tracking, and cancellation functionality.
/// 
/// Key Features:
/// - Asynchronous download operations with progress callbacks
/// - Cancellation support with resume capability
/// - Actor-based thread safety
/// - Flexible destination path configuration
/// 
/// Usage:
/// ```swift
/// let manager: ModelDownloadManagerProtocol = ModelDownloadManager(...)
/// try await manager.downloadModel(
///     to: "models",
///     modelId: "example-model",
///     modelName: "ExampleModel"
/// ) { progress in
///     print("Progress: \(progress * 100)%")
/// }
/// ```
@available(iOS 13.4, macOS 10.15, *)
public protocol ModelDownloadManagerProtocol: Actor, Sendable {
    
    /// Downloads a model from the repository with progress tracking
    /// 
    /// This method initiates an asynchronous download operation for the specified model.
    /// It supports resume functionality and provides real-time progress updates.
    /// 
    /// - Parameters:
    ///   - destinationFolder: Base folder where the model will be downloaded
    ///   - modelId: Unique identifier for the model in the repository
    ///   - modelName: Display name of the model (used for folder creation)
    ///   - progress: Optional closure called with download progress (0.0 to 1.0)
    /// - Throws: ModelScopeError for network, file system, or validation failures
    func downloadModel(
        to destinationFolder: String,
        modelId: String,
        modelName: String,
        progress: ((Double) -> Void)?
    ) async throws
    
    /// Cancels the current download operation
    /// 
    /// This method gracefully stops the download process while preserving temporary files
    /// to support resume functionality in future download attempts.
    func cancelDownload() async
}

/// Type-erased wrapper for ModelDownloadManagerProtocol
/// 
/// This class provides a concrete type that can be stored as a property while maintaining protocol flexibility.
/// It wraps any ModelDownloadManagerProtocol implementation and forwards method calls to the underlying manager.
/// 
/// Key Benefits:
/// - Enables storing protocol instances as properties
/// - Maintains type safety while providing flexibility
/// - Supports dependency injection patterns
/// - Preserves all protocol functionality
/// 
/// Usage:
/// ```swift
/// let concreteManager = ModelDownloadManager(...)
/// let anyManager = AnyModelDownloadManager(concreteManager)
/// // Can now store anyManager as a property
/// ```
@available(iOS 13.4, macOS 10.15, *)
public actor AnyModelDownloadManager: ModelDownloadManagerProtocol {
    
    private let _downloadModel: (String, String, String, ((Double) -> Void)?) async throws -> Void
    private let _cancelDownload: () async -> Void
    
    /// Creates a type-erased wrapper around any ModelDownloadManagerProtocol implementation
    /// 
    /// - Parameter manager: The concrete download manager to wrap
    public init<T: ModelDownloadManagerProtocol>(_ manager: T) {
        self._downloadModel = { destinationFolder, modelId, modelName, progress in
            try await manager.downloadModel(
                to: destinationFolder,
                modelId: modelId,
                modelName: modelName,
                progress: progress
            )
        }
        self._cancelDownload = {
            await manager.cancelDownload()
        }
    }
    
    public func downloadModel(
        to destinationFolder: String,
        modelId: String,
        modelName: String,
        progress: ((Double) -> Void)? = nil
    ) async throws {
        try await _downloadModel(destinationFolder, modelId, modelName, progress)
    }
    
    public func cancelDownload() async {
        await _cancelDownload()
    }
}

/// Factory protocol for creating download managers
/// 
/// This protocol enables different creation strategies while maintaining type safety.
/// It provides a standardized way to create download managers with various configurations.
/// 
/// Key Features:
/// - Supports multiple download manager implementations
/// - Enables dependency injection and testing
/// - Provides consistent creation interface
/// - Supports different model sources
/// 
/// Usage:
/// ```swift
/// let factory: ModelDownloadManagerFactory = DefaultModelDownloadManagerFactory()
/// let manager = factory.createDownloadManager(
///     repoPath: "owner/model-name",
///     source: .modelScope
/// )
/// ```
@available(iOS 13.4, macOS 10.15, *)
public protocol ModelDownloadManagerFactory {
    
    /// Creates a download manager for the specified repository and source
    /// 
    /// - Parameters:
    ///   - repoPath: Repository path in format "owner/model-name"
    ///   - source: The model source (ModelScope or Modeler)
    /// - Returns: A download manager conforming to ModelDownloadManagerProtocol
    func createDownloadManager(
        repoPath: String,
        source: ModelSource
    ) -> any ModelDownloadManagerProtocol
}

/// Default factory implementation that creates OptimizedModelScopeDownloadManager instances
/// 
/// This factory provides the standard implementation for creating download managers with
/// advanced features including dynamic concurrency control, optimized performance settings,
/// and comprehensive configuration options.
@available(iOS 13.4, macOS 10.15, *)
public struct DefaultModelDownloadManagerFactory: ModelDownloadManagerFactory {
    
    private let config: DownloadConfig
    private let sessionConfig: URLSessionConfiguration
    private let enableLogging: Bool
    private let concurrencyConfig: DynamicConcurrencyConfig
    
    /// Creates a factory with specified configuration
    /// 
    /// - Parameters:
    ///   - config: Download configuration settings
    ///   - sessionConfig: URLSession configuration
    ///   - enableLogging: Whether to enable debug logging
    ///   - concurrencyConfig: Dynamic concurrency configuration
    public init(
        config: DownloadConfig = .default,
        sessionConfig: URLSessionConfiguration = .default,
        enableLogging: Bool = true,
        concurrencyConfig: DynamicConcurrencyConfig = .default
    ) {
        self.config = config
        self.sessionConfig = sessionConfig
        self.enableLogging = enableLogging
        self.concurrencyConfig = concurrencyConfig
    }
    
    public func createDownloadManager(
        repoPath: String,
        source: ModelSource
    ) -> any ModelDownloadManagerProtocol {
        return ModelDownloadManager(
            repoPath: repoPath,
            config: config,
            sessionConfig: sessionConfig,
            enableLogging: enableLogging,
            source: source,
            concurrencyConfig: concurrencyConfig
        )
    }
}

/// Legacy factory implementation that creates ModelScopeDownloadManager instances
/// 
/// This factory is provided for backward compatibility and testing purposes.
/// It creates the original ModelScopeDownloadManager without advanced optimizations.
@available(iOS 13.4, macOS 10.15, *)
public struct LegacyModelDownloadManagerFactory: ModelDownloadManagerFactory {
    
    private let sessionConfig: URLSessionConfiguration
    private let enableLogging: Bool
    
    /// Creates a legacy factory with specified configuration
    /// 
    /// - Parameters:
    ///   - sessionConfig: URLSession configuration
    ///   - enableLogging: Whether to enable debug logging
    public init(
        sessionConfig: URLSessionConfiguration = .default,
        enableLogging: Bool = true
    ) {
        self.sessionConfig = sessionConfig
        self.enableLogging = enableLogging
    }
    
    public func createDownloadManager(
        repoPath: String,
        source: ModelSource
    ) -> any ModelDownloadManagerProtocol {
        return ModelScopeDownloadManager(
            repoPath: repoPath,
            config: sessionConfig,
            enableLogging: enableLogging,
            source: source
        )
    }
}
