//
//  DynamicConcurrencyManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/8/27.
//

import Foundation
import Network

// MARK: - Dynamic Concurrency Configuration

/// Dynamic concurrency control manager - intelligently adjusts concurrency parameters
/// based on chunk count and network conditions
/// 
/// Usage Examples:
///
/// ```swift
/// // 1. Create dynamic concurrency manager
/// let concurrencyManager = DynamicConcurrencyManager()
///
/// // 2. Get download strategy for file
/// let fileSize: Int64 = 240 * 1024 * 1024 // 240MB
/// let strategy = await concurrencyManager.recommendDownloadStrategy(fileSize: fileSize)
///
/// print(strategy.description)
/// // Output might be:
/// // Download Strategy:
/// // - Use Chunking: Yes
/// // - Chunk Size: 10MB
/// // - Chunk Count: 24
/// // - Concurrency: 6 (24 chunks / 4 ideal chunks per concurrency = 6)
/// // - Network Type: wifi
/// // - Device Performance: high
///
/// // 3. Create DownloadConfig using strategy
/// let dynamicConfig = DownloadConfig(
///     maxConcurrentDownloads: strategy.concurrency,
///     chunkSize: strategy.chunkSize,
///     largeFileThreshold: strategy.chunkSize * 2,
///     maxRetries: 3,
///     retryDelay: 2.0
/// )
/// ```
/// 
/// Best Practices:
///
/// 1. **Intelligent Concurrency Control**:
///    - For 24 chunks: use 6-8 concurrent downloads (instead of fixed 3)
///    - For 4 chunks: use 2-3 concurrent downloads
///    - For 1 chunk: use 1 concurrent download
///
/// 2. **Network Adaptation**:
///    - WiFi: larger chunk size and more concurrency
///    - 4G: medium chunk size and concurrency
///    - 3G: small chunk size and less concurrency
///
/// 3. **Device Performance Consideration**:
///    - High-performance devices: can handle more concurrency
///    - Low-performance devices: reduce concurrency to avoid lag
///
/// 4. **Dynamic Adjustment**:
///    - Automatically adjust strategy when network status changes
///    - Dynamically optimize based on actual download performance

/// Configuration for dynamic concurrency management
/// 
/// This structure defines the parameters used to dynamically adjust download concurrency
/// based on network conditions and device performance characteristics.
/// 
/// - Parameters:
///   - baseConcurrency: The baseline number of concurrent downloads
///   - maxConcurrency: The maximum number of concurrent downloads allowed
///   - minConcurrency: The minimum number of concurrent downloads to maintain
///   - idealChunksPerConcurrency: The ideal number of chunks per concurrent download
///   - networkTypeMultiplier: Multiplier for network type adjustments
///   - devicePerformanceMultiplier: Multiplier for device performance adjustments
///   - largeFileThreshold: File size threshold for enabling chunked downloads
public struct DynamicConcurrencyConfig {
    
    let baseConcurrency: Int
    
    let maxConcurrency: Int
    
    let minConcurrency: Int
    
    let idealChunksPerConcurrency: Int
    
    let networkTypeMultiplier: Double
    
    let devicePerformanceMultiplier: Double
    
    let largeFileThreshold: Int64
    
    public static let `default` = DynamicConcurrencyConfig(
        baseConcurrency: 3,
        maxConcurrency: 8,
        minConcurrency: 1,
        idealChunksPerConcurrency: 3,
        networkTypeMultiplier: 1.0,
        devicePerformanceMultiplier: 1.0,
        largeFileThreshold: 100 * 1024 * 1024
    )
}

// MARK: - Network Type Detection

/// Network type classification for optimization
/// 
/// Categorizes different network connection types to enable appropriate
/// download strategy selection and performance optimization.
public enum NetworkType {
    case wifi
    case cellular
    case lowBandwidth
    case unknown
    
    /// Multiplier for adjusting concurrency based on device performance
    /// 
    /// - Returns: A multiplier value used to scale the base concurrency level
    var concurrencyMultiplier: Double {
        switch self {
        case .wifi: return 1.5
        case .cellular: return 1.0
        case .lowBandwidth: return 0.5
        case .unknown: return 0.8
        }
    }
    
    var recommendedChunkSize: Int64 {
        switch self {
        case .wifi: return 20 * 1024 * 1024 // 20MB
        case .cellular: return 10 * 1024 * 1024 // 10MB
        case .lowBandwidth: return 5 * 1024 * 1024 // 5MB
        case .unknown: return 8 * 1024 * 1024 // 5MB
        }
    }
}

// MARK: - Device Performance Detection

/// Device performance classification
/// 
/// Categorizes device performance capabilities to optimize download strategies
/// based on available processing power and memory resources.
public enum DevicePerformance {
    case high
    case medium
    case low
    
    var concurrencyMultiplier: Double {
        switch self {
        case .high: return 1.3
        case .medium: return 1.0
        case .low: return 0.7
        }
    }
    
    /// Detect the current device's performance level
    /// 
    /// - Returns: The detected device performance classification
    static func detect() -> DevicePerformance {
        let processInfo = ProcessInfo.processInfo
        let physicalMemory = processInfo.physicalMemory
        let processorCount = processInfo.processorCount
        
        // Determine device performance based on memory and processor count
        if physicalMemory >= 6 * 1024 * 1024 * 1024 && processorCount >= 6 { // 6GB+ RAM, 6+ cores
            return .high
        } else if physicalMemory >= 3 * 1024 * 1024 * 1024 && processorCount >= 4 { // 3GB+ RAM, 4+ cores
            return .medium
        } else {
            return .low
        }
    }
}

// MARK: - Dynamic Concurrency Manager

@available(iOS 13.4, macOS 10.15, *)
public actor DynamicConcurrencyManager {
    private let config: DynamicConcurrencyConfig
    private let networkMonitor: NWPathMonitor
    private var currentNetworkType: NetworkType = .unknown
    private let devicePerformance: DevicePerformance
    
    public init(config: DynamicConcurrencyConfig = .default) {
        self.config = config
        self.networkMonitor = NWPathMonitor()
        self.devicePerformance = DevicePerformance.detect()
        
        Task {
            await startNetworkMonitoring()
        }
    }
    
    private func startNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            Task {
                await self?.updateNetworkType(from: path)
            }
        }
        
        let queue = DispatchQueue(label: "NetworkMonitor")
        networkMonitor.start(queue: queue)
    }
    
    private func updateNetworkType(from path: NWPath) {
        if path.usesInterfaceType(.wifi) {
            currentNetworkType = .wifi
        } else if path.usesInterfaceType(.cellular) {
            currentNetworkType = .cellular
        } else if path.status == .satisfied {
            currentNetworkType = .unknown
        } else {
            currentNetworkType = .lowBandwidth
        }
    }
    
    /// Calculate optimal concurrency based on chunk count and current network conditions
    /// 
    /// - Parameter chunkCount: The number of chunks to download
    /// - Returns: The recommended number of concurrent downloads
    public func calculateOptimalConcurrency(chunkCount: Int) -> Int {
        // Base calculation: based on chunk count and ideal ratio
        let baseConcurrency = max(1, min(chunkCount / config.idealChunksPerConcurrency, config.baseConcurrency))
        
        // Apply network type weight
        let networkAdjusted = Double(baseConcurrency) * currentNetworkType.concurrencyMultiplier
        
        // Apply device performance weight
        let performanceAdjusted = networkAdjusted * devicePerformance.concurrencyMultiplier
        
        // Ensure within reasonable range
        let finalConcurrency = Int(performanceAdjusted.rounded())
        
        return max(config.minConcurrency, min(config.maxConcurrency, finalConcurrency))
    }
    
    /// Get current network status information
    /// 
    /// - Returns: A tuple containing the current network type and device performance
    public func getNetworkInfo() -> (type: NetworkType, performance: DevicePerformance) {
        return (currentNetworkType, devicePerformance)
    }
    
    /// Get recommended chunk size based on current network conditions and device performance
    /// 
    /// - Returns: The recommended chunk size in bytes
    public func recommendedChunkSize() -> Int64 {
        let baseChunkSize = currentNetworkType.recommendedChunkSize
        let performanceMultiplier = devicePerformance.concurrencyMultiplier
        
        return Int64(Double(baseChunkSize) * performanceMultiplier)
    }
    
    /// Recommend download strategy based on file size and network conditions
    /// 
    /// - Parameter fileSize: The size of the file being downloaded
    /// - Returns: A complete download strategy configuration
    public func recommendDownloadStrategy(fileSize: Int64) -> DownloadStrategy {
        let chunkSize = recommendedChunkSize()
        let shouldUseChunking = fileSize > config.largeFileThreshold
        let chunkCount = shouldUseChunking ? Int(ceil(Double(fileSize) / Double(chunkSize))) : 1
        let optimalConcurrency = calculateOptimalConcurrency(chunkCount: chunkCount)
        
        return DownloadStrategy(
            shouldUseChunking: shouldUseChunking,
            chunkSize: chunkSize,
            chunkCount: chunkCount,
            concurrency: optimalConcurrency,
            networkType: currentNetworkType,
            devicePerformance: devicePerformance
        )
    }
    
    deinit {
        networkMonitor.cancel()
    }
}

// MARK: - Download Strategy

/// Download strategy configuration
/// 
/// Contains all the parameters needed to optimize download performance
/// based on current network and device conditions.
public struct DownloadStrategy {
    let shouldUseChunking: Bool
    let chunkSize: Int64
    let chunkCount: Int
    let concurrency: Int
    let networkType: NetworkType
    let devicePerformance: DevicePerformance
    
    var description: String {
        return """
        Download Strategy:
        - Use Chunking: \(shouldUseChunking ? "Yes" : "No")
        - Chunk Size: \(chunkSize / 1024 / 1024)MB
        - Chunk Count: \(chunkCount)
        - Concurrency: \(concurrency)
        - Network Type: \(networkType)
        - Device Performance: \(devicePerformance)
        """
    }
}
