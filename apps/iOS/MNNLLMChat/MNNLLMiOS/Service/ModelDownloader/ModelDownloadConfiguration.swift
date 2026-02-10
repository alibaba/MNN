//
//  ModelDownloadConfiguration.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/8/27.
//

import Foundation

public struct DownloadConfig {
    let maxConcurrentDownloads: Int
    let chunkSize: Int64
    let largeFileThreshold: Int64
    let maxRetries: Int
    let retryDelay: TimeInterval
    
    public static let `default` = DownloadConfig(
        maxConcurrentDownloads: 3,
        chunkSize: 20 * 1024 * 1024, // 20MB chunks
        largeFileThreshold: 100 * 1024 * 1024, // 100MB threshold for chunking
        maxRetries: 500,
        retryDelay: 1.0
    )
}

// MARK: - Download Task

struct DownloadTask {
    let file: ModelFile
    let destinationPath: String
    let priority: TaskPriority
    var chunks: [ChunkInfo] = []
    var isChunked: Bool { !chunks.isEmpty }
}

struct ChunkInfo {
    let index: Int
    let startOffset: Int64
    let endOffset: Int64
    let tempURL: URL
    var isCompleted: Bool = false
    var downloadedBytes: Int64 = 0
}

// MARK: - Progress Tracking

struct DownloadProgress {
    var totalBytes: Int64 = 0
    var activeDownloads: Int = 0
    var completedFiles: Int = 0
    var totalFiles: Int = 0
    
    // Track individual file progress
    var fileProgress: [String: FileDownloadProgress] = [:]
    var lastReportedProgress: Double = 0.0
    
    var progress: Double {
        guard totalBytes > 0 else { return 0.0 }
        
        let totalDownloadedBytes = fileProgress.values.reduce(0) { sum, fileProgress in
            return sum + fileProgress.downloadedBytes
        }
        
        let calculatedProgress = Double(totalDownloadedBytes) / Double(totalBytes)
        return min(calculatedProgress, 1.0) // Ensure progress never exceeds 100%
    }
}

struct FileDownloadProgress {
    let fileName: String
    let totalBytes: Int64
    var downloadedBytes: Int64 = 0
    var isCompleted: Bool = false
    
    var progress: Double {
        guard totalBytes > 0 else { return 0.0 }
        return Double(downloadedBytes) / Double(totalBytes)
    }
}
