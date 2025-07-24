//
//  ModelItem.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/**
 * Structure representing a model item with download state information.
 * Used for tracking model availability and download progress in the benchmark interface.
 */
struct ModelItem: Identifiable, Equatable {
    let id = UUID()
    let modelId: String
    let displayName: String
    let isLocal: Bool
    let localPath: String?
    let size: Int64?
    let downloadState: DownloadState
    
    enum DownloadState: Equatable {
        case notStarted
        case downloading(progress: Double)
        case completed
        case failed(error: String)
        case paused
    }
    
    static func == (lhs: ModelItem, rhs: ModelItem) -> Bool {
        return lhs.modelId == rhs.modelId
    }
}
