//
//  UIUpdateOptimizer.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/7.
//

import Foundation
import SwiftUI

/// UIUpdateOptimizer - A utility for batching and throttling UI updates to improve performance
///
/// This actor-based optimizer helps reduce the frequency of UI updates by batching multiple
/// updates together and applying throttling mechanisms. It's particularly useful for scenarios
/// like streaming text updates, real-time data feeds, or any situation where frequent UI
/// updates might cause performance issues.
///
/// Key Features:
/// - Batches multiple updates into a single operation
/// - Applies time-based throttling to limit update frequency
/// - Thread-safe actor implementation
/// - Automatic flush mechanism for pending updates
///
/// Usage Example:
/// ```swift
/// // For streaming text updates
/// await UIUpdateOptimizer.shared.addUpdate(newText) { batchedContent in
///     // Update UI with batched content
///     textView.text = batchedContent
/// }
///
/// // Force flush remaining updates when stream ends
/// await UIUpdateOptimizer.shared.forceFlush { finalContent in
///     textView.text = finalContent
/// }
/// ```
///
/// Configuration:
/// - batchSize: Number of updates to batch before triggering immediate flush (default: 5)
/// - flushInterval: Time interval in seconds between automatic flushes (default: 0.03s / 30ms)
actor UIUpdateOptimizer {
    static let shared = UIUpdateOptimizer()
    
    private var pendingUpdates: [String] = []
    private var lastFlushTime: Date = Date()
    private var flushTask: Task<Void, Never>?
    
    // Configuration constants
    private let batchSize: Int = 5          // Batch size threshold for immediate flush
    private let flushInterval: TimeInterval = 0.5
    
    private init() {}
    
    /// Adds a content update to the pending queue
    ///
    /// Updates are either flushed immediately if batch size or time threshold is reached,
    /// or scheduled for delayed flushing to optimize performance.
    ///
    /// - Parameters:
    ///   - content: The content string to add to the update queue
    ///   - completion: Callback executed with the batched content when flushed
    func addUpdate(_ content: String, completion: @escaping (String) -> Void) {
        pendingUpdates.append(content)
        
        // Determine if immediate flush is needed based on batch size or time interval
        let shouldFlushImmediately = pendingUpdates.count >= batchSize ||
                                   Date().timeIntervalSince(lastFlushTime) >= flushInterval
        
        if shouldFlushImmediately {
            flushUpdates(completion: completion)
        } else {
            // Schedule delayed flush to optimize performance
            scheduleFlush(completion: completion)
        }
    }
    
    /// Schedules a delayed flush operation
    ///
    /// Cancels any existing scheduled flush and creates a new one to avoid
    /// excessive flush operations while maintaining responsiveness.
    ///
    /// - Parameter completion: Callback to execute when flush occurs
    private func scheduleFlush(completion: @escaping (String) -> Void) {
        // Cancel previous scheduled flush to avoid redundant operations
        flushTask?.cancel()
        
        flushTask = Task {
            try? await Task.sleep(nanoseconds: UInt64(flushInterval * 1_000_000_000))
            
            if !Task.isCancelled && !pendingUpdates.isEmpty {
                flushUpdates(completion: completion)
            }
        }
    }
    
    /// Flushes all pending updates immediately
    ///
    /// Combines all pending updates into a single string and executes the completion
    /// callback on the main actor thread for UI updates.
    ///
    /// - Parameter completion: Callback executed with the combined content
    private func flushUpdates(completion: @escaping (String) -> Void) {
        guard !pendingUpdates.isEmpty else { return }
        
        let batchedContent = pendingUpdates.joined()
        pendingUpdates.removeAll()
        lastFlushTime = Date()
        
        Task { @MainActor in
            completion(batchedContent)
        }
    }
    
    /// Forces immediate flush of any remaining pending updates
    ///
    /// This method should be called when you need to ensure all pending updates
    /// are processed immediately, such as when a stream ends or the view is about
    /// to disappear.
    ///
    /// - Parameter completion: Callback executed with any remaining content
    func forceFlush(completion: @escaping (String) -> Void) {
        if !pendingUpdates.isEmpty {
            flushUpdates(completion: completion)
        }
    }
}
