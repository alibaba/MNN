//
//  PerformanceMonitor.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Foundation
import UIKit

/**
 * PerformanceMonitor - A singleton utility for monitoring and measuring UI performance
 * 
 * This class provides real-time performance monitoring capabilities to help identify
 * UI update bottlenecks, frame drops, and slow operations in iOS applications.
 * It's particularly useful during development to ensure smooth user experience.
 * 
 * Key Features:
 * - Real-time FPS monitoring and frame drop detection
 * - UI update lag detection with customizable thresholds
 * - Execution time measurement for specific operations
 * - Automatic performance statistics reporting
 * - Thread-safe singleton implementation
 * 
 * Usage Examples:
 * 
 * 1. Monitor UI Updates:
 * ```swift
 * // Call this in your UI update methods
 * PerformanceMonitor.shared.recordUIUpdate()
 * ```
 * 
 * 2. Measure Operation Performance:
 * ```swift
 * let result = PerformanceMonitor.shared.measureExecutionTime(operation: "Data Processing") {
 *     // Your expensive operation here
 *     return processLargeDataSet()
 * }
 * ```
 * 
 * 3. Integration in ViewModels:
 * ```swift
 * func updateUI() {
 *     PerformanceMonitor.shared.recordUIUpdate()
 *     // Your UI update code
 * }
 * ```
 * 
 * Performance Thresholds:
 * - Target FPS: 60 FPS
 * - Frame threshold: 25ms (1.5x normal frame time)
 * - Slow operation threshold: 16ms (1 frame time)
 */
class PerformanceMonitor {
    static let shared = PerformanceMonitor()
    
    private var lastUpdateTime: Date = Date()
    private var updateCount: Int = 0
    private var frameDropCount: Int = 0
    private let targetFPS: Double = 60.0
    private let frameThreshold: TimeInterval = 1.0 / 60.0 * 1.5 // Allow 1.5x normal frame time
    
    private init() {}
    
    /**
     * Records a UI update event and monitors performance metrics
     * 
     * Call this method whenever you perform UI updates to track performance.
     * It automatically detects frame drops and calculates FPS statistics.
     * Performance statistics are logged every second.
     */
    func recordUIUpdate() {
        let currentTime = Date()
        let timeDiff = currentTime.timeIntervalSince(lastUpdateTime)
        
        updateCount += 1
        
        // Detect frame drops
        if timeDiff > frameThreshold {
            frameDropCount += 1
            print("⚠️ UI Update Lag detected: \(timeDiff * 1000)ms (expected: \(frameThreshold * 1000)ms)")
        }
        
        // Report statistics every second
        if timeDiff >= 1.0 {
            let actualFPS = Double(updateCount) / timeDiff
            let dropRate = Double(frameDropCount) / Double(updateCount) * 100
            
            print("📊 Performance Stats - FPS: \(String(format: "%.1f", actualFPS)), Drop Rate: \(String(format: "%.1f", dropRate))%")
            
            // Reset counters for next measurement cycle
            updateCount = 0
            frameDropCount = 0
            lastUpdateTime = currentTime
        }
    }
    
    /**
     * Measures execution time for a specific operation
     * 
     * Wraps any operation and measures its execution time. Operations taking
     * longer than 16ms (1 frame time) are logged as slow operations.
     * 
     * - Parameters:
     *   - operation: A descriptive name for the operation being measured
     *   - block: The operation to measure
     * - Returns: The result of the operation
     * - Throws: Re-throws any error thrown by the operation
     */
    func measureExecutionTime<T>(operation: String, block: () throws -> T) rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let executionTime = CFAbsoluteTimeGetCurrent() - startTime
        
        if executionTime > 0.016 { // Over 16ms (1 frame time)
            print("⏱️ Slow Operation: \(operation) took \(String(format: "%.3f", executionTime * 1000))ms")
        }
        
        return result
    }
}