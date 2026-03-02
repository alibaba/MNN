//
//  BenchmarkProgress.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/// Structure containing detailed progress information for benchmark execution.
/// Provides real-time metrics including timing data and performance statistics.
struct BenchmarkProgress {
    let progress: Int // 0-100
    let statusMessage: String
    let progressType: ProgressType
    let currentIteration: Int
    let totalIterations: Int
    let nPrompt: Int
    let nGenerate: Int
    let runTimeSeconds: Float
    let prefillTimeSeconds: Float
    let decodeTimeSeconds: Float
    let prefillSpeed: Float
    let decodeSpeed: Float
    
    init(progress: Int,
         statusMessage: String,
         progressType: ProgressType = .unknown,
         currentIteration: Int = 0,
         totalIterations: Int = 0,
         nPrompt: Int = 0,
         nGenerate: Int = 0,
         runTimeSeconds: Float = 0.0,
         prefillTimeSeconds: Float = 0.0,
         decodeTimeSeconds: Float = 0.0,
         prefillSpeed: Float = 0.0,
         decodeSpeed: Float = 0.0) {
        self.progress = progress
        self.statusMessage = statusMessage
        self.progressType = progressType
        self.currentIteration = currentIteration
        self.totalIterations = totalIterations
        self.nPrompt = nPrompt
        self.nGenerate = nGenerate
        self.runTimeSeconds = runTimeSeconds
        self.prefillTimeSeconds = prefillTimeSeconds
        self.decodeTimeSeconds = decodeTimeSeconds
        self.prefillSpeed = prefillSpeed
        self.decodeSpeed = decodeSpeed
    }
}
