//
//  TestInstance.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation
import Combine

/**
 * Observable class representing a single benchmark test instance.
 * Contains test configuration parameters and stores timing results.
 */
class TestInstance: ObservableObject, Identifiable {
    let id = UUID()
    let modelConfigFile: String
    let modelType: String
    let modelSize: Int64
    let threads: Int
    let useMmap: Bool
    let nPrompt: Int
    let nGenerate: Int
    let backend: Int
    let precision: Int
    let power: Int
    let memory: Int
    let dynamicOption: Int
    
    @Published var prefillUs: [Int64] = []
    @Published var decodeUs: [Int64] = []
    @Published var samplesUs: [Int64] = []
    
    init(modelConfigFile: String,
         modelType: String,
         modelSize: Int64 = 0,
         threads: Int,
         useMmap: Bool,
         nPrompt: Int,
         nGenerate: Int,
         backend: Int,
         precision: Int,
         power: Int,
         memory: Int,
         dynamicOption: Int) {
        self.modelConfigFile = modelConfigFile
        self.modelType = modelType
        self.modelSize = modelSize
        self.threads = threads
        self.useMmap = useMmap
        self.nPrompt = nPrompt
        self.nGenerate = nGenerate
        self.backend = backend
        self.precision = precision
        self.power = power
        self.memory = memory
        self.dynamicOption = dynamicOption
    }
    
    /// Calculates tokens per second from timing data
    /// - Parameters:
    ///   - tokens: Number of tokens processed
    ///   - timesUs: Array of timing measurements in microseconds
    /// - Returns: Array of tokens per second calculations
    func getTokensPerSecond(tokens: Int, timesUs: [Int64]) -> [Double] {
        return timesUs.compactMap { timeUs in
            guard timeUs > 0 else { return 0.0 }
            return Double(tokens) * 1_000_000.0 / Double(timeUs)
        }
    }
}
