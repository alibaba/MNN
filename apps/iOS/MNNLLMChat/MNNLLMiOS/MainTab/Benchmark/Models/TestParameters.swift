//
//  TestParameters.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/**
 * Configuration parameters for benchmark test execution.
 * Defines test scenarios including prompt sizes, generation lengths, and repetition counts.
 */
struct TestParameters {
    let nPrompt: [Int]
    let nGenerate: [Int]
    let nPrompGen: [(Int, Int)]
    let nRepeat: [Int]
    let kvCache: String
    let loadTime: String
    
    static let `default` = TestParameters(
        nPrompt: [256, 512],
        nGenerate: [64, 128],
        nPrompGen: [(256, 64), (512, 128)],
        nRepeat: [3], // Reduced for mobile
        kvCache: "false", // llama-bench style test by default
        loadTime: "false"
    )
}
