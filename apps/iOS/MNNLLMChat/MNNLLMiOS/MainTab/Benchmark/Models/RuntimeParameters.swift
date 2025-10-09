//
//  RuntimeParameters.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/// Configuration parameters for benchmark runtime environment.
/// Defines hardware and execution settings for benchmark tests.
struct RuntimeParameters {
    let backends: [Int]
    let threads: [Int]
    let useMmap: Bool
    let power: [Int]
    let precision: [Int]
    let memory: [Int]
    let dynamicOption: [Int]
    
    static let `default` = RuntimeParameters(
        backends: [0], // CPU
        threads: [4],
        useMmap: false,
        power: [0],
        precision: [2], // Low precision
        memory: [2], // Low memory
        dynamicOption: [0]
    )
}
