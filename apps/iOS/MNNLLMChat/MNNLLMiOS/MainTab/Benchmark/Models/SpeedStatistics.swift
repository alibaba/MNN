//
//  SpeedStatistics.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/// Structure containing statistical analysis of benchmark speed metrics.
/// Provides average, standard deviation, and descriptive label for performance data.
struct SpeedStatistics {
    let average: Double
    let stdev: Double
    let label: String
    
    init(average: Double, stdev: Double, label: String) {
        self.average = average
        self.stdev = stdev
        self.label = label
    }
}
