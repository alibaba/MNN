//
//  ModelListManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation

/**
 * Manager class for integrating with ModelListViewModel to provide
 * downloaded models for benchmark testing.
 */
class ModelListManager {
    static let shared = ModelListManager()
    
    private let modelListViewModel = ModelListViewModel()
    
    private init() {}
    
    /// Loads available models, filtering for downloaded models suitable for benchmarking
    /// - Returns: Array of downloaded ModelInfo objects
    /// - Throws: Error if model loading fails
    func loadModels() async throws -> [ModelInfo] {
        // Ensure models are loaded from the view model
        await modelListViewModel.fetchModels()
        
        // Return only downloaded models that are available for benchmark
        return modelListViewModel.models.filter { $0.isDownloaded }
    }
}
