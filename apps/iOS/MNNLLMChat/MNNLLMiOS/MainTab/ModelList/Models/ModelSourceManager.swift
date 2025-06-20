//
//  ModelDownloadStorage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

final class ModelSourceManager {
    static let shared = ModelSourceManager()
    
    private init() {
        loadSelectedSource()
    }
    
    private(set) var selectedSource: ModelSource = .modelScope // Default source

    private func loadSelectedSource() {
        // Check user defaults for previously saved source
        if let savedSource = UserDefaults.standard.string(forKey: "selectedSource"),
           let source = ModelSource(rawValue: savedSource) {
            selectedSource = source
        } else {
            // Determine default source based on language
            let preferredLanguage = Locale.preferredLanguages.first ?? "en"
            selectedSource = preferredLanguage.starts(with: "zh") ? .modelScope : .huggingFace
        }
    }
    
    func updateSelectedSource(_ source: ModelSource) {
        selectedSource = source
        UserDefaults.standard.set(source.rawValue, forKey: "selectedSource")
    }
} 
