//
//  ModelDownloadStorage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

enum ModelSource: String, CaseIterable {
    case modelScope = "ModelScope"
    case huggingFace = "Hugging Face"
    
    var description: String {
        switch self {
        case .modelScope:
            return "Use ModelScope"
        case .huggingFace:
            return "Use Hugging Face"
        }
    }
} 
