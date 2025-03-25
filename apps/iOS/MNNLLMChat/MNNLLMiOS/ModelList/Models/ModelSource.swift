//
//  ModelDownloadStorage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

public enum ModelSource: String, CaseIterable {
    case modelScope = "ModelScope"
    case huggingFace = "Hugging Face"
    case modeler = "Modeler"
    
    var description: String {
        switch self {
        case .modelScope:
            return NSLocalizedString("Use ModelScope to download", comment: "")
        case .modeler:
            return NSLocalizedString("Use modeler to download", comment: "")
        case .huggingFace:
            return NSLocalizedString("Use HuggingFace to download", comment: "")
        }
    }
} 
