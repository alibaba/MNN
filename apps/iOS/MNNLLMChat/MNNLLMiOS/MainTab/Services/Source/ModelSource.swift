//
//  ModelSource.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

public enum ModelSource: String, CaseIterable, Identifiable {
    case modelScope = "ModelScope"
    case huggingFace = "HuggingFace"
    case modeler = "Modeler"
    
    public var id: Self { self }
    
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
