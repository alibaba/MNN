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
    case modeler = "Modelers"

    public var id: Self { self }

    var description: String {
        switch self {
        case .modelScope:
            return AppText.text("Use ModelScope to download", comment: "")
        case .modeler:
            return AppText.text("Use Modelers to download", comment: "")
        case .huggingFace:
            return AppText.text("Use HuggingFace to download", comment: "")
        }
    }
}
