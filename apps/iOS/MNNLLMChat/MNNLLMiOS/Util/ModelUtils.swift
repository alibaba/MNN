//
//  ModelUtils.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/8.
//

import Foundation

class ModelUtils {
    
    /// Check if the model supports thinking mode switching
    /// - Parameter tags: Model tags
    /// - Returns: Whether thinking mode switching is supported
    static func isSupportThinkingSwitch(_ tags: [String], modelName: String) -> Bool {
        
        return isQwen3(modelName) && !modelName.contains("2507") && (tags.contains(where: { $0.localizedCaseInsensitiveContains("Think") }) ||
               tags.contains(where: { $0.localizedCaseInsensitiveContains("思考") }))
    }
    
    /// Check if it's an R1 model
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's an R1 model
    static func isR1Model(_ modelName: String) -> Bool {
        return modelName.lowercased().contains("deepseek-r1")
    }
    
    /// Check if it's a Qwen3 model
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's a Qwen3 model
    static func isQwen3(_ modelName: String) -> Bool {
        return modelName.lowercased().contains("qwen3")
    }
    
    /// Check if it's an audio model
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's an audio model
    static func isAudioModel(_ modelName: String) -> Bool {
        return modelName.lowercased().contains("audio")
    }
    
    
    static func isThinkMode(_ modelTag: Array<Any>) -> Bool {
        
        return false
    }
    
    /// Check if it's a visual model
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's a visual model
    static func isVisualModel(_ modelName: String) -> Bool {
        return modelName.lowercased().contains("vl") || isOmni(modelName)
    }
    
    /// Check if it's an Omni model
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's an Omni model
    static func isOmni(_ modelName: String) -> Bool {
        return modelName.lowercased().contains("omni")
    }
    
    /// Check if it's a diffusion model
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's a diffusion model
    static func isDiffusionModel(_ modelName: String) -> Bool {
        return modelName.lowercased().contains("stable-diffusion")
    }
    
    /// Check if audio output is supported
    /// - Parameter modelName: Model name
    /// - Returns: Whether audio output is supported
    static func supportAudioOutput(_ modelName: String) -> Bool {
        return isOmni(modelName)
    }
}
