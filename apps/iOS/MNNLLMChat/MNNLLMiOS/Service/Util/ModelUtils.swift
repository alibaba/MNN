//
//  ModelUtils.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/9/29.
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

    /// Check if the model is built in local model
    /// - Parameter model: ModelInfo
    /// - Returns: Whether is built in local model
    static func isBuiltInLocalModel(_ model: ModelInfo) -> Bool {
        guard let vendor = model.vendor, vendor.lowercased() == "local" else { return false }
        guard let sources = model.sources, let localSource = sources["local"] else { return false }
        return true
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

    static func isThinkMode(_: [Any]) -> Bool {
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

    /// Check if it's a Sana Diffusion model (style transfer)
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's a Sana Diffusion model
    static func isSanaDiffusionModel(_ modelName: String) -> Bool {
        let lowercased = modelName.lowercased()
        return lowercased.contains("sana") || lowercased.contains("ghibli")
    }

    /// Check if it's a Sana Diffusion model by checking the model directory structure
    /// - Parameter path: Path to the model directory
    /// - Returns: Whether the directory contains a Sana Diffusion model
    static func isSanaDiffusionModel(atPath path: String) -> Bool {
        let fm = FileManager.default
        let llmPath = (path as NSString).appendingPathComponent("llm")
        let connectorPath = (path as NSString).appendingPathComponent("connector.mnn")
        let vaeEncoderPath = (path as NSString).appendingPathComponent("vae_encoder.mnn")
        
        // Sana Diffusion model requires llm/ subdirectory, connector.mnn, and vae_encoder.mnn
        return fm.fileExists(atPath: llmPath) &&
               fm.fileExists(atPath: connectorPath) &&
               fm.fileExists(atPath: vaeEncoderPath)
    }

    /// Check if it's any type of diffusion model (Stable Diffusion or Sana)
    /// - Parameter modelName: Model name
    /// - Returns: Whether it's any diffusion model
    static func isAnyDiffusionModel(_ modelName: String) -> Bool {
        return isDiffusionModel(modelName) || isSanaDiffusionModel(modelName)
    }

    /// Check if audio output is supported
    /// - Parameter modelName: Model name
    /// - Returns: Whether audio output is supported
    static func supportAudioOutput(_ modelName: String) -> Bool {
        return isOmni(modelName)
    }
}
