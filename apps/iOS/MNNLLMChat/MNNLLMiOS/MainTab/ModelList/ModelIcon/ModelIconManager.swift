//
//  ModelIconManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/13.
//

import Foundation

class ModelIconManager {
    
    static let shared = ModelIconManager()
    
    private init() {}
    
    func getModelImage(with modelName: String?) -> String? {
        guard let modelName = modelName?.lowercased() else {
            return nil
        }
        
        if modelName.contains("qwen") || modelName.contains("qwq") {
            return ModelIcon.qwen.imageName
        } else if modelName.contains("llama") || modelName.contains("mobilellm") {
            return ModelIcon.llama.imageName
        } else if modelName.contains("smo") {
            return ModelIcon.smolm.imageName
        } else if modelName.contains("phi") {
            return ModelIcon.phi.imageName
        } else if modelName.contains("baichuan") {
            return ModelIcon.baichuan.imageName
        } else if modelName.contains("yi") {
            return ModelIcon.yi.imageName
        } else if modelName.contains("glm") || modelName.contains("codegeex") {
            return ModelIcon.chatglm.imageName
        } else if modelName.contains("reader") {
            return ModelIcon.jina.imageName
        } else if modelName.contains("deepseek") {
            return ModelIcon.deepseek.imageName
        } else if modelName.contains("internlm") {
            return ModelIcon.internlm.imageName
        } else if modelName.contains("gemma") {
            return ModelIcon.gemma.imageName
        }
        
        return ModelIcon.defaultMNN.imageName
    }
}
