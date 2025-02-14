//
//  ModelIcon.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/13.
//


import Foundation

enum ModelIcon: String {
    case qwen = "qwen_icon"
    case llama = "llama_icon"
    case smolm = "smolm_icon"
    case phi = "phi_icon"
    case baichuan = "baichuan_icon"
    case yi = "yi_icon"
    case chatglm = "chatglm_icon"
    case jina = "jina_icon"
    case deepseek = "deepseek_icon"
    case internlm = "internlm_icon"
    case gemma = "gemma_icon"
    case defaultMNN = "mnn_icon"
    
    var imageName: String {
        return self.rawValue
    }
}
