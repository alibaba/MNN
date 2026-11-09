//
//  LanguageModelTypes.swift
//
//
//  Created by Pedro Cuenca on 8/5/23.
//

import CoreML
import Generation
import Tokenizers

public protocol LanguageModelProtocol {
    /// `name_or_path` in the Python world
    var modelName: String { get }

    var tokenizer: Tokenizer { get async throws }
    var model: MLModel { get }

    init(model: MLModel)

    /// Make prediction callable (this works like __call__ in Python)
    func predictNextTokenScores(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol
    func callAsFunction(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol
}

public extension LanguageModelProtocol {
    func callAsFunction(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol {
        predictNextTokenScores(tokens, config: config)
    }
}

public protocol TextGenerationModel: Generation, LanguageModelProtocol {
    var defaultGenerationConfig: GenerationConfig { get }
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback?) async throws -> String
}

public extension TextGenerationModel {
    @discardableResult
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback? = nil) async throws -> String {
        try await generate(config: config, prompt: prompt, model: callAsFunction, tokenizer: tokenizer, callback: callback)
    }
}
