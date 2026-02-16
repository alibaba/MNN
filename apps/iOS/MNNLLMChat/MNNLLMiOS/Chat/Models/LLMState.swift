//
//  LLMState.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import ExyteChat
import UIKit

actor LLMState {
    private var isProcessing: Bool = false

    func setProcessing(_ value: Bool) {
        isProcessing = value
    }

    func getProcessing() -> Bool {
        return isProcessing
    }

    func processContent(_ content: String, llm: LLMInferenceEngineWrapper?, showPerformance _: Bool, completion: @escaping (String) -> Void) {
        llm?.processInput(content, withOutput: completion, showPerformance: true)
    }

    /// Processes multimodal (text + images) content using the MultimodalPrompt API.
    /// - Parameters:
    ///   - content: Prompt template containing <img>placeholder</img> tags.
    ///   - images: Dictionary mapping placeholder keys to UIImage instances.
    ///   - llm: Inference engine wrapper.
    ///   - showPerformance: Whether to output performance statistics.
    ///   - completion: Streaming output callback.
    func processMultimodalContent(
        _ content: String,
        images: [String: UIImage],
        llm: LLMInferenceEngineWrapper?,
        showPerformance: Bool,
        completion: @escaping (String) -> Void
    ) {
        llm?.processMultimodalInput(
            content,
            images: images,
            withOutput: completion,
            showPerformance: showPerformance
        )
    }

    /// Processes a batch of prompts and returns their responses.
    ///
    /// This method delegates batch processing to the underlying inference engine wrapper.
    /// Responses are returned in the same order as the input prompts.
    /// - Parameters:
    ///   - prompts: An array of prompt strings to process.
    ///   - llm: The inference engine wrapper instance to use.
    ///   - completion: A closure invoked with an array of response strings.
    func processBatchTestContent(_ prompts: [String], llm: LLMInferenceEngineWrapper?, completion: @escaping ([String]) -> Void) {
        guard let llm = llm else {
            completion([])
            return
        }
        llm.processBatchPrompts(prompts) { responses in
            completion(responses)
        }
    }
}
