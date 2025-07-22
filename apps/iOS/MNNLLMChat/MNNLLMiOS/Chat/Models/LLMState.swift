//
//  LLMState.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import ExyteChat
import ExyteMediaPicker

actor LLMState {
    private var isProcessing: Bool = false
    
    func setProcessing(_ value: Bool) {
        isProcessing = value
    }
    
    func getProcessing() -> Bool {
        return isProcessing
    }
    
    func processContent(_ content: String, llm: LLMInferenceEngineWrapper?, showPerformance: Bool, completion: @escaping (String) -> Void) {
        llm?.processInput(content, withOutput: completion, showPerformance: true)
    }
}
