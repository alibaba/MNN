//
//  ThinkResultProcessor.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/11.
//

import Foundation

class ThinkResultProcessor {
    
    private let thinkingPrefix: String
    private var startTime: TimeInterval
    private var hasProcessed: Bool
    private let completePrefix: String
    
    var displayString: String
    
    init(thinkingPrefix: String, completePrefix: String) {
        self.thinkingPrefix = thinkingPrefix
        self.completePrefix = completePrefix
        self.displayString = "\(thinkingPrefix)\n> "
        self.startTime = Date().timeIntervalSince1970
        self.hasProcessed = false
    }
    
    func startNewChat() {
        displayString = ""
        hasProcessed = false
        self.startGeneration()
    }
    
    func startGeneration() {
        startTime = Date().timeIntervalSince1970
    }
    
    func getResult() -> String {
        return displayString
    }
    
    func process(progress: String?) -> String? {
        guard let progress = progress else { return nil }
        
        var updatedProgress = progress
        var rawBuilder = ""
        rawBuilder.append(progress)
        
        if progress.contains("</think>") {
            updatedProgress = updatedProgress.replacingOccurrences(of: "</think>", with: "\n")
            let thinkDuration = Int(Date().timeIntervalSince1970 - startTime)
            let prefixEndIndex = displayString.index(displayString.startIndex, offsetBy: thinkingPrefix.count, limitedBy: displayString.endIndex) ?? displayString.endIndex
            
            displayString.replaceSubrange(
                displayString.startIndex..<prefixEndIndex,
                with: completePrefix.replacingOccurrences(of: "ss", with: "\(thinkDuration)")
            )
            
            displayString = displayString.replacingOccurrences(of: "</think>", with: "")
            hasProcessed = true
        } else if !hasProcessed && progress.contains("\n") && !progress.contains("\n >") {
            updatedProgress = updatedProgress.replacingOccurrences(of: "\n", with: "\n > ")
        }
        
        displayString.append(updatedProgress)
        return displayString.replacingOccurrences(of: "<think>", with: "> ")
    }
}
