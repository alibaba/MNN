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
        displayString = ""
        startTime = Date().timeIntervalSince1970
        hasProcessed = false
    }

    func startNewChat() {
        displayString = ""
        hasProcessed = false
        startGeneration()
    }

    func startGeneration() {
        startTime = Date().timeIntervalSince1970
    }

    func getResult() -> String {
        return displayString
    }

    func process(progress: String?) -> String? {
        guard let progress = progress else { return nil }

        // Strip the opening marker regardless of how native streaming chunks it.
        // If a chunk ends with "<thi" and the next begins with "nk>", the
        // second cleanup below removes the reconstructed marker from displayString.
        var updatedProgress = progress.replacingOccurrences(of: thinkingPrefix, with: "")

        if updatedProgress.contains(completePrefix) {
            updatedProgress = updatedProgress.replacingOccurrences(of: completePrefix, with: "\n\n")
            hasProcessed = true
        }

        displayString.append(updatedProgress)
        displayString = displayString.replacingOccurrences(of: thinkingPrefix, with: "")

        // The closing marker can also straddle two native callbacks. Detect it
        // after appending so an unfinished Think response never becomes an HTML
        // element that hides the entire Markdown message.
        if displayString.contains(completePrefix) {
            displayString = displayString.replacingOccurrences(of: completePrefix, with: "\n\n")
            hasProcessed = true
        }
        return displayString
    }
}
