//
//  StreamingMessageState.swift
//  MNNLLMChat
//
//  Created by 游薪渝(揽清) on 2025/9/10.
//

import Foundation

/// Streaming message state enumeration
///
/// Used to accurately track different stages of streaming messages, solving timing issues between model output completion and UI animation completion
public enum StreamingMessageState: Equatable {
    /// Non-streaming message
    case none

    /// Currently streaming (model is generating content)
    case streaming

    /// Model output complete, but UI animation still in progress
    case outputCompleteAnimating

    /// Fully complete (both model output and UI animation are finished)
    case completed

    /// Whether should display as streaming state (for UI judgment)
    public var isStreaming: Bool {
        switch self {
        case .streaming, .outputCompleteAnimating:
            return true
        case .none, .completed:
            return false
        }
    }

    /// Whether model output is complete
    public var isOutputComplete: Bool {
        switch self {
        case .outputCompleteAnimating, .completed:
            return true
        case .none, .streaming:
            return false
        }
    }

    /// Whether fully complete
    public var isFullyComplete: Bool {
        return self == .completed
    }
}

/// Streaming message state manager
///
/// Responsible for managing streaming state transitions of individual messages
public class StreamingMessageStateManager: ObservableObject {
    @Published public private(set) var state: StreamingMessageState = .none

    private let messageId: String

    public init(messageId: String = "") {
        self.messageId = messageId
    }

    /// Start streaming output
    public func startStreaming() {
        state = .streaming
        postStateChangeNotification()
    }

    /// Model output complete (but animation may still be in progress)
    public func markOutputComplete() {
        if state == .streaming {
            state = .outputCompleteAnimating
            postStateChangeNotification()
        }
    }

    /// Animation complete
    public func markAnimationComplete() {
        switch state {
        case .streaming:
            // If animation completes while model is still outputting, keep streaming state
            break
        case .outputCompleteAnimating:
            // Both model output and animation are complete
            state = .completed
            postStateChangeNotification()
        case .none, .completed:
            break
        }
    }

    /// Reset state
    public func reset() {
        state = .none
        postStateChangeNotification()
    }

    /// Force complete (for cleanup or error handling)
    public func forceComplete() {
        state = .completed
        postStateChangeNotification()
    }

    /// Send state change notification
    private func postStateChangeNotification() {
        NotificationCenter.default.post(
            name: NSNotification.Name("StreamingStateChanged"),
            object: nil,
            userInfo: ["messageId": messageId, "state": state]
        )
    }
}
