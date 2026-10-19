//
//  StreamingMessageProvider.swift
//  Chat
//
//  Created by 游薪渝(揽清) on 2025/9/11.
//

import Foundation

/// Streaming message state provider protocol
///
/// This protocol defines the interface for accessing streaming message state, allowing the Chat library
/// to access streaming state management functionality without directly depending on specific ViewModel types.
/// This solves the problem of the Chat library as an independent package being unable to access the main project's LLMChatViewModel.
///
/// Usage scenarios:
/// - MessageView needs to determine if a message is in streaming state
/// - Integration with the new StreamingMessageState system
/// - Maintaining the independence and reusability of the Chat library
public protocol StreamingMessageProvider {
    /// Checks if the specified message is in streaming state
    /// - Parameter messageId: The message ID
    /// - Returns: `true` if the message is currently streaming, `false` otherwise
    func isMessageStreaming(_ messageId: String) -> Bool

    /// Gets the streaming state of the specified message (optional method for more detailed state information)
    /// - Parameter messageId: The message ID
    /// - Returns: The streaming state of the message, or `nil` if the message doesn't exist
    func getStreamingState(_ messageId: String) -> StreamingMessageState?
}

/// Default implementation of StreamingMessageProvider protocol
///
/// Provides default behavior for ViewModels that do not support streaming state
public extension StreamingMessageProvider {
    func getStreamingState(_ messageId: String) -> StreamingMessageState? {
        return isMessageStreaming(messageId) ? .streaming : .none
    }
}

/// Extension of ChatViewModel providing default streaming state implementation
///
/// For base ChatViewModel that does not support streaming functionality, provides default non-streaming state return
extension ChatViewModel: StreamingMessageProvider {
    public func isMessageStreaming(_: String) -> Bool {
        // Base ChatViewModel doesn't support streaming state, always returns false
        return false
    }

    public func getStreamingState(_: String) -> StreamingMessageState? {
        // Base ChatViewModel doesn't support streaming state, always returns .none
        return .none
    }
}
