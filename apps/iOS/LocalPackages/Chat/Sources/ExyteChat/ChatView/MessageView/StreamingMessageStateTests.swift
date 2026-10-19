//
//  StreamingMessageStateTests.swift
//  MNNLLMChat
//
//  Created by 游薪渝(揽清) on 2025/9/10.
//

import Foundation

class StreamingMessageStateTests {
    /// Test normal streaming output flow
    static func testNormalStreamingFlow() {
        let stateManager = StreamingMessageStateManager()

        // Initial state
        assert(stateManager.state == .none, "Initial state should be .none")
        assert(!stateManager.state.isStreaming, "Should not be streaming initially")

        // Start streaming output
        stateManager.startStreaming()
        assert(stateManager.state == .streaming, "State should be .streaming")
        assert(stateManager.state.isStreaming, "Should be streaming")
        assert(!stateManager.state.isOutputComplete, "Output should not be complete yet")

        // Model output complete
        stateManager.markOutputComplete()
        assert(stateManager.state == .outputCompleteAnimating, "State should be .outputCompleteAnimating")
        assert(stateManager.state.isStreaming, "UI should still show as streaming")
        assert(stateManager.state.isOutputComplete, "Output should be complete")

        // Animation complete
        stateManager.markAnimationComplete()
        assert(stateManager.state == .completed, "State should be .completed")
        assert(!stateManager.state.isStreaming, "Should not show as streaming anymore")
        assert(stateManager.state.isFullyComplete, "Should be fully complete")

        print("✅ Normal streaming flow test passed")
    }

    /// Test scenario where animation completes before model output
    static func testAnimationCompleteBeforeOutput() {
        let stateManager = StreamingMessageStateManager()

        // Start streaming output
        stateManager.startStreaming()
        assert(stateManager.state == .streaming, "State should be .streaming")

        // Animation completes first (model still outputting)
        stateManager.markAnimationComplete()
        assert(stateManager.state == .streaming, "Should maintain streaming state")
        assert(stateManager.state.isStreaming, "Should still show as streaming")

        // Model output complete
        stateManager.markOutputComplete()
        assert(stateManager.state == .outputCompleteAnimating, "State should be .outputCompleteAnimating")

        // Mark animation complete again
        stateManager.markAnimationComplete()
        assert(stateManager.state == .completed, "State should be .completed")

        print("✅ Animation complete before output test passed")
    }

    /// Test force complete functionality
    static func testForceComplete() {
        let stateManager = StreamingMessageStateManager()

        // Start streaming output
        stateManager.startStreaming()
        assert(stateManager.state == .streaming, "Should be streaming state")

        // Force complete
        stateManager.forceComplete()
        assert(stateManager.state == .completed, "Should be completed state")
        assert(stateManager.state.isFullyComplete, "Should be fully complete")

        print("✅ Force complete functionality test passed")
    }

    /// Test reset functionality
    static func testReset() {
        let stateManager = StreamingMessageStateManager()

        // Set to some state
        stateManager.startStreaming()
        stateManager.markOutputComplete()
        assert(stateManager.state == .outputCompleteAnimating, "Should be outputCompleteAnimating state")

        // Reset
        stateManager.reset()
        assert(stateManager.state == .none, "Should reset to none state")
        assert(!stateManager.state.isStreaming, "Should not be streaming state")

        print("✅ Reset functionality test passed")
    }

    /// Run all tests
    static func runAllTests() {
        print("🧪 Starting streaming message state management tests...")

        testNormalStreamingFlow()
        testAnimationCompleteBeforeOutput()
        testForceComplete()
        testReset()

        print("🎉 All tests passed! Streaming message state management system is working properly.")
    }
}

// MARK: - Usage Examples

/// Usage example: Demonstrates how to use the new state management system in real scenarios
class StreamingMessageStateUsageExample {
    /// Simulate usage in LLMChatViewModel
    static func simulateViewModelUsage() {
        print("\n📝 Simulating ViewModel usage scenario...")

        var streamingStates: [String: StreamingMessageStateManager] = [:]
        let messageId = "test-message-123"

        // 1. Start streaming output
        let stateManager = StreamingMessageStateManager()
        streamingStates[messageId] = stateManager
        stateManager.startStreaming()
        print("Started streaming output, state: \(stateManager.state)")

        // 2. Check if should display as streaming
        let isStreaming = stateManager.state.isStreaming
        print("UI should display as streaming: \(isStreaming)")

        // 3. Model output complete
        stateManager.markOutputComplete()
        print("Model output complete, state: \(stateManager.state)")
        print("UI should still display as streaming: \(stateManager.state.isStreaming)")

        // 4. Animation complete
        stateManager.markAnimationComplete()
        print("Animation complete, state: \(stateManager.state)")

        // 5. Clean up state
        if stateManager.state.isFullyComplete {
            streamingStates.removeValue(forKey: messageId)
            print("State has been cleaned up")
        }

        print("✅ ViewModel usage scenario simulation complete")
    }
}
