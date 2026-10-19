import ExyteChat
import Foundation

/// Extract earlier requests, including requests that never got a final answer.
/// Thinking text, performance data and system status bubbles are display-only.
enum ConversationContextBuilder {
    struct Turn: Equatable {
        let user: String
        let assistant: String
        var prompts: [[String: String]] {
            [["user": user]] + (assistant.isEmpty ? [] : [["assistant": assistant]])
        }
        var transcript: String {
            "User: \(user)\n" + (assistant.isEmpty ? "[No final assistant response]\n" : "Assistant: \(assistant)\n")
        }
    }

    static func prompts(from messages: [Message]) -> [[String: String]] {
        turns(from: messages).flatMap(\.prompts)
    }

    static func turns(from messages: [Message]) -> [Turn] {
        var turns: [Turn] = []
        var pendingUser: String?
        for message in messages {
            switch message.user.id {
            case "1":
                if let user = pendingUser {
                    // A stopped or token-limited thinking-only response still
                    // contains a user request that the next turn must retain.
                    turns.append(Turn(user: user, assistant: ""))
                }
                var text = message.text
                if !message.attachments.isEmpty { text += "\n[media]" }
                if message.recording != nil { text += "\n[audio]" }
                pendingUser = text
            case "2":
                guard let user = pendingUser,
                      !message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                turns.append(Turn(user: user, assistant: message.text))
                pendingUser = nil
            default:
                continue
            }
        }

        // The pending request at the end is the current prompt. The caller
        // appends it separately; do not duplicate it while it is unanswered.
        return turns
    }
}
