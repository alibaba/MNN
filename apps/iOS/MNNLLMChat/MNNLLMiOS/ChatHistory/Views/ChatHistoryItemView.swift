//
//  ChatHistoryItemView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/16.
//

import ExyteChat
import MarkdownView
import SwiftUI

struct ChatHistoryItemView: View {
    let history: ChatHistory

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let previewMessage = getPreviewMessage() {
                Text(previewMessage.content.prefix(100))
                    .font(.system(size: 15, weight: .medium))
                    .frame(maxHeight: 60)
                    .lineLimit(3)
            }

            HStack(alignment: .bottom) {
                ModelIconView(modelId: history.modelId)
                    .frame(width: 20, height: 20)
                    .clipShape(Circle())
                    .padding(.trailing, 0)

                Text(history.modelName)
                    .lineLimit(1)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.black.opacity(0.5))

                Spacer()

                Text(history.updatedAt.formatAgo())
                    .font(.system(size: 12, weight: .regular))
                    .foregroundColor(.black.opacity(0.5))
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 0)
    }

    /// Gets the preview message from the chat history
    /// Priority: first user message -> first model message -> nil
    /// - Returns: The preview message, or nil if no messages exist
    private func getPreviewMessage() -> HistoryMessage? {
        // First try to find the first user message
        for message in history.messages {
            if message.isUser, !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message
            }
        }

        // If no user message found, try to find the first model message
        for message in history.messages {
            if !message.isUser, !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message
            }
        }

        // If no messages found, return nil
        return nil
    }
}
