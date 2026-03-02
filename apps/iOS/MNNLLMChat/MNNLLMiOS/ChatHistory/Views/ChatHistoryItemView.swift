//
//  ChatHistoryItemView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/16.
//

import SwiftUI
import ExyteChat
import MarkdownView

struct ChatHistoryItemView: View {
    let history: ChatHistory
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            
            if let lastMessage = getLastNonEmptyMessage() {
                MarkdownTextViewWrapper(text: String(lastMessage.content.prefix(100)), bindScrollView: true)
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
    
    private func getLastNonEmptyMessage() -> HistoryMessage? {
        for message in history.messages.reversed() {
            if !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message
            }
        }
        return nil
    }
}
