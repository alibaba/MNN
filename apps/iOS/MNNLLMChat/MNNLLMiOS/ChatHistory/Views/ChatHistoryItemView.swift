//
//  ChatHistoryItemView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/16.
//

import SwiftUI
import MarkdownUI

struct ChatHistoryItemView: View {
    let history: ChatHistory
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            
            if let lastMessage = getLastNonEmptyMessage() {
                
                Markdown(String(lastMessage.content.prefix(100)))
                    .markdownTextStyle {
                       FontSize(15)
                       FontWeight(.regular)
                    }
                    .markdownBlockStyle(\.blockquote) { configuration in
                      configuration.label
                            .lineLimit(3)
                            .font(.system(size: 15, weight: .medium))
                            .foregroundColor(.primary)
                        .markdownTextStyle {
                            BackgroundColor(nil)
                            FontSize(15)
                        }
                    }
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
