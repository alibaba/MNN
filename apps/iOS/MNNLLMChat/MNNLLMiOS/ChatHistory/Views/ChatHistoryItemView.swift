//
//  ChatHistoryItemView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/16.
//

import SwiftUI

struct ChatHistoryItemView: View {
    let history: ChatHistory
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            
            if let firstMessage = history.messages.last {
                Text(String(firstMessage.content.prefix(200)))
                    .lineLimit(1)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(.primary)
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
}
