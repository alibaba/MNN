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
        HStack(spacing: 12) {
            
            ModelIconView(modelId: history.modelId)
                .frame(width: 36, height: 36)
                .clipShape(Circle())
            
            VStack(alignment: .leading, spacing: 4) {
                
                if let firstMessage = history.messages.last {
                    Text(String(firstMessage.content.prefix(50)) + "...")
                        .lineLimit(2)
                        .font(.system(size: 14))
                }
                
                HStack {
                    VStack(alignment: .leading) {
                        Text(history.modelName)
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                        
                        Text(history.updatedAt.formatAgo())
                            .font(.system(size: 10))
                            .foregroundColor(.gray)
                    }
                }
            }
        }
        .padding(.vertical, 8)
    }
}
