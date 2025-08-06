//
//  LocalModelRowView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/6/26.
//

import SwiftUI

struct LocalModelRowView: View {
    
    let model: ModelInfo

    private var localizedTags: [String] {
        model.localizedTags
    }
    
    private var formattedSize: String {
        model.formattedSize
    }
    
    var body: some View {
        HStack(alignment: .center) {
            
            ModelIconView(modelId: model.id)
                .frame(width: 40, height: 40)
            
            VStack(alignment: .leading, spacing: 8) {
                Text(model.modelName)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                
                if !localizedTags.isEmpty {
                    TagsView(tags: localizedTags)
                }
                
                HStack {
                    HStack(alignment: .center, spacing: 2) {
                        Image(systemName: "folder")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.gray)
                            .frame(width: 20, height: 20)
                        
                        Text(formattedSize)
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.gray)
                    }

                    Spacer()
                    
                    if let lastUsedAt = model.lastUsedAt {
                    Text("\(lastUsedAt.formatAgo())")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.gray)
                    }
                }
            }
            
            Spacer()
        }
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
