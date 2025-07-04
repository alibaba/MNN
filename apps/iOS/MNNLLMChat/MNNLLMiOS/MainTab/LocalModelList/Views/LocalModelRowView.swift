//
//  LocalModelRowView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/6/26.
//

import SwiftUI

struct LocalModelRowView: View {
    
    let model: ModelInfo

    var body: some View {
        HStack(alignment: .center) {
            
            ModelIconView(modelId: model.modelId)
                .frame(width: 50, height: 50)
            
            VStack(alignment: .leading, spacing: 8) {
                Text(model.name)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                
                if !model.tags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(model.tags, id: \.self) { tag in
                                Text(tag)
                                    .fontWeight(.regular)
                                    .font(.caption)
                                    .foregroundColor(Color(red: 151/255, green: 151/255, blue: 151/255))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(
                                        RoundedRectangle(cornerRadius: 10)
                                            .stroke(Color(red: 151/255, green: 151/255, blue: 151/255), lineWidth: 0.5)
                                            .padding(1)
                                    )
                            }
                        }
                    }
                }
                
                HStack {
                    HStack(alignment: .center, spacing: 2) {
                        Image(systemName: "folder")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.gray)
                            .frame(width: 20, height: 20)
                        
                        Text(model.formattedSize)
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
        }
    }
}
