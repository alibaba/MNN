//
//  ModelRowView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct ModelRowView: View {
    
    let model: ModelInfo
    let downloadProgress: Double
    let isDownloading: Bool
    let isOtherDownloading: Bool
    let onDownload: () -> Void
    
    var body: some View {
        HStack(alignment: .top) {
            
            ModelIconView(modelId: model.modelId)
                .frame(width: 50, height: 50)
            
            VStack(alignment: .leading, spacing: 8) {
                Text(model.name)
                    .font(.headline)
                    .lineLimit(1)
                
                if !model.tags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(model.tags, id: \.self) { tag in
                                Text(tag)
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.blue.opacity(0.1))
                                    .cornerRadius(8)
                            }
                        }
                    }
                }
                
                if isDownloading {
                    ProgressView(value: downloadProgress) {
                        Text(String(format: "%.2f%%", downloadProgress * 100))
                            .font(.system(size: 14, weight: .regular, design: .default))
                    }
                } else {
                    Button(action: onDownload) {
                        Label(model.isDownloaded ? "Chat" : "Download",
                              systemImage: model.isDownloaded ? "message" : "arrow.down.circle")
                        .font(.system(size: 14, weight: .medium, design: .default))
                    }
                    .disabled(isOtherDownloading)
                }
            }
        }
    }
}
