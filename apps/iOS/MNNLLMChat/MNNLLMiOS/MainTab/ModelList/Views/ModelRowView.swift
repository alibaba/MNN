//
//  ModelRowView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct ModelRowView: View {
    
    let model: ModelInfo
    @ObservedObject var viewModel: ModelListViewModel
    
    let downloadProgress: Double
    let isDownloading: Bool
    let isOtherDownloading: Bool
    let onDownload: () -> Void
    
    
    @State private var showDeleteAlert = false
    
    var body: some View {
        HStack(alignment: .top) {
            ModelIconView(modelId: model.modelId)
                .frame(width: 40, height: 40)
            
            VStack(alignment: .leading, spacing: 5) {
                Text(model.name)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                
                if let lastUsedAt = model.lastUsedAt {
                    Text("Last used: \(lastUsedAt.formatAgo())")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
                
                if !model.tags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(model.tags, id: \.self) { tag in
                                Text(tag)
                                    .fontWeight(.regular)
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(Color.gray.opacity(0.5), lineWidth: 0.5)
                                    )
                            }
                        }
                    }
                    .frame(height: 25)
                }
            }
            
            Spacer()
            
            VStack(alignment: .center, spacing: 4) {
                if model.isDownloaded {
                    Button(action: {
                        showDeleteAlert = true
                    }) {
                        Image(systemName: "trash")
                            .fontWeight(.regular)
                            .foregroundColor(.black.opacity(0.8))
                            .frame(width: 20, height: 20)
                        
                        Text("已下载")
                            .font(.caption2)
                            .foregroundColor(.gray)
                            .padding(.top, 4)
                    }
                } else {
                    if isDownloading {
                        Button(action: {
                            Task {
                                await viewModel.cancelDownload()
                            }
                        }) {
                            ProgressView(value: downloadProgress)
                                .progressViewStyle(CircularProgressViewStyle())
                                .frame(width: 28, height: 28)
                            Text(String(format: "%.2f%%", downloadProgress * 100))
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                    } else {
                        Button(action: onDownload) {
                            Image(systemName: "arrow.down.circle.fill")
                                .font(.title2)
                            }
                            .foregroundColor(isOtherDownloading ? .gray : .primaryPurple)
                            .disabled(isOtherDownloading)
                        
                        HStack(alignment: .bottom, spacing: 2) {
                            Image(systemName: "folder")
                                .font(.caption2)
                            Text(model.formattedSize)
                                .font(.caption2)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                                .offset(y: 1)
                                .onAppear {
                                    if !model.isDownloaded && model.cachedSize == nil {
                                        Task {
                                            if let size = await model.fetchRemoteSize() {
                                                await MainActor.run {
                                                    if let index = viewModel.models.firstIndex(where: { $0.modelId == model.modelId }) {
                                                        viewModel.models[index].cachedSize = size
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                        }
                        .foregroundColor(.gray)
                    }
                }
            }
            
            .frame(width: 60)
        }
        .padding(.vertical, 8)
        .alert(isPresented: $showDeleteAlert) {
            Alert(
                title: Text("确认删除"),
                message: Text("是否确认删除该模型？"),
                primaryButton: .destructive(Text("删除")) {
                    Task {
                        await viewModel.deleteModel(model)
                    }
                },
                secondaryButton: .cancel(Text("取消"))
            )
        }
    }
}
