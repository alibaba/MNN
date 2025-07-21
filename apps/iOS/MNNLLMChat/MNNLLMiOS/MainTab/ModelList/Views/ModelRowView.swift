//
//  ModelRowView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
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
    
    private var localizedTags: [String] {
        model.localizedTags
    }
    
    private var formattedSize: String {
        model.formattedSize
    }
    
    var body: some View {
        HStack(alignment: .center, spacing: 0) {
            
            ModelIconView(modelId: model.id)
                .frame(width: 40, height: 40)
            
            VStack(alignment: .leading, spacing: 6) {
                
                Text(model.modelName)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                
                if !localizedTags.isEmpty {
                    TagsView(tags: localizedTags)
                }

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
            }
            .padding(.leading, 8)
            
            Spacer()
            
            VStack {
                Spacer()
                ActionButtonsView(
                    model: model,
                    viewModel: viewModel,
                    downloadProgress: downloadProgress,
                    isDownloading: isDownloading,
                    isOtherDownloading: isOtherDownloading,
                    formattedSize: formattedSize,
                    onDownload: onDownload,
                    showDeleteAlert: $showDeleteAlert
                )
                Spacer()
            }
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .onTapGesture {
            handleRowTap()
        }
        .alert(LocalizedStringKey("alert.deleteModel.title"), isPresented: $showDeleteAlert) {
            Button("Delete", role: .destructive) {
                Task {
                    await viewModel.deleteModel(model)
                }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text(LocalizedStringKey("alert.deleteModel.message"))
        }
    }
    
    private func handleRowTap() {
        if model.isDownloaded {
            return
        } else if isDownloading {
            Task {
                await viewModel.cancelDownload()
            }
        } else if !isOtherDownloading {
            onDownload()
        }
    }
}
