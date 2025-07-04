//
//  TBModelRowView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

struct TBModelRowView: View {
    
    let model: TBModelInfo
    @ObservedObject var viewModel: TBModelListViewModel
    
    let downloadProgress: Double
    let isDownloading: Bool
    let isOtherDownloading: Bool
    let onDownload: () -> Void
    
    @State private var showDeleteAlert = false
    
    // 预计算本地化标签，避免重复计算
    private var localizedTags: [String] {
        model.localizedTags
    }
    
    // 预计算格式化大小，避免重复计算
    private var formattedSize: String {
        model.formattedSize
    }
    
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            // 模型图标
            ModelIconView(modelId: model.id)
                .frame(width: 40, height: 40)
            
            // 主要信息区域
            VStack(alignment: .leading, spacing: 6) {
                // 模型名称
                Text(model.modelName)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                
                // 标签列表
                if !localizedTags.isEmpty {
                    TagsView(tags: localizedTags)
                }
            }
            .padding(.leading, 8)
            
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
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle()) // 确保整个区域都可以点击
        .onTapGesture {
            handleRowTap()
        }
        .alert("确认删除", isPresented: $showDeleteAlert) {
            Button("删除", role: .destructive) {
                Task {
                    await viewModel.deleteModel(model)
                }
            }
            Button("取消", role: .cancel) { }
        } message: {
            Text("是否确认删除该模型？")
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
