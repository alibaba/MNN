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

// MARK: - 子视图组件

private struct TagsView: View {
    let tags: [String]
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(tags, id: \.self) { tag in
                    TagChip(text: tag)
                }
            }
            .padding(.horizontal, 1)
        }
        .frame(height: 25)
    }
}

private struct TagChip: View {
    let text: String
    
    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
            )
    }
}

private struct ActionButtonsView: View {
    let model: TBModelInfo
    @ObservedObject var viewModel: TBModelListViewModel
    let downloadProgress: Double
    let isDownloading: Bool
    let isOtherDownloading: Bool
    let formattedSize: String
    let onDownload: () -> Void
    @Binding var showDeleteAlert: Bool
    
    var body: some View {
        VStack(alignment: .center, spacing: 4) {
            if model.isDownloaded {
                // 已下载状态
                DownloadedButtonView(showDeleteAlert: $showDeleteAlert)
            } else if isDownloading {
                // 下载中状态
                DownloadingButtonView(
                    viewModel: viewModel,
                    downloadProgress: downloadProgress
                )
            } else {
                // 待下载状态
                PendingDownloadButtonView(
                    isOtherDownloading: isOtherDownloading,
                    formattedSize: formattedSize,
                    onDownload: onDownload
                )
            }
        }
        .frame(width: 60)
    }
}

private struct DownloadedButtonView: View {
    @Binding var showDeleteAlert: Bool
    
    var body: some View {
        Button(action: { showDeleteAlert = true }) {
            VStack(spacing: 2) {
                Image(systemName: "trash")
                    .font(.system(size: 16))
                    .foregroundColor(.primary.opacity(0.8))
                
                Text("已下载")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
}

private struct DownloadingButtonView: View {
    @ObservedObject var viewModel: TBModelListViewModel
    let downloadProgress: Double
    
    var body: some View {
        Button(action: {
            Task {
                await viewModel.cancelDownload()
            }
        }) {
            VStack(spacing: 2) {
                ProgressView(value: downloadProgress)
                    .progressViewStyle(CircularProgressViewStyle(tint: .accentColor))
                    .frame(width: 24, height: 24)
                
                Text(String(format: "%.2f%%", downloadProgress * 100))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
}

private struct PendingDownloadButtonView: View {
    let isOtherDownloading: Bool
    let formattedSize: String
    let onDownload: () -> Void
    
    var body: some View {
        Button(action: onDownload) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.title2)
                .foregroundColor(isOtherDownloading ? .secondary : .primaryPurple)
        }
        .disabled(isOtherDownloading)
        
        HStack(alignment: .center, spacing: 2) {
            Image(systemName: "folder")
                .font(.caption2)
            
            Text(formattedSize)
                .font(.caption2)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .foregroundColor(.secondary)
    }
}
