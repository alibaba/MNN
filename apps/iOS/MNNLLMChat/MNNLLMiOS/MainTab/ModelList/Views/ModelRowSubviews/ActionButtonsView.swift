//
//  ActionButtonsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct ActionButtonsView: View {
    let model: ModelInfo
    @ObservedObject var viewModel: ModelListViewModel
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
