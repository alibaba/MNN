//
//  PendingDownloadButtonView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

// MARK: - 待下载按钮视图
struct PendingDownloadButtonView: View {
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