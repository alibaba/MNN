//
//  DownloadedButtonView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

// MARK: - 已下载按钮视图
struct DownloadedButtonView: View {
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