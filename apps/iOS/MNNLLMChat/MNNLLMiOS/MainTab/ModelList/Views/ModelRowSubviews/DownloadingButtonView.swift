//
//  DownloadingButtonView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct DownloadingButtonView: View {
    @ObservedObject var viewModel: ModelListViewModel
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
