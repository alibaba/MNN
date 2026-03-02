//
//  PendingDownloadButtonView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

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
    }
}
