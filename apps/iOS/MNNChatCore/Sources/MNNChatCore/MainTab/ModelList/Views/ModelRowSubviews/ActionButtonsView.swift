//
//  ActionButtonsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct ActionButtonsView: View {
    let model: ModelInfo
    let state: ModelDownloadState
    let isOtherDownloading: Bool
    let onDownload: () -> Void
    let onPause: () -> Void
    let onCancel: () -> Void

    var body: some View {
        if state == .downloaded {
            // Deleting lives in the row's swipe actions.
            Label(AppText.text("button.downloaded"), systemImage: "checkmark.circle.fill")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            DownloadControls(state: state, isOtherDownloading: isOtherDownloading,
                             onDownload: onDownload, onPause: onPause, onCancel: onCancel)
        }
    }
}
