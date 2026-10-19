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
    @ObservedObject private var nameStore = ModelNameStore.shared

    let isOtherDownloading: Bool
    /// Loaded by the chat right now, so it cannot be deleted.
    var isInUse = false

    @State private var deletionRequest: ModelDeletionRequest?
    @State private var confirmsLargeDownload = false

    private var state: ModelDownloadState {
        viewModel.downloadState(for: model)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 0) {
            ModelIconView(modelId: model.id)
                .frame(width: 40, height: 40)

            VStack(alignment: .leading, spacing: 6) {
                Text(model.name)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(1)

                if !model.localizedTags.isEmpty {
                    TagsView(tags: model.localizedTags)
                }

                switch state {
                case .downloading, .paused, .failed:
                    DownloadStatusLine(state: state, totalBytes: viewModel.totalBytes(for: model),
                                       speed: viewModel.downloadSpeed)
                case .idle, .downloaded:
                    HStack(alignment: .center, spacing: 6) {
                        HStack(alignment: .center, spacing: 2) {
                            Image(systemName: "folder")
                                .font(.caption)
                                .fontWeight(.medium)
                                .frame(width: 20, height: 20)
                            Text(model.formattedSize)
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.gray)
                        if state == .idle, let fit = model.deviceFit {
                            DeviceFitBadge(fit: fit)
                        }
                    }
                }
            }
            .padding(.leading, 8)

            Spacer(minLength: 8)

            ActionButtonsView(
                model: model,
                state: state,
                isOtherDownloading: isOtherDownloading,
                onDownload: requestDownload,
                onPause: { Task { await viewModel.pauseDownload() } },
                onCancel: { Task { await viewModel.discardDownload(model) } }
            )
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .onTapGesture {
            handleRowTap()
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if state == .downloaded && !ModelUtils.isBuiltInLocalModel(model) {
                ModelDeleteSwipeButton { deletionRequest = .make(for: model, isInUse: isInUse) }
            }
        }
        .modelDeletionAlert($deletionRequest) { model in
            Task { await viewModel.deleteModel(model) }
        }
        .tooLargeModelConfirmation(isPresented: $confirmsLargeDownload, model: model, onConfirm: startDownload)
    }

    /// Tapping a row starts, resumes or retries; it never stops a running
    /// download, which only the explicit pause/cancel buttons do.
    private func handleRowTap() {
        switch state {
        case .idle, .paused, .failed:
            if !isOtherDownloading { requestDownload() }
        case .downloading, .downloaded:
            break
        }
    }

    private func requestDownload() {
        if state == .idle, model.deviceFit == .tooLarge {
            confirmsLargeDownload = true
        } else {
            startDownload()
        }
    }

    private func startDownload() {
        Task { await viewModel.downloadModel(model) }
    }
}
