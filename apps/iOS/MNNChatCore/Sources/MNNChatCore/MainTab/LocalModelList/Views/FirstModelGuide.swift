import SwiftUI

struct FirstModelGuide: View {
    @ObservedObject var viewModel: ModelListViewModel
    let browseModels: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 48))
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
            Text("onboarding.title").font(.title2.bold())
            Text("onboarding.description")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            RecommendedModelCard(viewModel: viewModel)
            Button("onboarding.browse", action: browseModels)
                .buttonStyle(.bordered)
            Text("onboarding.networkNote").font(.footnote).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 28)
    }
}

/// One-tap download of the suggested first model, with progress, pause,
/// resume, retry and cancel in place. Selects the model once it is ready.
struct RecommendedModelCard: View {
    @ObservedObject var viewModel: ModelListViewModel
    @State private var confirmsLargeDownload = false

    // This model was exercised on the target device. If the catalog doesn't
    // offer it, show nothing instead of inventing a download destination.
    static let modelName = "Qwen3.5-0.8B-MNN"

    private var model: ModelInfo? {
        viewModel.models.first { $0.modelName == Self.modelName }
    }

    var body: some View {
        if let model {
            card(for: model)
        } else if viewModel.isLoadingModels {
            ProgressView(AppText.text("onboarding.loading"))
        }
    }

    private func card(for model: ModelInfo) -> some View {
        let state = viewModel.downloadState(for: model)
        let totalBytes = viewModel.totalBytes(for: model)
        let isOtherDownloading = viewModel.currentlyDownloading != nil && viewModel.currentlyDownloading != model.id
        return VStack(alignment: .leading, spacing: 12) {
            Text(AppText.text("onboarding.recommended")).font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                ModelIconView(modelId: model.id)
                    .frame(width: 40, height: 40)
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name).font(.headline).lineLimit(1)
                    HStack(spacing: 6) {
                        if let totalBytes {
                            Text(ByteText.file(totalBytes)).font(.caption).foregroundStyle(.secondary)
                        }
                        if let fit = model.deviceFit {
                            DeviceFitBadge(fit: fit)
                        }
                    }
                }
                Spacer(minLength: 0)
            }
            switch state {
            case .idle:
                Button { requestDownload(model) } label: {
                    Text(AppText.text("onboarding.download")).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isOtherDownloading)
            case .downloading, .paused, .failed:
                HStack(alignment: .center, spacing: 12) {
                    DownloadStatusLine(state: state, totalBytes: totalBytes, speed: viewModel.downloadSpeed)
                    DownloadControls(state: state, isOtherDownloading: isOtherDownloading,
                                     onDownload: { download(model) },
                                     onPause: { Task { await viewModel.pauseDownload() } },
                                     onCancel: { Task { await viewModel.discardDownload(model) } })
                }
            case .downloaded:
                EmptyView()
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
        .tooLargeModelConfirmation(isPresented: $confirmsLargeDownload, model: model) { download(model) }
    }

    private func requestDownload(_ model: ModelInfo) {
        if model.deviceFit == .tooLarge {
            confirmsLargeDownload = true
        } else {
            download(model)
        }
    }

    private func download(_ model: ModelInfo) {
        Task {
            await viewModel.downloadModel(model)
            if let downloaded = viewModel.models.first(where: {
                $0.modelName == model.modelName && $0.isDownloaded
            }) {
                viewModel.selectModel(downloaded)
            }
        }
    }
}
