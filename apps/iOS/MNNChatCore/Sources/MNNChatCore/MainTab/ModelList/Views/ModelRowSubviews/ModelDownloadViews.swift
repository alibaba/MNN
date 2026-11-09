import SwiftUI

struct DeviceFitBadge: View {
    let fit: ModelDeviceFit

    var body: some View {
        Label(fit.title, systemImage: fit.systemImage)
            .font(.caption2.weight(.medium))
            .lineLimit(1)
            .foregroundStyle(fit.color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(fit.color.opacity(0.12), in: Capsule())
    }
}

/// Progress bar plus "transferred / total · speed", the paused state, or the
/// failure reason.
struct DownloadStatusLine: View {
    let state: ModelDownloadState
    let totalBytes: Int64?
    let speed: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            switch state {
            case let .downloading(progress):
                ProgressView(value: progress).tint(.accentColor)
            case let .paused(progress):
                ProgressView(value: progress).tint(.secondary)
            default:
                EmptyView()
            }
            Text(caption)
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(isFailure ? Color.red : Color.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }

    private var isFailure: Bool {
        if case .failed = state { return true }
        return false
    }

    private var caption: String {
        switch state {
        case let .downloading(progress):
            guard progress > 0 || speed != nil else { return AppText.text("download.status.preparing") }
            var text = transferred(progress)
            if let speed, speed > 0 { text += " · " + ByteText.file(Int64(speed)) + "/s" }
            return text
        case let .paused(progress):
            return AppText.text("download.status.paused") + " · " + transferred(progress)
        case let .failed(message, _):
            return message
        case .idle, .downloaded:
            return ""
        }
    }

    private func transferred(_ progress: Double) -> String {
        guard let totalBytes else { return String(format: "%.0f%%", progress * 100) }
        return ByteText.file(Int64(Double(totalBytes) * progress)) + " / " + ByteText.file(totalBytes)
    }
}

/// Download / pause / resume / retry / cancel buttons for one model. Cancel
/// deletes the partial download, so it asks first.
struct DownloadControls: View {
    let state: ModelDownloadState
    let isOtherDownloading: Bool
    let onDownload: () -> Void
    let onPause: () -> Void
    let onCancel: () -> Void
    @State private var confirmsCancel = false

    var body: some View {
        HStack(spacing: 14) {
            switch state {
            case .idle:
                iconButton("arrow.down.circle.fill", label: "download.action.download", action: onDownload)
                    .foregroundStyle(isOtherDownloading ? Color.secondary : Color.primaryPurple)
                    .disabled(isOtherDownloading)
            case .downloading:
                iconButton("pause.circle.fill", label: "download.action.pause", action: onPause)
                    .foregroundStyle(Color.primaryPurple)
                cancelButton
            case .paused:
                iconButton("play.circle.fill", label: "download.action.resume", action: onDownload)
                    .foregroundStyle(isOtherDownloading ? Color.secondary : Color.primaryPurple)
                    .disabled(isOtherDownloading)
                cancelButton
            case let .failed(_, progress):
                iconButton("arrow.clockwise.circle.fill", label: "download.action.retry", action: onDownload)
                    .foregroundStyle(isOtherDownloading ? Color.secondary : Color.primaryPurple)
                    .disabled(isOtherDownloading)
                if progress > 0 { cancelButton }
            case .downloaded:
                EmptyView()
            }
        }
        .confirmationDialog(AppText.text("download.cancel.title"), isPresented: $confirmsCancel, titleVisibility: .visible) {
            Button(AppText.text("download.cancel.confirm"), role: .destructive, action: onCancel)
            Button(AppText.text("download.cancel.keep"), role: .cancel) {}
        } message: {
            Text(AppText.text("download.cancel.message"))
        }
    }

    private var cancelButton: some View {
        iconButton("xmark.circle", label: "download.action.cancel") { confirmsCancel = true }
            .foregroundStyle(.secondary)
    }

    private func iconButton(_ systemImage: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage).font(.title2)
        }
        .buttonStyle(.borderless)
        .accessibilityLabel(AppText.text(label))
    }
}

extension View {
    /// Asks before downloading a model that likely exceeds this device's memory.
    func tooLargeModelConfirmation(isPresented: Binding<Bool>, model: ModelInfo,
                                   onConfirm: @escaping () -> Void) -> some View {
        alert(AppText.text("fit.confirm.title"), isPresented: isPresented) {
            Button(AppText.text("fit.confirm.download"), action: onConfirm)
            Button(AppText.text("Cancel"), role: .cancel) {}
        } message: {
            Text(String(format: AppText.text("fit.confirm.message"),
                        model.expectedBytes.map(ByteText.file) ?? "-", ByteText.deviceMemory))
        }
    }
}
