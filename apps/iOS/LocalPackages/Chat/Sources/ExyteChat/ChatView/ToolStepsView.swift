import SwiftUI

/// Agent tool calls of one message, separate from the thinking panel.
/// Expanded while tools run; folds into a one-line summary once the answer
/// starts, and a manual tap always wins.
@available(iOS 17.0, *)
struct ToolStepsView: View {
    let steps: [ToolStep]
    /// True once the answer has started (or the output is complete).
    let finished: Bool
    let messageId: String

    @State private var manualExpansion: Bool?

    private var isExpanded: Bool { manualExpansion ?? !finished }
    private var isRunning: Bool { steps.contains { $0.status == .running } }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.25)) { manualExpansion = !isExpanded }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "wrench.and.screwdriver")
                        .font(.system(size: 13, weight: .medium))
                    Text(ChatTextLocalization.text("agent.tools.title"))
                        .font(.system(size: 14, weight: .medium))
                    Text("\(steps.filter { $0.status != .skipped }.count)")
                        .font(.system(size: 11, weight: .medium))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.12)))
                    if isRunning { ProgressView().controlSize(.mini) }
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 12, weight: .medium))
                }
                .foregroundColor(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("tool-steps-" + messageId)

            if isExpanded {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(Array(steps.enumerated()), id: \.offset) { _, step in
                        row(step)
                    }
                }
                .transition(.opacity)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.08)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.15), lineWidth: 1))
        .onChange(of: finished) { _, isFinished in
            if !isFinished { manualExpansion = nil }
        }
    }

    private func row(_ step: ToolStep) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon(step.kind))
                .font(.system(size: 13))
                .foregroundColor(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(step.title).font(.system(size: 13, weight: .medium))
                    if !step.result.isEmpty {
                        Text("· " + step.result).font(.system(size: 12)).foregroundColor(.secondary)
                    }
                }
                if !step.detail.isEmpty {
                    Text(step.detail)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
            }
            Spacer(minLength: 4)
            status(step.status)
        }
        .opacity(step.status == .skipped ? 0.55 : 1)
    }

    @ViewBuilder
    private func status(_ status: ToolStep.Status) -> some View {
        switch status {
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.system(size: 13))
        case .failed: Image(systemName: "xmark.circle.fill").foregroundColor(.red).font(.system(size: 13))
        case .skipped: Image(systemName: "minus.circle").foregroundColor(.secondary).font(.system(size: 13))
        }
    }

    private func icon(_ kind: ToolStep.Kind) -> String {
        switch kind {
        case .search: "magnifyingglass"
        case .browse: "doc.text"
        case .time: "clock"
        case .memory: "bookmark"
        case .calculate: "function"
        case .other: "wrench"
        }
    }
}
