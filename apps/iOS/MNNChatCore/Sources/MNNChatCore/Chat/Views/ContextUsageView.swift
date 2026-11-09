import SwiftUI

struct ContextUsageButton: View {
    @ObservedObject var viewModel: LLMChatViewModel
    @State private var isExpanded = false

    var body: some View {
        Button { isExpanded.toggle() } label: {
            Image(systemName: "rectangle.stack")
                .font(.system(size: 20))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(viewModel.contextUsage.fraction(limit: viewModel.contextLimitTokens) >= 0.8 ? .orange : .blue)
                .frame(width: 44, height: 48)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(AppText.text("context.showUsage"))
        .accessibilityValue("\(viewModel.contextUsage.totalTokens) / \(viewModel.contextLimitTokens) tokens")
        .accessibilityIdentifier("chat.contextUsage")
        .popover(isPresented: $isExpanded, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text("context.usage").font(.headline)
                    Spacer()
                    Button { isExpanded = false } label: {
                        Image(systemName: "xmark")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 32, height: 32)
                            .background(Color(uiColor: .tertiarySystemFill), in: Circle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(AppText.text("context.hideUsage"))
                    .accessibilityIdentifier("context.close")
                }

                ContextUsageView(viewModel: viewModel)
                Divider()

                VStack(spacing: 16) {
                    Menu {
                        Picker("context.limit", selection: Binding(
                            get: { viewModel.contextLimitTokens }, set: { viewModel.setContextTokenLimit($0) }
                        )) {
                            ForEach(viewModel.contextLimitChoices, id: \.self) { limit in
                                    Text("\(limit.formatted()) tokens").tag(limit)
                                }
                        }
                    } label: {
                        HStack {
                            Text("context.limit").foregroundStyle(.primary)
                            Spacer()
                            Text("\(viewModel.contextLimitTokens.formatted()) tokens").monospacedDigit()
                            Image(systemName: "chevron.up.chevron.down").font(.caption2)
                        }
                        .font(.subheadline)
                    }
                    .disabled(viewModel.isProcessing)
                    .accessibilityIdentifier("context.popoverLimit")

                    if let recommendation = viewModel.contextRecommendation {
                        Button {
                            viewModel.setContextTokenLimit(recommendation.tokens)
                        } label: {
                            HStack {
                                Text("context.useRecommendation")
                                Spacer()
                                Text("\(recommendation.tokens.formatted()) tokens").monospacedDigit()
                            }
                        }
                        .disabled(viewModel.isProcessing)
                        .accessibilityIdentifier("context.popoverRecommendation")
                    }

                    Toggle("conversation.context.toggle", isOn: Binding(
                        get: { viewModel.isConversationContextEnabled },
                        set: { viewModel.setConversationContextEnabled($0) }
                    ))
                    .disabled(viewModel.isProcessing)
                    .accessibilityIdentifier("context.popoverConversation")

                    Toggle("context.autoCompress", isOn: Binding(
                        get: { viewModel.autoCompressContext }, set: { viewModel.setAutoCompressContext($0) }
                    ))
                    .disabled(viewModel.isProcessing || !viewModel.isConversationContextEnabled)
                    .accessibilityIdentifier("context.popoverAutoCompress")
                }
                .font(.subheadline)
                .tint(.blue)
            }
            .padding(20)
            .frame(width: min(340, UIScreen.main.bounds.width - 40))
            .background(Color(uiColor: .systemBackground), in: RoundedRectangle(cornerRadius: 24))
            .presentationCompactAdaptation(.popover)
            .accessibilityIdentifier("context.details")
            .onAppear { viewModel.refreshContextUsage() }
        }
    }
}

struct ContextUsageView: View {
    @ObservedObject var viewModel: LLMChatViewModel

    private var fraction: Double {
        viewModel.contextUsage.fraction(limit: viewModel.contextLimitTokens)
    }

    private var tint: Color { fraction >= 0.8 ? .orange : .blue }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(viewModel.contextUsage.totalTokens.formatted())
                        .font(.system(size: 28, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary)
                        .monospacedDigit()
                    Text("/ \(viewModel.contextLimitTokens.formatted()) tokens")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    Spacer(minLength: 4)
                    Text(fraction.formatted(.percent.precision(.fractionLength(0))))
                        .font(.caption.weight(.semibold))
                        .monospacedDigit()
                        .foregroundStyle(tint)
                        .padding(.horizontal, 8).padding(.vertical, 5)
                        .background(tint.opacity(0.1), in: Capsule())
                }

                GeometryReader { geometry in
                    Capsule().fill(Color(uiColor: .tertiarySystemFill))
                        .overlay(alignment: .leading) {
                            Capsule().fill(tint.gradient).frame(width: geometry.size.width * fraction)
                        }
                }
                .frame(height: 7)
                .accessibilityLabel(AppText.text("context.usage"))
                .accessibilityValue(fraction.formatted(.percent.precision(.fractionLength(0))))
                .accessibilityIdentifier("context-usage-progress")
            }

            HStack(spacing: 12) {
                metric("context.input", subtitle: "context.inputDescription",
                       value: viewModel.contextUsage.inputTokens, icon: "arrow.down.left")
                metric("context.generated", subtitle: "context.includesThinking",
                       value: viewModel.contextUsage.generatedTokens, icon: "sparkles")
            }

            if viewModel.isCompactingContext {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("context.compacting").font(.caption).foregroundStyle(.secondary)
                }
            } else if let notice = viewModel.contextNotice {
                Label(notice, systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityIdentifier("context.usageSummary")
    }

    private func metric(_ title: LocalizedStringKey, subtitle: LocalizedStringKey, value: Int, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Label(title, systemImage: icon)
                .font(.caption).foregroundStyle(.secondary)
            Text(value.formatted()).font(.system(.title3, design: .rounded).weight(.semibold)).monospacedDigit()
            Text(subtitle).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}
