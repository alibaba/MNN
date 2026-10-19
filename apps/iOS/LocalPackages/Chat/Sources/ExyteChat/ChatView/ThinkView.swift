//
//  ThinkView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/1/25.
//

import SwiftUI

/// A view component for displaying AI thinking process content.
///
/// Behaves as a sub-panel that is expanded while the model is thinking and
/// automatically folds into a one-line pill once the answer starts streaming
/// (`thinkingFinished`), keeping the main bubble focused on the answer.
/// The pill stays tappable so the thinking content remains reviewable, and a
/// manual tap always wins over the auto-fold until the next thinking round.
struct ThinkView: View {
    let thinkContent: String
    let messageId: String
    /// True once the answer body has started (or the output is complete).
    var thinkingFinished: Bool = false

    @State private var expansion: ThinkingPanelExpansion
    private var isExpanded: Bool { expansion.isExpanded }
    @State private var contentHeight: CGFloat = 24
    @State private var followsLatest = true

    // Reserve room for the header and panel padding within the one-fifth cap.
    private var maximumContentHeight: CGFloat {
        max(24, UIScreen.main.bounds.height * 0.2 - 44)
    }

    private var scrollAnchor: String { messageId + "-thinking-bottom" }

    init(thinkContent: String, thinkingFinished: Bool = false, messageId: String = "thinking-preview") {
        self.thinkContent = thinkContent
        self.messageId = messageId
        self.thinkingFinished = thinkingFinished
        // Completed messages (e.g. restored history) start folded; only an
        // in-flight thinking round starts expanded.
        _expansion = State(initialValue: ThinkingPanelExpansion(thinkingFinished: thinkingFinished))
    }

    private let animationDuration: Double = 0.3

    var body: some View {
        if !thinkContent.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                // Header: status + expand/collapse affordance
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.secondary)
                        .font(.system(size: 14, weight: .medium))

                    Text(ChatTextLocalization.text("AI Thinking Process"))
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)

                    if thinkingFinished {
                        Text(ChatTextLocalization.text("think.done", defaultValue: "已完成"))
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.secondary.opacity(0.8))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.secondary.opacity(0.12)))
                    }

                    Spacer()

                    Button(action: {
                        withAnimation(.easeInOut(duration: animationDuration)) {
                            expansion.toggle()
                        }
                    }) {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(.secondary)
                            .font(.system(size: 12, weight: .medium))
                    }
                    .buttonStyle(PlainButtonStyle())
                    .accessibilityLabel(Text(ChatTextLocalization.text(isExpanded ? "think.collapse" : "think.expand")))
                    .accessibilityValue(isExpanded ? "expanded" : "collapsed")
                    .accessibilityIdentifier("thinking-toggle-" + messageId)
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, isExpanded ? 0 : 8)

                // Keep long thinking inside the panel instead of growing the chat bubble.
                if isExpanded {
                    ScrollViewReader { proxy in
                        ScrollView(.vertical) {
                            VStack(alignment: .leading, spacing: 0) {
                                thinkingText
                                Color.clear.frame(height: 1).id(scrollAnchor)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(GeometryReader { geometry in
                                Color.clear.preference(key: ThinkingContentHeightKey.self,
                                                       value: geometry.size.height)
                            })
                        }
                        .frame(height: min(contentHeight, maximumContentHeight))
                        .onPreferenceChange(ThinkingContentHeightKey.self) { height in
                            let boundedHeight = min(height, maximumContentHeight)
                            if abs(contentHeight - boundedHeight) > 0.5 {
                                contentHeight = boundedHeight
                            }
                            if followsLatest && !thinkingFinished {
                                DispatchQueue.main.async {
                                    proxy.scrollTo(scrollAnchor, anchor: .bottom)
                                }
                            }
                        }
                        .simultaneousGesture(DragGesture(minimumDistance: 5).onChanged { _ in
                            // Let the user review earlier text while generation continues.
                            followsLatest = false
                        })
                        .onAppear {
                            followsLatest = !thinkingFinished
                            if followsLatest {
                                DispatchQueue.main.async {
                                    proxy.scrollTo(scrollAnchor, anchor: .bottom)
                                }
                            }
                        }
                    }
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
            )
            .padding(.horizontal, 4)
            .padding(.vertical, 4)
            .onChange(of: thinkingFinished) { finished in
                // Auto-fold the moment the answer takes over.
                if finished {
                    withAnimation(.easeInOut(duration: animationDuration)) {
                        expansion.update(thinkingFinished: finished)
                    }
                }
            }
            .onChange(of: thinkContent) { _ in
                // Incoming chunks belong to this same message/round. A manual
                // fold stays in place; a new message gets its own view state.
                expansion.update(thinkingFinished: thinkingFinished)
            }
        }
    }

    @ViewBuilder
    private var thinkingText: some View {
        Group {
            if #available(iOS 17.0, *) {
                LLMMessageTextView(
                    text: thinkContent,
                    messageUseMarkdown: false,
                    messageId: messageId + "-thinking",
                    isAssistantMessage: true,
                    isStreamingMessage: !thinkingFinished,
                    isOutputComplete: thinkingFinished,
                    plainTextFont: .system(size: 13, weight: .regular),
                    plainTextColor: .primary.opacity(0.8)
                )
            } else {
                Text(thinkContent)
                    .font(.system(size: 13, weight: .regular))
                    .foregroundColor(.primary.opacity(0.8))
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .multilineTextAlignment(.leading)
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }
}

/// A manual choice lasts for the whole message, including later stream chunks.
struct ThinkingPanelExpansion {
    private(set) var isExpanded: Bool
    private var userToggled = false

    init(thinkingFinished: Bool) {
        isExpanded = !thinkingFinished
    }

    mutating func toggle() {
        userToggled = true
        isExpanded.toggle()
    }

    mutating func update(thinkingFinished: Bool) {
        guard !userToggled else { return }
        isExpanded = !thinkingFinished
    }
}

private struct ThinkingContentHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

#if DEBUG
struct ThinkView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 16) {
            // Thinking in progress: expanded
            ThinkView(thinkContent: "This is a short thinking process.", thinkingFinished: false)

            // Thinking finished: folded into a pill
            ThinkView(thinkContent: "This is a much longer thinking process that demonstrates how the component folds once the answer starts streaming.", thinkingFinished: true)

            // Empty content (should not display)
            ThinkView(thinkContent: "")
        }
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
#endif
