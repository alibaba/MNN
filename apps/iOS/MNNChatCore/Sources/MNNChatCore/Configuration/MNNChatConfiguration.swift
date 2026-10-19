//
//  MNNChatConfiguration.swift
//  MNNChatCore
//
//  Per-app behavior switches for the shared chat stack. Each app shell calls
//  `MNNChatConfiguration.install(_:)` once from its @main init before any
//  view is created. The configuration is process-global and read-only after
//  installation, so it is safe to read from any thread.
//

import Foundation

public struct MNNChatConfiguration: Sendable {
    public enum Variant: Sendable {
        /// Full-featured MNN Chat app (model market, benchmark, history).
        case classic
        /// Single-model MNN Pocket demo (bundled models, dark welcome screen).
        case pocket
    }

    public var variant: Variant

    /// Fresh conversations open with a greeting bubble (classic) or with an
    /// empty conversation behind a branded welcome screen (pocket).
    public var showsGreetingBubble: Bool

    /// While the model loads, post a "Model Loading..." chat bubble (classic)
    /// or surface progress only through the chat status line (pocket).
    public var showsModelLoadingBubble: Bool

    /// Rate-limit markdown re-rendering of the streaming answer (~60fps
    /// throttle) instead of re-parsing per token.
    public var streamsAnswerThrottled: Bool

    /// Keep `MNN_*` folders (Metal pipeline / tuning cache) when cleaning the
    /// tmp directory, avoiding a cold pipeline recompile on every launch.
    public var preservesMNNCacheInTmp: Bool

    /// Match preset prompts by model-family prefix instead of exact name, so
    /// variants (e.g. "-DFlash") share one prompt set.
    public var presetPromptPrefixMatching: Bool

    /// Badge bundled models whose name indicates a speculative-decoding stack.
    public var showsSpeculativeBadge: Bool

    public init(variant: Variant,
                showsGreetingBubble: Bool,
                showsModelLoadingBubble: Bool,
                streamsAnswerThrottled: Bool,
                preservesMNNCacheInTmp: Bool,
                presetPromptPrefixMatching: Bool,
                showsSpeculativeBadge: Bool) {
        self.variant = variant
        self.showsGreetingBubble = showsGreetingBubble
        self.showsModelLoadingBubble = showsModelLoadingBubble
        self.streamsAnswerThrottled = streamsAnswerThrottled
        self.preservesMNNCacheInTmp = preservesMNNCacheInTmp
        self.presetPromptPrefixMatching = presetPromptPrefixMatching
        self.showsSpeculativeBadge = showsSpeculativeBadge
    }

    public static let classic = MNNChatConfiguration(
        variant: .classic,
        showsGreetingBubble: true,
        showsModelLoadingBubble: true,
        streamsAnswerThrottled: false,
        preservesMNNCacheInTmp: false,
        presetPromptPrefixMatching: false,
        showsSpeculativeBadge: false
    )

    public static let pocket = MNNChatConfiguration(
        variant: .pocket,
        showsGreetingBubble: false,
        showsModelLoadingBubble: false,
        streamsAnswerThrottled: true,
        preservesMNNCacheInTmp: true,
        presetPromptPrefixMatching: true,
        showsSpeculativeBadge: true
    )

    /// Active configuration. Defaults to `.classic` so code paths taken before
    /// `install(_:)` (unit tests, previews) behave like the original app.
    public private(set) static var current: MNNChatConfiguration = .classic

    /// Installs the process-wide configuration. Call once from @main init.
    public static func install(_ configuration: MNNChatConfiguration) {
        current = configuration
    }
}
