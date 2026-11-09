//
//  PocketAppView.swift
//  MNNChatCore
//
//  Root views of the MNN Pocket demo app. Moved here from MNNPocketApp.swift
//  so the app target stays a thin shell (entry point + resources + signing).
//

import Foundation
import SwiftUI

// MARK: - Pocket App Root

/// Root of the MNN Pocket demo: a model picker over the bundled models, with
/// no model market / download / benchmark tabs. Picking a model opens the
/// chat; the chat header routes back here.
public struct PocketAppView: View {
    private let models: [ModelInfo]
    /// Used when the picker has to be bypassed (`--auto-preset` harness).
    private let defaultModel: ModelInfo

    @State private var selectedModel: ModelInfo?

    public init() {
        // Every LLM folder under LocalModel/ is picked up automatically, so
        // dropping in another model folder makes it appear here (the folder is
        // surfaced under its product name via ModelInfo.bundledModelName).
        let scanned = ModelInfo.getAvailableLocalModels()
        // Fall back to an explicit path so the app still boots if folder
        // scanning regresses.
        let pocket = scanned.first {
            $0.name.caseInsensitiveCompare("MNN-Pocket-0.3B") == .orderedSame
        } ?? ModelInfo(
            modelName: "MNN-Pocket-0.3B",
            tags: ["Build-In", "local"],
            categories: ["Local Models"],
            vendor: "Local",
            sources: ["local": "LocalModel/slm270M"],
            isDownloaded: true
        )
        models = scanned.isEmpty ? [pocket] : scanned
        defaultModel = pocket
    }

    public var body: some View {
        NavigationStack {
            if Self.shouldAutoSendPreset {
                // Harness path: the picker would swallow the automated run, so
                // go straight to the chat on the default model.
                PocketChatView(modelInfo: defaultModel, autoSendPreset: true)
                    .navigationBarTitleDisplayMode(.inline)
            } else {
                ModelSelectionView(models: models) { model in
                    selectedModel = model
                }
                .navigationDestination(item: $selectedModel) { model in
                    PocketChatView(modelInfo: model, onBack: { selectedModel = nil })
                        .navigationBarTitleDisplayMode(.inline)
                }
            }
        }
        .preferredColorScheme(.light)
    }

    /// `--auto-preset` launch argument: automatically send the first welcome
    /// prompt once the model finishes loading. Used by the relaunch-race
    /// reproduction harness (tmp/race_test.sh) to classify good/bad runs
    /// without manual taps.
    static var shouldAutoSendPreset: Bool {
        CommandLine.arguments.contains("--auto-preset")
    }
}

// MARK: - Model Picker

/// Bundled-model picker. Deliberately minimal (no search / rename / download):
/// it lists what `ModelInfo.getAvailableLocalModels()` finds on disk.
struct ModelSelectionView: View {
    let models: [ModelInfo]
    let onSelect: (ModelInfo) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text(AppText.text("Choose a model"))
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(PocketPalette.primaryText)
                    .padding(.top, 8)

                Text(AppText.text("Runs fully on device. Your data never leaves it."))
                    .font(.system(size: 13))
                    .foregroundColor(PocketPalette.caption)
                    .padding(.bottom, 4)

                ForEach(models, id: \.id) { model in
                    Button {
                        onSelect(model)
                    } label: {
                        modelRow(model)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
        .background(PocketPalette.background)
        .navigationTitle("")
        .toolbar(.hidden, for: .navigationBar)
    }

    private func modelRow(_ model: ModelInfo) -> some View {
        HStack(spacing: 12) {
            // The asset ships with a white backdrop, so clipping it to a circle
            // reads as an app badge on the light grey card.
            Image("mnn_icon")
                .resizable()
                .scaledToFit()
                .frame(width: 40, height: 40)
                .clipShape(Circle())
                // Speculative-decoding variants get a bolt overlapping the logo,
                // so the Pocket variants are distinguishable at a glance. The
                // ring is card-coloured to lift the badge off the logo.
                .overlay(alignment: .bottomTrailing) {
                    if model.isSpeculativeDecoding {
                        Image(systemName: "bolt.fill")
                            .font(.system(size: 9, weight: .black))
                            .foregroundColor(.white)
                            .frame(width: 18, height: 18)
                            .background(Circle().fill(PocketPalette.accent))
                            .overlay(Circle().strokeBorder(PocketPalette.surface, lineWidth: 1.5))
                            .offset(x: 3, y: 3)
                    }
                }

            VStack(alignment: .leading, spacing: 3) {
                Text(model.name)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(PocketPalette.primaryText)

                if !model.localizedTags.isEmpty {
                    Text(model.localizedTags.joined(separator: " · "))
                        .font(.system(size: 12))
                        .foregroundColor(PocketPalette.caption)
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 0)

            Image(systemName: "chevron.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(PocketPalette.caption)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(RoundedRectangle(cornerRadius: 14).fill(PocketPalette.surface))
    }
}
