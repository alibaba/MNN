//
//  ModelSettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/17.
//

import Foundation
import SwiftUI

struct ModelSettingsView: View {
    @Binding var showSettings: Bool
    @ObservedObject var viewModel: LLMChatViewModel
    @State private var showAlert = false
    @State private var showReloadAlert = false
    @State private var iterations: Int = 20
    @State private var seed: Int = 42
    @State private var useRandomSeed: Bool = false
    @State private var backendType: String = "metal"
    @State private var precision: String = "low"
    @State private var threadNum: Int = 4
    @State private var maxNewTokens: Int = 128
    @State private var requiresReload = false
    @AppStorage("chat.showPrefillSpeed") private var showPrefillSpeed = true
    @AppStorage("chat.showDecodeSpeed") private var showDecodeSpeed = true

    @State private var temperature: Double = 1.0
    @State private var topK: Double = 40
    @State private var topP: Double = 0.9
    @State private var minP: Double = 0.1

    @State private var tfsZ: Double = 1.0
    @State private var typical: Double = 1.0
    @State private var penalty: Double = 0.0
    @State private var nGram: Double = 8.0
    @State private var nGramFactor: Double = 1.0

    @State private var selectedSampler: SamplerType = .temperature
    @State private var selectedMixedSamplers: Set<String> = []
    @State private var mixedSamplersOrder: [String] = []

    @State private var penaltySampler: PenaltySamplerType = .greedy
    @State private var videoMaxFrames: Int = 8
    @State private var defaultMultimodalPrompt: String = ""
    @State private var enableAudioOutput: Bool = false
    @State private var talkerSpeaker: String = "default"

    // iOS supported backends: CPU, Metal (GPU), NPU (CoreML/Neural Engine)
    // Note: "npu" maps to MNN_FORWARD_NN which uses CoreML on iOS
    private let backendOptions = ["cpu", "metal", "npu"]
    private let precisionOptions = ["low", "high"]
    private let talkerSpeakerOptions = ["Ethan", "Chelsie"]

    /// Mixed and Greedy only; a model config already using another sampler
    /// keeps it listed so the picker still shows the active choice.
    private var samplerChoices: [SamplerType] {
        let choices: [SamplerType] = [.mixed, .greedy]
        return choices.contains(selectedSampler) ? choices : choices + [selectedSampler]
    }

    var body: some View {
        NavigationView {
            Form {
                if !viewModel.isAnyDiffusionModel {
                    Section {
                        Toggle(isOn: $showPrefillSpeed) { SettingLabel("chat.performance.prefill", help: "help.prefill") }
                            .accessibilityIdentifier("chat.showPrefillSpeed")
                        Toggle(isOn: $showDecodeSpeed) { SettingLabel("chat.performance.decode", help: "help.decode") }
                            .accessibilityIdentifier("chat.showDecodeSpeed")
                    } header: {
                        Text("chat.performance.section")
                    } footer: {
                        Text("chat.performance.description")
                    }

                    Section {
                        SettingToggleRow(titleKey: "conversation.context.toggle", helpKey: "help.context",
                                         isOn: Binding(get: { viewModel.isConversationContextEnabled },
                                                       set: { viewModel.setConversationContextEnabled($0) }),
                                         isDisabled: viewModel.isProcessing)
                        .accessibilityIdentifier("conversation-context-toggle")

                        HStack {
                            SettingLabel("context.limit", help: "help.contextLimit", isDimmed: viewModel.isProcessing)
                            Spacer(minLength: 8)
                            Picker("", selection: Binding(
                                get: { viewModel.contextLimitTokens }, set: { viewModel.setContextTokenLimit($0) }
                            )) {
                                ForEach(viewModel.contextLimitChoices, id: \.self) { limit in
                                    Text("\(limit) tokens").tag(limit)
                                }
                            }
                            .labelsHidden()
                            .disabled(viewModel.isProcessing)
                        }
                        .accessibilityIdentifier("context-limit-picker")

                        SettingToggleRow(titleKey: "context.autoCompress", helpKey: "help.autoCompress",
                                         isOn: Binding(get: { viewModel.autoCompressContext },
                                                       set: { viewModel.setAutoCompressContext($0) }),
                                         isDisabled: viewModel.isProcessing || !viewModel.isConversationContextEnabled)

                        ContextUsageView(viewModel: viewModel)

                        HStack {
                            SettingLabel(viewModel.modelContextLimit?.fromOriginalModel == true ? "context.originalMaximum" : "context.modelMaximum",
                                         help: "help.modelMaximum")
                            Spacer()
                            if let maximum = viewModel.modelMaximumContextTokens {
                                Text("\(maximum) tokens").foregroundStyle(.secondary)
                            } else { Text("context.unknown").foregroundStyle(.secondary) }
                        }
                        HStack {
                            SettingLabel("context.deviceRecommendation", help: "help.deviceRecommendation")
                            Spacer()
                            if let recommendation = viewModel.contextRecommendation {
                                Text("\(recommendation.tokens.formatted()) tokens").foregroundStyle(.secondary)
                            } else { Text("context.recommendationUnavailable").foregroundStyle(.secondary) }
                        }
                        if let recommendation = viewModel.contextRecommendation {
                            HStack {
                                SettingLabel("context.cachePerThousandTokens", help: "help.kvCache")
                                Spacer()
                                Text(String(format: "%.1f MB", recommendation.kvBytesPerToken * 1024 / (1024 * 1024)))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if let bytes = viewModel.contextUsage.availableMemoryBytes {
                            HStack {
                                Text("context.availableMemory")
                                Spacer()
                                Text("\(bytes / (1024 * 1024)) MB").foregroundStyle(.secondary)
                            }
                        }
                        Button("context.useRecommendation") { viewModel.setContextTokenLimit(viewModel.recommendedContextTokens) }
                            .disabled(viewModel.isProcessing || viewModel.contextRecommendation == nil)

                        if let recommendation = viewModel.contextRecommendation, viewModel.contextLimitTokens > recommendation.tokens {
                            Text("context.aboveRecommendation").font(.footnote).foregroundStyle(.secondary)
                        }

                        Button("conversation.context.newChat") {
                            viewModel.resetConversation()
                            showSettings = false
                        }
                        .disabled(viewModel.chatInputUnavilable)
                    } header: {
                        Text("conversation.context.section")
                    } footer: {
                        Text("context.settings.description")
                    }
                }

                Section {
                    Toggle(isOn: $viewModel.useMmap) { SettingLabel("Use mmap", help: "help.mmap") }
                        .onChange(of: viewModel.useMmap) { newValue in
                            requiresReload = true
                            viewModel.modelConfigManager.updateUseMmap(newValue)
                        }

                    if viewModel.supportsThinkingMode {
                        Toggle(isOn: Binding(
                            get: { viewModel.isThinkingModeEnabled },
                            set: { newValue in
                                if newValue != viewModel.isThinkingModeEnabled {
                                    viewModel.toggleThinkingMode()
                                }
                            }
                        )) { SettingLabel("Thinking Mode", help: "help.thinking") }
                        // Applies live through setThinkingModeEnabled; no model
                        // reload needed (unlike backend/precision/mmap).
                    }

                    HStack(spacing: 4) {
                        Button("Clear mmap Cache") {
                            viewModel.cleanModelTmpFolder()
                            showAlert = true
                        }
                        .buttonStyle(.borderless)
                        SettingHelpButton(titleKey: "Clear mmap Cache", helpKey: "help.clearMmap")
                        Spacer()
                    }

                    Picker(selection: $backendType) {
                        ForEach(backendOptions, id: \.self) { backend in
                            Text(backend.uppercased()).tag(backend)
                        }
                    } label: { SettingLabel("Backend", help: "help.backend") }
                    .onChange(of: backendType) { _, newValue in
                        requiresReload = true
                        viewModel.modelConfigManager.updateBackendType(newValue)
                    }

                    Picker(selection: $precision) {
                        ForEach(precisionOptions, id: \.self) { option in
                            Text(option.capitalized).tag(option)
                        }
                    } label: { SettingLabel("Precision", help: "help.precision") }
                    .onChange(of: precision) { _, newValue in
                        requiresReload = true
                        viewModel.modelConfigManager.updatePrecision(newValue)
                    }

                    Stepper(value: $threadNum, in: 1 ... viewModel.modelConfigManager.maxThreads) {
                        HStack {
                            SettingLabel("Threads", help: "help.threads")
                            Spacer()
                            Text("\(threadNum)")
                        }
                    }
                    .onChange(of: threadNum) { _, newValue in
                        requiresReload = true
                        viewModel.modelConfigManager.updateThreadNum(newValue)
                    }
                } header: {
                    Text("Model Configuration")
                }

                if !viewModel.isAnyDiffusionModel {
                    Section {
                        HStack {
                            SettingLabel("Max Output Tokens", help: "help.maxTokens")
                            Spacer()
                            TextField("128", value: $maxNewTokens, format: .number.grouping(.never))
                                .keyboardType(.numberPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                                .onChange(of: maxNewTokens) { _, newValue in
                                    let clamped = max(1, min(4096, newValue))
                                    if clamped != newValue {
                                        maxNewTokens = clamped
                                    } else {
                                        viewModel.modelConfigManager.updateMaxNewTokens(clamped)
                                    }
                                }
                        }
                    } header: {
                        Text("Generation")
                    } footer: {
                        Text("Enter any integer from 1 to 4096.")
                    }
                }

                Section {
                    // FIXME: Hidden The new Multimodal Prompt API
                    // Toggle("Use Multimodal Prompt API", isOn: $viewModel.useMultimodalPromptAPI)
                    //    .onChange(of: viewModel.useMultimodalPromptAPI) { _, newValue in
                    //        viewModel.updateUseMultimodalPromptAPI(newValue)
                    //    }

                    Stepper(value: $videoMaxFrames, in: 1 ... 32) {
                        HStack {
                            SettingLabel("Video Frames", help: "help.videoFrames")
                            Spacer()
                            Text("\(videoMaxFrames)")
                        }
                    }
                    .onChange(of: videoMaxFrames) { _, newValue in
                        viewModel.updateVideoMaxFrames(newValue)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        SettingLabel("Default Multimodal Prompt", help: "help.multimodalPrompt")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        TextEditor(text: $defaultMultimodalPrompt)
                            .frame(minHeight: 80)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.secondary.opacity(0.3))
                            )
                            .onChange(of: defaultMultimodalPrompt) { _, newValue in
                                viewModel.updateDefaultMultimodalPrompt(newValue)
                            }
                    }
                } header: {
                    Text("Multimodal Inputs")
                }

                // Audio Output Settings (Omni)
                if ModelUtils.supportAudioOutput(viewModel.modelInfo.name) {
                    Section {
                        Toggle(isOn: $enableAudioOutput) { SettingLabel("Enable Audio Output", help: "help.audioOutput") }
                            .onChange(of: enableAudioOutput) { _, newValue in
                                print("[AudioUI] Enable Audio Output changed to: \(newValue)")
                                viewModel.updateEnableAudioOutput(newValue)
                            }

                        Picker(selection: $talkerSpeaker) {
                            ForEach(talkerSpeakerOptions, id: \.self) { option in
                                Text(option.capitalized).tag(option)
                            }
                        } label: { SettingLabel("Talker Speaker", help: "help.speaker") }
                        .onChange(of: talkerSpeaker) { _, newValue in
                            print("[AudioUI] Talker Speaker changed to: \(newValue)")
                            viewModel.updateTalkerSpeaker(newValue)
                        }
                    } header: {
                        Text("Audio Output (Omni)")
                    }
                }

                // Diffusion Settings (Stable Diffusion and Sana Diffusion)
                if viewModel.isAnyDiffusionModel {
                    Section {
                        // Model type indicator for Sana Diffusion
                        if viewModel.isSanaDiffusionModel {
                            HStack {
                                Text("Model Type")
                                Spacer()
                                Text("Sana Style Transfer")
                                    .foregroundColor(.secondary)
                            }
                        }

                        Stepper(value: $iterations, in: 1 ... 100) {
                            HStack {
                                SettingLabel("Iterations", help: "help.iterations")
                                Spacer()
                                Text("\(iterations)")
                            }
                        }
                        .onChange(of: iterations) { _, newValue in
                            viewModel.modelConfigManager.updateIterations(newValue)
                        }

                        Toggle(isOn: $useRandomSeed) { SettingLabel("Random Seed", help: "help.randomSeed") }
                            .onChange(of: useRandomSeed) { _, newValue in
                                if newValue {
                                    seed = -1
                                    viewModel.modelConfigManager.updateSeed(-1)
                                }
                            }

                        if !useRandomSeed {
                            HStack {
                                SettingLabel("Seed", help: "help.seed")
                                Spacer()
                                TextField("Seed", value: $seed, formatter: NumberFormatter())
                                    .keyboardType(.numberPad)
                                    .multilineTextAlignment(.trailing)
                                    .frame(width: 100)
                                    .onChange(of: seed) { _, newValue in
                                        viewModel.modelConfigManager.updateSeed(max(0, newValue))
                                    }
                            }
                        }
                    } header: {
                        Text(viewModel.isSanaDiffusionModel ? "Style Transfer Settings" : "Diffusion Settings")
                    }
                } else {
                    Section {
                        Picker(selection: $selectedSampler) {
                            ForEach(samplerChoices, id: \.self) { sampler in
                                Text(sampler.displayName)
                                    .tag(sampler)
                            }
                        } label: { SettingLabel("Sampler Type", help: "help.sampler") }
                        .onChange(of: selectedSampler) { _, newValue in
                            updateSamplerType(newValue)
                        }

                        switch selectedSampler {
                        case .temperature:
                            ParameterSliderView(
                                title: "Temperature",
                                value: $temperature,
                                range: 0.0 ... 2.0,
                                format: "%.2f",
                                intValue: false,
                                onChanged: updateTemperature(_:)
                            )
                        case .topK:
                            ParameterSliderView(
                                title: "Top K",
                                value: $topK,
                                range: 1 ... 100,
                                format: "%.0f",
                                intValue: true,
                                onChanged: { updateTopK(Int($0)) }
                            )
                        case .topP:
                            ParameterSliderView(
                                title: "Top P",
                                value: $topP,
                                range: 0.0 ... 1.0,
                                format: "%.2f",
                                intValue: false,
                                onChanged: updateTopP(_:)
                            )
                        case .minP:
                            ParameterSliderView(
                                title: "Min P",
                                value: $minP,
                                range: 0.05 ... 0.3,
                                format: "%.2f",
                                intValue: false,
                                onChanged: updateMinP(_:)
                            )
                        case .tfs:
                            ParameterSliderView(
                                title: "TFS-Z",
                                value: $tfsZ,
                                range: 0.9 ... 0.99,
                                format: "%.2f",
                                intValue: false,
                                onChanged: updateTfsZ(_:)
                            )
                        case .typical:
                            ParameterSliderView(
                                title: "Typical",
                                value: $typical,
                                range: 0.8 ... 0.95,
                                format: "%.2f",
                                intValue: false,
                                onChanged: updateTypical(_:)
                            )
                        case .penalty:
                            VStack(spacing: 8) {
                                // Penalty parameters
                                ParameterSliderView(
                                    title: "Penalty",
                                    value: $penalty,
                                    range: SamplerType.penaltyRange,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: updatePenalty(_:)
                                )

                                // N-gram Size parameters
                                ParameterSliderView(
                                    title: "N-gram Size",
                                    value: $nGram,
                                    range: 3 ... 8,
                                    format: "%.0f",
                                    intValue: true,
                                    onChanged: { updateNGram(Int($0)) }
                                )

                                // N-gram Factor parameters
                                ParameterSliderView(
                                    title: "N-gram Factor",
                                    value: $nGramFactor,
                                    range: 1.0 ... 3.0,
                                    format: "%.1f",
                                    intValue: false,
                                    onChanged: updateNGramFactor(_:)
                                )

                                // Penalty Sampler selector
                                Picker(selection: $penaltySampler) {
                                    ForEach(PenaltySamplerType.allCases, id: \.self) { samplerType in
                                        Text(samplerType.displayName)
                                            .tag(samplerType)
                                    }
                                } label: { SettingLabel("Penalty Sampler", help: "help.penaltySampler") }
                                .onChange(of: penaltySampler) { newValue in
                                    updatePenaltySampler(newValue)
                                }
                            }
                        case .mixed:
                            MixedSamplersView(
                                selectedSamplers: $selectedMixedSamplers,
                                samplersOrder: $mixedSamplersOrder,
                                onUpdate: updateMixedSamplers,
                                temperature: $temperature,
                                topK: $topK,
                                topP: $topP,
                                minP: $minP,
                                tfsZ: $tfsZ,
                                typical: $typical,
                                penalty: $penalty,
                                onUpdateTemperature: updateTemperature(_:),
                                onUpdateTopK: updateTopK(_:),
                                onUpdateTopP: updateTopP(_:),
                                onUpdateMinP: updateMinP(_:),
                                onUpdateTfsZ: updateTfsZ(_:),
                                onUpdateTypical: updateTypical(_:),
                                onUpdatePenalty: updatePenalty(_:)
                            )
                        default:
                            EmptyView()
                        }
                    } header: {
                        Text("Sampling Strategy")
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        if requiresReload {
                            showReloadAlert = true
                        } else {
                            showSettings = false
                        }
                    }
                }
            }
        }
        .alert(AppText.text("Success", comment: ""), isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(AppText.text("Cache Cleared Successfully", comment: ""))
        }
        .alert(AppText.text("Model Reload Required", comment: ""), isPresented: $showReloadAlert) {
            Button(AppText.text("Reload Now", comment: "")) {
                viewModel.reloadCurrentModel()
                requiresReload = false
                showSettings = false
            }
            Button(AppText.text("Later", comment: ""), role: .cancel) {
                showSettings = false
            }
        } message: {
            Text(AppText.text("Changes to runtime or sampling settings require the model to reload.", comment: ""))
        }
        .onAppear {
            viewModel.refreshContextUsage()
            selectedSampler = viewModel.modelConfigManager.readSamplerType()
            backendType = viewModel.modelConfigManager.readBackendType()
            precision = viewModel.modelConfigManager.readPrecision()
            threadNum = viewModel.modelConfigManager.readThreadNum()
            maxNewTokens = viewModel.modelConfigManager.readMaxNewTokens()
            requiresReload = false

            if viewModel.isAnyDiffusionModel {
                iterations = viewModel.modelConfigManager.readIterations()
                seed = viewModel.modelConfigManager.readSeed()
                useRandomSeed = (seed < 0)
                
                // Set default iterations for Sana Diffusion (fewer iterations needed)
                if viewModel.isSanaDiffusionModel && iterations == 20 {
                    iterations = 5  // Sana uses fewer iterations by default
                }
            } else {
                temperature = viewModel.modelConfigManager.readTemperature()
                topK = Double(viewModel.modelConfigManager.readTopK())
                topP = viewModel.modelConfigManager.readTopP()
                minP = viewModel.modelConfigManager.readMinP()
                tfsZ = viewModel.modelConfigManager.readTfsZ()
                typical = viewModel.modelConfigManager.readTypical()
                // Older builds wrote values below 1.0, which the engine ignores.
                penalty = min(max(viewModel.modelConfigManager.readPenalty(), SamplerType.penaltyRange.lowerBound),
                              SamplerType.penaltyRange.upperBound)
                nGram = Double(viewModel.modelConfigManager.readNGram())
                nGramFactor = viewModel.modelConfigManager.readNGramFactor()
            }
            videoMaxFrames = viewModel.modelConfigManager.readVideoMaxFrames()
            defaultMultimodalPrompt = viewModel.modelConfigManager.readDefaultMultimodalPrompt()
            enableAudioOutput = viewModel.modelConfigManager.readEnableAudioOutput()
            
            // Validate talkerSpeaker value - if not in options, reset to default
            let savedTalkerSpeaker = viewModel.modelConfigManager.readTalkerSpeaker()
            if talkerSpeakerOptions.contains(savedTalkerSpeaker) {
                talkerSpeaker = savedTalkerSpeaker
            } else {
                talkerSpeaker = "default"
                // Update config with valid default value
                viewModel.updateTalkerSpeaker("default")
            }

            // Initialize mixed samplers
            // Every adjustable sampler is listed (penalty included, first, since
            // the engine always runs it first); saved ones keep their order.
            // Configs may use the engine's snake_case names.
            let aliases = ["top_k": "topK", "top_p": "topP", "min_p": "minP"]
            let known = ["penalty", "topK", "tfs", "typical", "topP", "minP", "temperature"]
            let saved = viewModel.modelConfigManager.readMixedSamplers()
                .map { aliases[$0] ?? $0 }
                .filter { known.contains($0) }
            mixedSamplersOrder = saved + known.filter { !saved.contains($0) }
            selectedMixedSamplers = Set(saved)

            // Initialize penalty sampler
            penaltySampler = viewModel.modelConfigManager.readPenaltySampler()
        }
        .onDisappear {
            viewModel.setModelConfig()
            viewModel.updateVideoMaxFrames(videoMaxFrames)
            viewModel.updateDefaultMultimodalPrompt(defaultMultimodalPrompt)
        }
    }

    private func updateMixedSamplers() {
        let orderedSelection = mixedSamplersOrder.filter { selectedMixedSamplers.contains($0) }
        if viewModel.modelConfigManager.readMixedSamplers() != orderedSelection {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateMixedSamplers(orderedSelection)
    }

    private func updateSamplerType(_ value: SamplerType) {
        if viewModel.modelConfigManager.readSamplerType() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateSamplerType(value)
    }

    private func updateTemperature(_ value: Double) {
        if viewModel.modelConfigManager.readTemperature() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateTemperature(value)
    }

    private func updateTopK(_ value: Int) {
        if viewModel.modelConfigManager.readTopK() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateTopK(value)
    }

    private func updateTopP(_ value: Double) {
        if viewModel.modelConfigManager.readTopP() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateTopP(value)
    }

    private func updateMinP(_ value: Double) {
        if viewModel.modelConfigManager.readMinP() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateMinP(value)
    }

    private func updateTfsZ(_ value: Double) {
        if viewModel.modelConfigManager.readTfsZ() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateTfsZ(value)
    }

    private func updateTypical(_ value: Double) {
        if viewModel.modelConfigManager.readTypical() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateTypical(value)
    }

    private func updatePenalty(_ value: Double) {
        // Values up to 1.0 all mean "no penalty"; clamping a legacy 0.x on
        // open must not ask for a reload.
        let floor = SamplerType.penaltyRange.lowerBound
        if max(viewModel.modelConfigManager.readPenalty(), floor) != max(value, floor) {
            requiresReload = true
        }
        viewModel.modelConfigManager.updatePenalty(value)
    }

    private func updateNGram(_ value: Int) {
        if viewModel.modelConfigManager.readNGram() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateNGram(value)
    }

    private func updateNGramFactor(_ value: Double) {
        if viewModel.modelConfigManager.readNGramFactor() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updateNGramFactor(value)
    }

    private func updatePenaltySampler(_ value: PenaltySamplerType) {
        if viewModel.modelConfigManager.readPenaltySampler() != value {
            requiresReload = true
        }
        viewModel.modelConfigManager.updatePenaltySampler(value)
    }
}
