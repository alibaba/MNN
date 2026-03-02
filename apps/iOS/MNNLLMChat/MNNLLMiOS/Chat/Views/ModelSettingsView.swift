//
//  ModelSettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/17.
//

import SwiftUI

struct ModelSettingsView: View {
    @Binding var showSettings: Bool
    @ObservedObject var viewModel: LLMChatViewModel
    @State private var showAlert = false
    @State private var showReloadAlert = false
    @State private var iterations: Int = 20
    @State private var seed: Int = -1
    @State private var useRandomSeed: Bool = true
    @State private var backendType: String = "cpu"
    @State private var precision: String = "low"
    @State private var threadNum: Int = 4
    @State private var requiresReload = false

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

    private let backendOptions = ["cpu", "opencl"]
    private let precisionOptions = ["low", "high"]

    var body: some View {
        NavigationView {
            Form {
                Section {
                    Toggle("Use mmap", isOn: $viewModel.useMmap)
                        .onChange(of: viewModel.useMmap) { newValue in
                            viewModel.modelConfigManager.updateUseMmap(newValue)
                        }

                    Button("Clear mmap Cache") {
                        viewModel.cleanModelTmpFolder()
                        showAlert = true
                    }

                    Picker("Backend", selection: $backendType) {
                        ForEach(backendOptions, id: \.self) { backend in
                            Text(backend.uppercased()).tag(backend)
                        }
                    }
                    .onChange(of: backendType) { _, newValue in
                        requiresReload = true
                        viewModel.modelConfigManager.updateBackendType(newValue)
                    }

                    Picker("Precision", selection: $precision) {
                        ForEach(precisionOptions, id: \.self) { option in
                            Text(option.capitalized).tag(option)
                        }
                    }
                    .onChange(of: precision) { _, newValue in
                        requiresReload = true
                        viewModel.modelConfigManager.updatePrecision(newValue)
                    }

                    Stepper(value: $threadNum, in: 1 ... viewModel.modelConfigManager.maxThreads) {
                        HStack {
                            Text("Threads")
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

                Section {
                    Stepper(value: $videoMaxFrames, in: 1 ... 32) {
                        HStack {
                            Text("Video Frames")
                            Spacer()
                            Text("\(videoMaxFrames)")
                        }
                    }
                    .onChange(of: videoMaxFrames) { _, newValue in
                        viewModel.updateVideoMaxFrames(newValue)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Default Multimodal Prompt")
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

                // Diffusion Settings
                if viewModel.isDiffusionModel {
                    Section {
                        Stepper(value: $iterations, in: 1 ... 100) {
                            HStack {
                                Text("Iterations")
                                Spacer()
                                Text("\(iterations)")
                            }
                        }
                        .onChange(of: iterations) { _, newValue in
                            viewModel.modelConfigManager.updateIterations(newValue)
                        }

                        Toggle("Random Seed", isOn: $useRandomSeed)
                            .onChange(of: useRandomSeed) { _, newValue in
                                if newValue {
                                    seed = -1
                                    viewModel.modelConfigManager.updateSeed(-1)
                                }
                            }

                        if !useRandomSeed {
                            HStack {
                                Text("Seed")
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
                        Text("Diffusion Settings")
                    }
                } else {
                    Section {
                        Picker("Sampler Type", selection: $selectedSampler) {
                            ForEach(SamplerType.allCases, id: \.self) { sampler in
                                Text(sampler.displayName)
                                    .tag(sampler)
                            }
                        }
                        .onChange(of: selectedSampler) { _, newValue in
                            viewModel.modelConfigManager.updateSamplerType(newValue)
                        }

                        switch selectedSampler {
                        case .temperature:
                            ParameterSliderView(
                                title: "Temperature",
                                value: $temperature,
                                range: 0.0 ... 2.0,
                                format: "%.2f",
                                intValue: false,
                                onChanged: viewModel.modelConfigManager.updateTemperature(_:)
                            )
                        case .topK:
                            ParameterSliderView(
                                title: "Top K",
                                value: $topK,
                                range: 1 ... 100,
                                format: "%.0f",
                                intValue: true,
                                onChanged: { viewModel.modelConfigManager.updateTopK(Int($0)) }
                            )
                        case .topP:
                            ParameterSliderView(
                                title: "Top P",
                                value: $topP,
                                range: 0.0 ... 1.0,
                                format: "%.2f",
                                intValue: false,
                                onChanged: viewModel.modelConfigManager.updateTopP(_:)
                            )
                        case .minP:
                            ParameterSliderView(
                                title: "Min P",
                                value: $minP,
                                range: 0.05 ... 0.3,
                                format: "%.2f",
                                intValue: false,
                                onChanged: viewModel.modelConfigManager.updateMinP(_:)
                            )
                        case .tfs:
                            ParameterSliderView(
                                title: "TFS-Z",
                                value: $tfsZ,
                                range: 0.9 ... 0.99,
                                format: "%.2f",
                                intValue: false,
                                onChanged: viewModel.modelConfigManager.updateTfsZ(_:)
                            )
                        case .typical:
                            ParameterSliderView(
                                title: "Typical",
                                value: $typical,
                                range: 0.8 ... 0.95,
                                format: "%.2f",
                                intValue: false,
                                onChanged: viewModel.modelConfigManager.updateTypical(_:)
                            )
                        case .penalty:
                            VStack(spacing: 8) {
                                // Penalty parameters
                                ParameterSliderView(
                                    title: "Penalty",
                                    value: $penalty,
                                    range: 0.0 ... 0.5,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: viewModel.modelConfigManager.updatePenalty(_:)
                                )

                                // N-gram Size parameters
                                ParameterSliderView(
                                    title: "N-gram Size",
                                    value: $nGram,
                                    range: 3 ... 8,
                                    format: "%.0f",
                                    intValue: true,
                                    onChanged: { viewModel.modelConfigManager.updateNGram(Int($0)) }
                                )

                                // N-gram Factor parameters
                                ParameterSliderView(
                                    title: "N-gram Factor",
                                    value: $nGramFactor,
                                    range: 1.0 ... 3.0,
                                    format: "%.1f",
                                    intValue: false,
                                    onChanged: viewModel.modelConfigManager.updateNGramFactor(_:)
                                )

                                // Penalty Sampler selector
                                Picker("Penalty Sampler", selection: $penaltySampler) {
                                    ForEach(PenaltySamplerType.allCases, id: \.self) { samplerType in
                                        Text(samplerType.displayName)
                                            .tag(samplerType)
                                    }
                                }
                                .onChange(of: penaltySampler) { newValue in
                                    viewModel.modelConfigManager.updatePenaltySampler(newValue)
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
                                onUpdateTemperature: viewModel.modelConfigManager.updateTemperature(_:),
                                onUpdateTopK: viewModel.modelConfigManager.updateTopK(_:),
                                onUpdateTopP: viewModel.modelConfigManager.updateTopP(_:),
                                onUpdateMinP: viewModel.modelConfigManager.updateMinP(_:),
                                onUpdateTfsZ: viewModel.modelConfigManager.updateTfsZ(_:),
                                onUpdateTypical: viewModel.modelConfigManager.updateTypical(_:),
                                onUpdatePenalty: viewModel.modelConfigManager.updatePenalty(_:)
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
        .alert(NSLocalizedString("Success", comment: ""), isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(NSLocalizedString("Cache Cleared Successfully", comment: ""))
        }
        .alert(NSLocalizedString("Model Reload Required", comment: ""), isPresented: $showReloadAlert) {
            Button(NSLocalizedString("Reload Now", comment: "")) {
                viewModel.reloadCurrentModel()
                requiresReload = false
                showSettings = false
            }
            Button(NSLocalizedString("Later", comment: ""), role: .cancel) {
                showSettings = false
            }
        } message: {
            Text(NSLocalizedString("Changes to backend, precision, or thread count require the model to reload.", comment: ""))
        }
        .onAppear {
            selectedSampler = viewModel.modelConfigManager.readSamplerType()
            backendType = viewModel.modelConfigManager.readBackendType()
            precision = viewModel.modelConfigManager.readPrecision()
            threadNum = viewModel.modelConfigManager.readThreadNum()
            requiresReload = false

            if viewModel.isDiffusionModel {
                iterations = viewModel.modelConfigManager.readIterations()
                seed = viewModel.modelConfigManager.readSeed()
                useRandomSeed = (seed < 0)
            } else {
                temperature = viewModel.modelConfigManager.readTemperature()
                topK = Double(viewModel.modelConfigManager.readTopK())
                topP = viewModel.modelConfigManager.readTopP()
                minP = viewModel.modelConfigManager.readMinP()
                tfsZ = viewModel.modelConfigManager.readTfsZ()
                typical = viewModel.modelConfigManager.readTypical()
                penalty = viewModel.modelConfigManager.readPenalty()
                nGram = Double(viewModel.modelConfigManager.readNGram())
                nGramFactor = viewModel.modelConfigManager.readNGramFactor()
            }
            videoMaxFrames = viewModel.modelConfigManager.readVideoMaxFrames()
            defaultMultimodalPrompt = viewModel.modelConfigManager.readDefaultMultimodalPrompt()

            // Initialize mixed samplers
            let savedMixedSamplers = viewModel.modelConfigManager.readMixedSamplers()
            mixedSamplersOrder = ["topK", "tfs", "typical", "topP", "minP", "temperature"]
            selectedMixedSamplers = Set(savedMixedSamplers)

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
        viewModel.modelConfigManager.updateMixedSamplers(orderedSelection)
    }
}
