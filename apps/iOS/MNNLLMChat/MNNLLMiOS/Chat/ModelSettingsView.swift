//
//  ModelSettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/17.
//

import SwiftUI

struct ModelSettingsView: View {
    
    @Binding var showSettings: Bool
    @State private var useMmap: Bool = false
    @ObservedObject var viewModel: LLMChatViewModel
    @State private var showAlert = false
    @State private var iterations: Int = 20
    @State private var seed: Int = -1
    @State private var useRandomSeed: Bool = true
    
    @State private var temperature: Double = 1.0
    @State private var topK: Double = 40
    @State private var topP: Double = 0.9
    @State private var minP: Double = 0.1
    
    @State private var tfsZ: Double = 1.0
    @State private var typical: Double = 1.0
    @State private var penalty: Double = 0.0
    @State private var nGram: Double = 8.0
    @State private var nGramFactor: Double = 1.0
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model Configuration")) {
                    Toggle("Use mmap", isOn: $viewModel.useMmap)
                        .onChange(of: viewModel.useMmap) { newValue in
                            viewModel.modelConfigManager.updateUseMmap(newValue)
                        }
                    
                    Button("Clear mmap Cache") {
                        viewModel.cleanModelTmpFolder()
                        showAlert = true
                    }
                }
                
                // Diffusion Settings
                if viewModel.isDiffusionModel {
                    Section(header: Text("Diffusion Settings")) {
                        Stepper(value: $iterations, in: 1...100) {
                            HStack {
                                Text("Iterations")
                                Spacer()
                                Text("\(iterations)")
                            }
                        }
                        .onChange(of: iterations) { oldValue, newValue in
                            viewModel.modelConfigManager.updateIterations(newValue)
                        }
                        
                        Toggle("Random Seed", isOn: $useRandomSeed)
                            .onChange(of: useRandomSeed) { oldValue, newValue in
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
                                    .onChange(of: seed) { oldValue, newValue in
                                        viewModel.modelConfigManager.updateSeed(max(0, newValue))
                                    }
                            }
                        }
                    }
                } else {
                    Section(header: Text("Generation Parameters")) {
                        ParameterSliderView(
                            title: "Temperature",
                            value: $temperature,
                            range: 0.0...2.0,
                            format: "%.2f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updateTemperature(_:)
                        )
                        
                        ParameterSliderView(
                            title: "Top K",
                            value: $topK,
                            range: 1...100,
                            format: "%.0f",
                            intValue: true,
                            onChanged: { viewModel.modelConfigManager.updateTopP(Double($0)) }
                        )
                        
                        ParameterSliderView(
                            title: "Top P",
                            value: $topP,
                            range: 0.0...1.0,
                            format: "%.2f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updateTopP(_:)
                        )
                        
                        ParameterSliderView(
                            title: "Min P",
                            value: $minP,
                            range: 0.05...0.3,
                            format: "%.2f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updateMinP(_:)
                        )
                        
                        ParameterSliderView(
                            title: "TFS-Z",
                            value: $tfsZ,
                            range: 0.9...0.99,
                            format: "%.2f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updateTfsZ(_:)
                        )
                        
                        
                        ParameterSliderView(
                            title: "Typical",
                            value: $typical,
                            range: 0.8...0.95,
                            format: "%.2f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updateTypical(_:)
                        )
                        
                        ParameterSliderView(
                            title: "Penalty",
                            value: $penalty,
                            range: 0.0...0.5,
                            format: "%.2f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updatePenalty(_:)
                        )
                        
                        ParameterSliderView(
                            title: "N-gram",
                            value: $nGram,
                            range: 3...8,
                            format: "%.0f",
                            intValue: true,
                            onChanged: { viewModel.modelConfigManager.updateNGram(Int($0)) }
                        )
                        
                        ParameterSliderView(
                            title: "N-gram Factor",
                            value: $nGramFactor,
                            range: 1.0...3.0,
                            format: "%.1f",
                            intValue: false,
                            onChanged: viewModel.modelConfigManager.updateNGramFactor(_:)
                        )
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        showSettings = false
                    }
            }
       }
        }
        .alert(NSLocalizedString("Success", comment: ""), isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(NSLocalizedString("Cache Cleared Successfully", comment: ""))
        }
        .onAppear {
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
        }
    }
}


