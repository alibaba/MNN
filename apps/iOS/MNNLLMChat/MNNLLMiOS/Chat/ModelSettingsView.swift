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
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model Configuration")) {
                    Toggle("Use mmap", isOn: Binding(
                        get: { viewModel.useMmap },
                        set: { viewModel.updateUseMmap($0) }
                    ))
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
                        .onChange(of: iterations) { newValue in
                            viewModel.updateIterations(newValue)
                        }
                        
                        Toggle("Random Seed", isOn: $useRandomSeed)
                            .onChange(of: useRandomSeed) { newValue in
                                if newValue {
                                    seed = -1
                                    viewModel.updateSeed(-1)
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
                                    .onChange(of: seed) { newValue in
                                        viewModel.updateSeed(max(0, newValue))
                                    }
                            }
                        }
                    }
                }
                
                Section {
                    Button("Clear mmap Cache") {
                        viewModel.cleanModelTmpFolder()
                        showAlert = true
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarItems(trailing: Button("Done") {
                showSettings = false
            })
            .alert(NSLocalizedString("Success", comment: ""), isPresented: $showAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(NSLocalizedString("Cache Cleared Successfully", comment: ""))
            }
            .onAppear {
                if viewModel.isDiffusionModel {
                    iterations = viewModel.iterations
                    seed = viewModel.seed
                    useRandomSeed = (seed < 0)
                }
            }
        }
    }
}

