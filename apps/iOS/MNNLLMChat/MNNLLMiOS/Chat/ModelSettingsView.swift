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

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model Configuration")) {
                    Toggle("Use mmap", isOn: Binding(
                        get: { viewModel.useMmap },
                        set: { viewModel.updateUseMmap($0) }
                    ))
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
        }
    }
}

