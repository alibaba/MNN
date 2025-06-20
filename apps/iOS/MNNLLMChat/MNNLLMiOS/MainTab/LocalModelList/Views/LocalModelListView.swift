//
//  MNNLLMiOSApp.swift
//  LocalModelListView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct LocalModelListView: View {
    @StateObject private var viewModel = ModelListViewModel()
    
    var body: some View {
        NavigationView {
            List {
                ForEach(viewModel.filteredModels.filter { $0.isDownloaded }, id: \ .modelId) { model in
                    ModelRowView(model: model,
                                 downloadProgress: viewModel.downloadProgress[model.modelId] ?? 0,
                                 isDownloading: viewModel.currentlyDownloading == model.modelId,
                                 isOtherDownloading: viewModel.currentlyDownloading != nil) {
                        viewModel.selectModel(model)
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button(role: .destructive) {
                            Task {
                                await viewModel.deleteModel(model)
                            }
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
            .listStyle(.plain)
            .navigationTitle("本地模型")
            .navigationBarTitleDisplayMode(.large)
            .refreshable {
                await viewModel.fetchModels()
            }
            .alert("Error", isPresented: $viewModel.showError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage)
            }
            .background(
                NavigationLink(
                    destination: {
                        if let selectedModel = viewModel.selectedModel {
                            return AnyView(LLMChatView(modelInfo: selectedModel))
                        }
                        return AnyView(EmptyView())
                    }(),
                    isActive: Binding(
                        get: { viewModel.selectedModel != nil },
                        set: { if !$0 { viewModel.selectedModel = nil } }
                    )
                ) {
                    EmptyView()
                }
            )
        }
    }
} 
