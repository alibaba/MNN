//
//  MNNLLMiOSApp.swift
//  LocalModelListView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct LocalModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    @State private var localSearchText = ""
    
    private var filteredLocalModels: [ModelInfo] {
        let downloadedModels = viewModel.models.filter { $0.isDownloaded }
        
        if localSearchText.isEmpty {
            return downloadedModels
        } else {
            return downloadedModels.filter { model in
                model.id.localizedCaseInsensitiveContains(localSearchText) ||
                model.modelName.localizedCaseInsensitiveContains(localSearchText) ||
                model.localizedTags.contains { $0.localizedCaseInsensitiveContains(localSearchText) }
            }
        }
    }
    
    var body: some View {
        List {
            ForEach(filteredLocalModels, id: \.id) { model in
                Button(action: {
                    viewModel.selectModel(model)
                }) {
                    LocalModelRowView(model: model)
                }
                .listRowBackground(viewModel.pinnedModelIds.contains(model.id) ? Color.black.opacity(0.05) : Color.clear)
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    SwipeActionsView(model: model, viewModel: viewModel)
                }
            }
        }
        .listStyle(.plain)
        .searchable(text: $localSearchText, prompt: "搜索本地模型...")
        .refreshable {
            await viewModel.fetchModels()
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(viewModel.errorMessage)
        }
    }
}
