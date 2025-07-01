//
//  MNNLLMiOSApp.swift
//  LocalModelListView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct LocalModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    
    var body: some View {
        List {
            ForEach(viewModel.filteredModels.filter { $0.isDownloaded }, id: \.modelId) { model in
                Button(action: {
                    viewModel.selectModel(model)
                }) {
                    LocalModelRowView(model: model)
                }
                .listRowSeparator(.hidden)
                .listRowBackground(viewModel.pinnedModelIds.contains(model.modelId) ? Color.black.opacity(0.05) : Color.clear)
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    SwipeActionsView(model: model, viewModel: viewModel)
                }
            }
        }
        .listStyle(.plain)
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
