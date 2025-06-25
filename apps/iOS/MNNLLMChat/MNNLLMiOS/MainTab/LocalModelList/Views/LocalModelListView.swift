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
                ModelRowView(model: model,
                             downloadProgress: viewModel.downloadProgress[model.modelId] ?? 0,
                             isDownloading: viewModel.currentlyDownloading == model.modelId,
                             isOtherDownloading: viewModel.currentlyDownloading != nil) {
                    viewModel.selectModel(model)
                }
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
