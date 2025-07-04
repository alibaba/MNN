//
//  TBModelListView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

struct TBModelListView: View {
    @StateObject private var viewModel = TBModelListViewModel()
    @State private var searchText = ""
    
    var body: some View {

        ScrollView {
            LazyVStack(spacing: 8) {
                ForEach(Array(viewModel.filteredModels.enumerated()), id: \.element.id) { index, model in
                    TBModelRowView(
                        model: model,
                        viewModel: viewModel,
                        downloadProgress: viewModel.downloadProgress[model.id] ?? 0,
                        isDownloading: viewModel.currentlyDownloading == model.id,
                        isOtherDownloading: viewModel.currentlyDownloading != nil && viewModel.currentlyDownloading != model.id
                    ) {
                        Task {
                            await viewModel.downloadModel(model)
                        }
                    }
                    .padding(.horizontal, 16)
                    
                    if index < viewModel.filteredModels.count - 1 {
                        Divider()
                            .padding(.horizontal, 16)
                    }
                }
            }
            .padding(.vertical, 8)
        }
        .searchable(text: $searchText, prompt: "Search models...")
        .onChange(of: searchText) { _, newValue in
            viewModel.searchText = newValue
        }
        .refreshable {
            await viewModel.fetchModels()
        }
        .alert("错误", isPresented: $viewModel.showError) {
            Button("确定") { }
        } message: {
            Text(viewModel.errorMessage)
        }
    }
}

#Preview {
    TBModelListView()
}
