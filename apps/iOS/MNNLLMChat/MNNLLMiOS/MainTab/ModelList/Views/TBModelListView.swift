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
        NavigationView {
            VStack {
                SearchBar(text: $searchText)
                    .onChange(of: searchText) { _, newValue in
                        viewModel.searchText = newValue
                    }
                
                List {
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
                        .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                        .listRowSeparator(.hidden)
                    }
                }
                .listStyle(PlainListStyle())
                .refreshable {
                    await viewModel.fetchModels()
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .alert("错误", isPresented: $viewModel.showError) {
                Button("确定") { }
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }
}

#Preview {
    TBModelListView()
}
