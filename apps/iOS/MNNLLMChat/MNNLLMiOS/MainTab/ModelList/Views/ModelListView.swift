//
//  ModelListView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct ModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    
    @State private var scrollOffset: CGFloat = 0
    @State private var showHelp = false
    @State private var showUserGuide = false
    
    var body: some View {
        List {
            SearchBar(text: $viewModel.searchText)
                .listRowInsets(EdgeInsets())
                .listRowSeparator(.hidden)
                .padding(.horizontal)
                
            ForEach(viewModel.filteredModels, id: \.modelId) { model in
                ModelRowView(model: model,
                           downloadProgress: viewModel.downloadProgress[model.modelId] ?? 0,
                           isDownloading: viewModel.currentlyDownloading == model.modelId,
                           isOtherDownloading: viewModel.currentlyDownloading != nil) {
                    if model.isDownloaded {
                        viewModel.selectModel(model)
                    } else {
                        Task {
                            await viewModel.downloadModel(model)
                        }
                    }
                }
               .listRowBackground(viewModel.pinnedModelIds.contains(model.modelId) ? Color.black.opacity(0.05) : Color.clear)
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    SwipeActionsView(model: model, viewModel: viewModel)
                }
            }
        }
        .listStyle(.plain)
        .sheet(isPresented: $showHelp) {
            HelpView()
        }
        .refreshable {
            await viewModel.fetchModels()
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(viewModel.errorMessage)
        }
        .onAppear {
            checkFirstLaunch()
        }
        .alert(isPresented: $showUserGuide) {
            Alert(
                title: Text("User Guide"),
                message: Text("""
                This is a local large model application that requires certain performance from your device.
                It is recommended to choose different model sizes based on your device's memory. 
                
                The model recommendations for iPhone are as follows:
                - For 8GB of RAM, models up to 8B are recommended (e.g., iPhone 16 Pro).
                - For 6GB of RAM, models up to 3B are recommended (e.g., iPhone 15 Pro).
                - For 4GB of RAM, models up to 1B or smaller are recommended (e.g., iPhone 13).
                
                Choosing a model that is too large may cause insufficient memory and crashes.
                """),
                dismissButton: .default(Text("OK"))
            )
        }
    }
    
    private func checkFirstLaunch() {
        let hasLaunchedBefore = UserDefaults.standard.bool(forKey: "hasLaunchedBefore")
        if !hasLaunchedBefore {
            // Show the user guide alert
            showUserGuide = true
            // Set the flag to true so it doesn't show again
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")
        }
    }
}
