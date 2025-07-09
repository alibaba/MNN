//
//  TBModelListView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

struct TBModelListView: View {
    @ObservedObject var viewModel: TBModelListViewModel
    @State private var searchText = ""
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    @State private var showSourceMenu = false
    @State private var selectedTags: Set<String> = []
    @State private var selectedCategories: Set<String> = []
    @State private var selectedVendors: Set<String> = []
    @State private var showFilterMenu = false
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                Section {
                    modelListSection
                } header: {
                    toolbarSection
                }
            }
        }
        .searchable(text: $searchText, prompt: "Search models...")
        .onChange(of: searchText) { _, newValue in
            viewModel.searchText = newValue
        }
        .refreshable {
            await viewModel.fetchModels()
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK") { }
        } message: {
            Text(viewModel.errorMessage)
        }
    }
    
    // Extract model list section as independent view
    @ViewBuilder
    private var modelListSection: some View {
        LazyVStack(spacing: 8) {
            ForEach(Array(filteredModels.enumerated()), id: \.element.id) { index, model in
                modelRowView(model: model, index: index)
                
                if index < filteredModels.count - 1 {
                    Divider()
                        .padding(.horizontal, 16)
                }
            }
        }
        .padding(.vertical, 8)
    }
    
    // Extract toolbar section as independent view
    @ViewBuilder
    private var toolbarSection: some View {
        ToolbarView(
            viewModel: viewModel, selectedSource: $selectedSource,
            showSourceMenu: $showSourceMenu,
            selectedTags: $selectedTags,
            selectedCategories: $selectedCategories,
            selectedVendors: $selectedVendors,
            quickFilterTags: viewModel.quickFilterTags,
            showFilterMenu: $showFilterMenu,
            onSourceChange: handleSourceChange
        )
    }
    
    // Extract single model row view as independent method
    @ViewBuilder
    private func modelRowView(model: TBModelInfo, index: Int) -> some View {
        TBModelRowView(
            model: model,
            viewModel: viewModel,
            downloadProgress: viewModel.downloadProgress[model.id] ?? 0,
            isDownloading: viewModel.currentlyDownloading == model.id,
            isOtherDownloading: isOtherDownloadingCheck(model: model)
        ) {
            Task {
                await viewModel.downloadModel(model)
            }
        }
        .padding(.horizontal, 16)
    }
    
    // Extract complex boolean logic as independent method
    private func isOtherDownloadingCheck(model: TBModelInfo) -> Bool {
        return viewModel.currentlyDownloading != nil && viewModel.currentlyDownloading != model.id
    }
    
    // Extract source change handling logic as independent method
    private func handleSourceChange(_ source: ModelSource) {
        ModelSourceManager.shared.updateSelectedSource(source)
        selectedSource = source
        Task {
            await viewModel.fetchModels()
        }
    }
    
    // Filter models based on selected tags, categories and vendors
    private var filteredModels: [TBModelInfo] {
        let baseFiltered = viewModel.filteredModels
        
        if selectedTags.isEmpty && selectedCategories.isEmpty && selectedVendors.isEmpty {
            return baseFiltered
        }
        
        return baseFiltered.filter { model in
            let tagMatch = checkTagMatch(model: model)
            let categoryMatch = checkCategoryMatch(model: model)
            let vendorMatch = checkVendorMatch(model: model)
            
            return tagMatch && categoryMatch && vendorMatch
        }
    }
    
    // Extract tag matching logic as independent method
    private func checkTagMatch(model: TBModelInfo) -> Bool {
        return selectedTags.isEmpty || selectedTags.allSatisfy { selectedTag in
            model.localizedTags.contains { tag in
                tag.localizedCaseInsensitiveContains(selectedTag)
            }
        }
    }
    
    // Extract category matching logic as independent method
    private func checkCategoryMatch(model: TBModelInfo) -> Bool {
        return selectedCategories.isEmpty || selectedCategories.allSatisfy { selectedCategory in
            model.categories?.contains { category in
                category.localizedCaseInsensitiveContains(selectedCategory)
            } ?? false
        }
    }
    
    // Extract vendor matching logic as independent method
    private func checkVendorMatch(model: TBModelInfo) -> Bool {
        return selectedVendors.isEmpty || selectedVendors.contains { selectedVendor in
            model.vendor?.localizedCaseInsensitiveContains(selectedVendor) ?? false
        }
    }
}
