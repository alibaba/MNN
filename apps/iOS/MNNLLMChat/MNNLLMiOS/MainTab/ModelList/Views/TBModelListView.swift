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
                    LazyVStack(spacing: 8) {
                        ForEach(Array(filteredModels.enumerated()), id: \.element.id) { index, model in
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
                            
                            if index < filteredModels.count - 1 {
                                Divider()
                                    .padding(.horizontal, 16)
                            }
                        }
                    }
                    .padding(.vertical, 8)
                } header: {
                    ToolbarView(
                        selectedSource: $selectedSource,
                        showSourceMenu: $showSourceMenu,
                        selectedTags: $selectedTags,
                        selectedCategories: $selectedCategories,
                        selectedVendors: $selectedVendors,
                        quickFilterTags: viewModel.quickFilterTags,
                        showFilterMenu: $showFilterMenu,
                        onSourceChange: { source in
                            ModelSourceManager.shared.updateSelectedSource(source)
                            selectedSource = source
                            Task {
                                await viewModel.fetchModels()
                            }
                        }
                    )
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
        .alert("错误", isPresented: $viewModel.showError) {
            Button("确定") { }
        } message: {
            Text(viewModel.errorMessage)
        }
    }
    
    // 根据选中的标签、分类和厂商筛选模型
    private var filteredModels: [TBModelInfo] {
        let baseFiltered = viewModel.filteredModels
        
        if selectedTags.isEmpty && selectedCategories.isEmpty && selectedVendors.isEmpty {
            return baseFiltered
        }
        
        return baseFiltered.filter { model in
            let tagMatch = selectedTags.isEmpty || selectedTags.allSatisfy { selectedTag in
                model.localizedTags.contains { tag in
                    tag.localizedCaseInsensitiveContains(selectedTag)
                }
            }
            
            let categoryMatch = selectedCategories.isEmpty || selectedCategories.allSatisfy { selectedCategory in
                model.categories?.contains { category in
                    category.localizedCaseInsensitiveContains(selectedCategory)
                } ?? false
            }
            
            let vendorMatch = selectedVendors.isEmpty || selectedVendors.contains { selectedVendor in
                model.vendor?.localizedCaseInsensitiveContains(selectedVendor) ?? false
            }
            
            return tagMatch && categoryMatch && vendorMatch
        }
    }
}

#Preview {
    TBModelListView()
}