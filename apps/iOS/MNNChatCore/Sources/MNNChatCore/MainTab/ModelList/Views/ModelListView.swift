//
//  ModelListView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

struct ModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    var activeModelName: String? = nil
    @ObservedObject private var nameStore = ModelNameStore.shared
    @State private var searchText = ""
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    @State private var showSourceMenu = false
    @State private var selectedTags: Set<String> = []
    @State private var selectedCategories: Set<String> = []
    @State private var selectedVendors: Set<String> = []
    @State private var showFilterMenu = false

    var body: some View {
        ScrollViewReader { proxy in
            // A List (not a LazyVStack) so rows get native swipe actions.
            List {
                Section {
                    ForEach(filteredModels, id: \.id) { model in
                        modelRowView(model: model)
                            .listRowInsets(EdgeInsets(top: 0, leading: 16, bottom: 0, trailing: 16))
                    }
                } header: {
                    toolbarSection
                        .listRowInsets(EdgeInsets())
                        .textCase(nil)
                }
            }
            .listStyle(.plain)
            .overlay {
                if filteredModels.isEmpty {
                    if viewModel.isLoadingModels {
                        ProgressView(AppText.text("onboarding.loading"))
                    } else if !searchText.isEmpty {
                        ContentUnavailableView.search(text: searchText)
                    } else {
                        ContentUnavailableView {
                            Label(AppText.text("workspace.catalogEmpty"), systemImage: "cube")
                        } description: {
                            Text(AppText.text("workspace.catalogRetry"))
                        } actions: {
                            Button(AppText.text("workspace.retry")) { Task { await viewModel.fetchModels() } }
                                .buttonStyle(.bordered)
                        }.padding(.top, 60)
                    }
                }
            }
            .searchable(text: $searchText, prompt: "Search models...")
            .refreshable {
                await viewModel.fetchModels()
            }
            // Auto-scroll to top when filters change to avoid blank screen when data shrinks
            .onChange(of: selectedTags) { _, _ in scrollToTop(proxy) }
            .onChange(of: selectedCategories) { _, _ in scrollToTop(proxy) }
            .onChange(of: selectedVendors) { _, _ in scrollToTop(proxy) }
            .onChange(of: showFilterMenu) { _, _ in scrollToTop(proxy) }
        }
    }

    private func scrollToTop(_ proxy: ScrollViewProxy) {
        guard let first = filteredModels.first else { return }
        withAnimation { proxy.scrollTo(first.id, anchor: .top) }
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

    @ViewBuilder
    private func modelRowView(model: ModelInfo) -> some View {
        ModelRowView(
            model: model,
            viewModel: viewModel,
            isOtherDownloading: isOtherDownloadingCheck(model: model),
            isInUse: model.isDownloaded && model.modelName == activeModelName
        )
    }

    // Extract complex boolean logic as independent method
    private func isOtherDownloadingCheck(model: ModelInfo) -> Bool {
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

    private var filteredModels: [ModelInfo] {
        let searchFiltered = searchText.isEmpty ? viewModel.models : viewModel.models.filter { model in
            model.id.localizedCaseInsensitiveContains(searchText) ||
                model.name.localizedCaseInsensitiveContains(searchText) ||
                model.localizedTags.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }

        let tagFiltered: [ModelInfo]
        if selectedTags.isEmpty, selectedCategories.isEmpty, selectedVendors.isEmpty {
            tagFiltered = searchFiltered
        } else {
            tagFiltered = searchFiltered.filter { model in
                let tagMatch = checkTagMatch(model: model)
                let categoryMatch = checkCategoryMatch(model: model)
                let vendorMatch = checkVendorMatch(model: model)

                return tagMatch && categoryMatch && vendorMatch
            }
        }

        let downloaded = tagFiltered.filter { $0.isDownloaded }
        let notDownloaded = tagFiltered.filter { !$0.isDownloaded }

        return downloaded + notDownloaded
    }

    // Extract tag matching logic as independent method
    private func checkTagMatch(model: ModelInfo) -> Bool {
        return selectedTags.isEmpty || selectedTags.allSatisfy { selectedTag in
            model.localizedTags.contains { tag in
                tag.localizedCaseInsensitiveContains(selectedTag)
            }
        }
    }

    // Extract category matching logic as independent method
    private func checkCategoryMatch(model: ModelInfo) -> Bool {
        return selectedCategories.isEmpty || selectedCategories.allSatisfy { selectedCategory in
            model.categories?.contains { category in
                category.localizedCaseInsensitiveContains(selectedCategory)
            } ?? false
        }
    }

    // Extract vendor matching logic as independent method
    private func checkVendorMatch(model: ModelInfo) -> Bool {
        return selectedVendors.isEmpty || selectedVendors.contains { selectedVendor in
            model.vendor?.localizedCaseInsensitiveContains(selectedVendor) ?? false
        }
    }
}
