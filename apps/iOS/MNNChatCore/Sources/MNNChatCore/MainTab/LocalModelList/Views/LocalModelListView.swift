//
//  LocalModelListView.swift
//  LocalModelListView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct LocalModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    var browseModels: () -> Void = {}
    var activeModelName: String? = nil
    var allowsSelection = true
    @ObservedObject private var nameStore = ModelNameStore.shared
    @State private var localSearchText = ""
    @State private var isShowingRenameDialog = false
    @State private var modelBeingRenamed: ModelInfo?
    @State private var renameText = ""
    @State private var deletionRequest: ModelDeletionRequest?

    private var filteredLocalModels: [ModelInfo] {
        // Filter for truly local models (vendor is "Local") or downloaded models
        let localModels = viewModel.models.filter { model in
            // Check if it's a built-in local model or a downloaded remote model
            if let vendor = model.vendor, vendor.lowercased() == "local" {
                return true // Built-in local model
            }
            return model.isDownloaded // Downloaded remote model
        }

        if localSearchText.isEmpty {
            return localModels
        } else {
            return localModels.filter { model in
                model.id.localizedCaseInsensitiveContains(localSearchText) ||
                    model.name.localizedCaseInsensitiveContains(localSearchText) ||
                    model.localizedTags.contains { $0.localizedCaseInsensitiveContains(localSearchText) }
            }
        }
    }

    var body: some View {
        List {
            if filteredLocalModels.isEmpty {
                if viewModel.isLoadingModels {
                    HStack { Spacer(); ProgressView("onboarding.loading"); Spacer() }
                        .listRowSeparator(.hidden)
                } else if localSearchText.isEmpty {
                    FirstModelGuide(viewModel: viewModel, browseModels: browseModels)
                        .listRowSeparator(.hidden)
                } else {
                    ContentUnavailableView.search(text: localSearchText)
                        .listRowSeparator(.hidden)
                }
            }
            ForEach(filteredLocalModels, id: \.id) { model in
                Button(action: {
                    viewModel.selectModel(model)
                }) {
                    LocalModelRowView(model: model)
                        .contentShape(Rectangle())
                }
                .buttonStyle(PlainButtonStyle())
                .disabled(!allowsSelection)
                .listRowBackground(viewModel.pinnedModelIds.contains(model.id) ? Color.black.opacity(0.05) : Color.clear)
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    SwipeActionsView(model: model, viewModel: viewModel, onRename: {
                        modelBeingRenamed = model
                        renameText = model.name
                        isShowingRenameDialog = true
                    }, onDelete: {
                        deletionRequest = .make(for: model, isInUse: model.modelName == activeModelName)
                    })
                }
            }
        }
        .listStyle(.plain)
        .searchable(text: $localSearchText, prompt: "Search models...")
        .refreshable {
            await viewModel.fetchModels()
        }
        .modelDeletionAlert($deletionRequest) { model in
            Task { await viewModel.deleteModel(model) }
        }
        .alert(
            AppText.text("Rename Model"),
            isPresented: $isShowingRenameDialog
        ) {
            TextField(AppText.text("Model Name"), text: $renameText)
            Button(AppText.text("Cancel"), role: .cancel) {
                modelBeingRenamed = nil
            }
            Button(AppText.text("Save")) {
                if let modelBeingRenamed {
                    viewModel.renameModel(modelBeingRenamed, to: renameText)
                }
                modelBeingRenamed = nil
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }
}
