//
//  LocalModelListView.swift
//  LocalModelListView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct LocalModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    @ObservedObject private var nameStore = ModelNameStore.shared
    @State private var localSearchText = ""
    @State private var isShowingRenameDialog = false
    @State private var modelBeingRenamed: ModelInfo?
    @State private var renameText = ""

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
            ForEach(filteredLocalModels, id: \.id) { model in
                Button(action: {
                    viewModel.selectModel(model)
                }) {
                    LocalModelRowView(model: model)
                        .contentShape(Rectangle())
                }
                .buttonStyle(PlainButtonStyle())
                .listRowBackground(viewModel.pinnedModelIds.contains(model.id) ? Color.black.opacity(0.05) : Color.clear)
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    SwipeActionsView(model: model, viewModel: viewModel) {
                        modelBeingRenamed = model
                        renameText = model.name
                        isShowingRenameDialog = true
                    }
                }
            }
        }
        .listStyle(.plain)
        .searchable(text: $localSearchText, prompt: "搜索模型...")
        .refreshable {
            await viewModel.fetchModels()
        }
        .alert(
            String(localized: "Rename Model"),
            isPresented: $isShowingRenameDialog
        ) {
            TextField(String(localized: "Model Name"), text: $renameText)
            Button(String(localized: "Cancel"), role: .cancel) {
                modelBeingRenamed = nil
            }
            Button(String(localized: "Save")) {
                if let modelBeingRenamed {
                    viewModel.renameModel(modelBeingRenamed, to: renameText)
                }
                modelBeingRenamed = nil
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }
}
