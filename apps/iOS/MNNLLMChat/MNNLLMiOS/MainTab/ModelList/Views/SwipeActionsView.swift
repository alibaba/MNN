//
//  SwipeActionsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//


import SwiftUI

struct SwipeActionsView: View {
    let model: ModelInfo
    @ObservedObject var viewModel: ModelListViewModel
    
    private func isBuiltInLocalModel(_ model: ModelInfo) -> Bool {
        guard let vendor = model.vendor, vendor == "Local" else { return false }
        guard let sources = model.sources, let localSource = sources["local"] else { return false }
        return localSource.hasPrefix("bundle_root/")
    }
    
    var body: some View {
        if viewModel.pinnedModelIds.contains(model.id) {
            Button {
                viewModel.unpinModel(model)
            } label: {
                Label(LocalizedStringKey("button.unpin"), systemImage: "pin.slash")
            }.tint(.gray)
        } else {
            Button {
                viewModel.pinModel(model)
            } label: {
                Label(LocalizedStringKey("button.pin"), systemImage: "pin")
            }.tint(.primaryBlue)
        }
        if model.isDownloaded && !isBuiltInLocalModel(model) {
            Button(role: .destructive) {
                Task {
                    await viewModel.deleteModel(model)
                }
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .tint(.primaryRed)
        }
    }
}
