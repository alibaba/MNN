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
    let onRename: () -> Void
    let onDelete: () -> Void

    var body: some View {
        if model.isDownloaded && !ModelUtils.isBuiltInLocalModel(model) {
            ModelDeleteSwipeButton(action: onDelete)
        }

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

        Button(action: onRename) {
            Label(AppText.text("Rename"), systemImage: "pencil")
        }
        .tint(.blue)
    }
}

/// Not `role: .destructive`: that animates the row away before the user
/// has confirmed, and catalog rows stay in the list after deletion anyway.
struct ModelDeleteSwipeButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(AppText.text("Delete"), systemImage: "trash")
        }
        .tint(.primaryRed)
    }
}

struct ModelDeletionRequest {
    let model: ModelInfo
    /// The chat has this model loaded; deleting unloads it first.
    let isInUse: Bool

    static func make(for model: ModelInfo, isInUse: Bool) -> ModelDeletionRequest {
        ModelDeletionRequest(model: model, isInUse: isInUse)
    }
}

extension View {
    func modelDeletionAlert(_ request: Binding<ModelDeletionRequest?>,
                            onDelete: @escaping (ModelInfo) -> Void) -> some View {
        alert(
            AppText.text("alert.deleteModel.title"),
            isPresented: Binding(get: { request.wrappedValue != nil },
                                 set: { if !$0 { request.wrappedValue = nil } }),
            presenting: request.wrappedValue
        ) { pending in
            Button(AppText.text("Delete"), role: .destructive) { onDelete(pending.model) }
            Button(AppText.text("Cancel"), role: .cancel) {}
        } message: { pending in
            Text(Self.deletionMessage(for: pending))
        }
    }

    private static func deletionMessage(for request: ModelDeletionRequest) -> String {
        var message = request.model.expectedBytes.map {
            String(format: AppText.text("alert.deleteModel.detail"), request.model.name, ByteText.file($0))
        } ?? AppText.text("alert.deleteModel.message")
        if request.isInUse { message += "\n" + AppText.text("alert.deleteModel.inUse.message") }
        return message
    }
}
