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
    
    var body: some View {
        if viewModel.pinnedModelIds.contains(model.modelId) {
            Button {
                viewModel.unpinModel(model)
            } label: {
                Label("取消置顶", systemImage: "pin.slash")
            }.tint(.gray)
        } else {
            Button {
                viewModel.pinModel(model)
            } label: {
                Label("置顶", systemImage: "pin")
            }.tint(.primaryBlue)
        }
        if model.isDownloaded {
            Button(role: .destructive) {
                Task {
                    await viewModel.deleteModel(model)
                }
            } label: {
                Label("删除", systemImage: "trash")
            }
            .tint(.primaryRed)
        }
    }
} 
