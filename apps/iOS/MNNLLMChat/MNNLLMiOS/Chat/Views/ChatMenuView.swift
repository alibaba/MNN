//
//  ChatMenuView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/9/29.
//

import SwiftUI

/// Chat menu view that provides additional options for the chat interface
/// Displays a dropdown menu with options like batch file testing
struct ChatMenuView: View {
    // MARK: - Properties

    /// Binding to control batch file test presentation
    @Binding var showBatchFileTest: Bool

    /// Internal state for menu visibility
    @State private var showMenu = false

    // MARK: - Body

    var body: some View {
        Menu {
            
            Button(action: {
                showBatchFileTest = true
            }) {
                Label("Batch File Test", systemImage: "doc.text.fill")
            }
        } label: {
            Image(systemName: "ellipsis")
                .foregroundColor(.primary)
                .font(.system(size: 16, weight: .medium))
        }
    }
}

// MARK: - Preview

#Preview {
    ChatMenuView(
        showBatchFileTest: .constant(false)
    )
}
