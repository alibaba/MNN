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
    @Binding var showBatchFileTest: Bool
    @State private var showMenu = false
    
    var body: some View {
        Menu {
            Button(action: {
                showBatchFileTest = true
            }) {
                Label("批量文件测试", systemImage: "doc.text.fill")
            }
            
            // Future menu items can be added here
            // Button(action: {}) {
            //     Label("其他功能", systemImage: "star.fill")
            // }
        } label: {
            Image(systemName: "ellipsis")
                .foregroundColor(.primary)
                .font(.system(size: 16, weight: .medium))
        }
    }
}

#Preview {
    ChatMenuView(showBatchFileTest: .constant(false))
}