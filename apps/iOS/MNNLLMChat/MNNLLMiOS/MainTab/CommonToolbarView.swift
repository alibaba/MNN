//
//  CommonToolbarView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/07/18.
//

import SwiftUI

struct CommonToolbarView: ToolbarContent {
    @Binding var showHistory: Bool
    @Binding var showHistoryButton: Bool
    @Binding var histories: [ChatHistory]
    
    var body: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            if showHistoryButton {
                Button(action: {
                    showHistory = true
                    showHistoryButton = false
                    histories = ChatHistoryManager.shared.getAllHistory()
                }) {
                    Image(systemName: "sidebar.left")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 20, height: 20)
                        .foregroundColor(.black)
                }
            }
        }
        
        ToolbarItem(placement: .navigationBarTrailing) {
            Button(action: {
                if let url = URL(string: "https://github.com/alibaba/MNN") {
                    UIApplication.shared.open(url)
                }
            }) {
                Image(systemName: "star")
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 20, height: 20)
                    .foregroundColor(.black)
            }
        }
    }
}