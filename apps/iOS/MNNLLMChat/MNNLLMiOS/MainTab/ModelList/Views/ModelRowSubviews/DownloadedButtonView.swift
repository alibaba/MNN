//
//  DownloadedButtonView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct DownloadedButtonView: View {
    @Binding var showDeleteAlert: Bool
    
    var body: some View {
        Button(action: { showDeleteAlert = true }) {
            VStack(spacing: 2) {
                Image(systemName: "trash")
                    .font(.system(size: 16))
                    .foregroundColor(.primary.opacity(0.8))
                
                Text(LocalizedStringKey("button.downloaded"))
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .allowsTightening(true)
            }
        }
    }
}
