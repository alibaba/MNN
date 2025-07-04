//
//  SourceSelector.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

// MARK: - 下载源选择器
struct SourceSelector: View {
    @Binding var selectedSource: ModelSource
    @Binding var showSourceMenu: Bool
    let onSourceChange: (ModelSource) -> Void
    
    var body: some View {
        Menu {
            ForEach(ModelSource.allCases) { source in
                Button(action: {
                    onSourceChange(source)
                }) {
                    HStack {
                        Text(source.rawValue)
                        if source == selectedSource {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text("下载源:")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.primary)
                
                Text(selectedSource.rawValue)
                    .font(.system(size: 12, weight: .regular))
                    .foregroundColor(.primary)
                
                Image(systemName: "chevron.down")
                    .font(.system(size: 10))
                    .foregroundColor(.primary)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 6)
        }
    }
}
