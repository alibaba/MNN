//
//  ToolbarView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

// MARK: - 工具栏视图
struct ToolbarView: View {
    @Binding var selectedSource: ModelSource
    @Binding var showSourceMenu: Bool
    @Binding var selectedTags: Set<String>
    @Binding var selectedCategories: Set<String>
    @Binding var selectedVendors: Set<String>
    let quickFilterTags: [String]
    @Binding var showFilterMenu: Bool
    let onSourceChange: (ModelSource) -> Void
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                // 下载源选择器
                SourceSelector(
                    selectedSource: $selectedSource,
                    showSourceMenu: $showSourceMenu,
                    onSourceChange: onSourceChange
                )
                
                // 快捷筛选标签
                QuickFilterTags(
                    tags: quickFilterTags,
                    selectedTags: $selectedTags
                )
                
                Spacer()
                
                // 右侧筛选按钮
                FilterButton(
                    showFilterMenu: $showFilterMenu,
                    selectedTags: $selectedTags,
                    selectedCategories: $selectedCategories,
                    selectedVendors: $selectedVendors
                )
            }
            .padding(.horizontal, 16)
        }
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .overlay(
            Rectangle()
                .frame(height: 0.5)
                .foregroundColor(Color(.separator)),
            alignment: .bottom
        )
    }
}
