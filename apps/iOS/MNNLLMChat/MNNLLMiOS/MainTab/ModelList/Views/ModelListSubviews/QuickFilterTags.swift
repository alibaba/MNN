//
//  QuickFilterTags.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

// MARK: - 快捷筛选标签
struct QuickFilterTags: View {
    let tags: [String]
    @Binding var selectedTags: Set<String>
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(tags, id: \.self) { tag in
                    FilterTagChip(
                        text: TagTranslationManager.shared.getLocalizedTag(tag),
                        isSelected: selectedTags.contains(tag)
                    ) {
                        if selectedTags.contains(tag) {
                            selectedTags.remove(tag)
                        } else {
                            selectedTags.insert(tag)
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
        }
    }
}
