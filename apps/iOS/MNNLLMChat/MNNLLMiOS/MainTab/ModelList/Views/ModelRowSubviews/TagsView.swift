//
//  TagsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

// MARK: - 标签视图
struct TagsView: View {
    let tags: [String]
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(tags, id: \.self) { tag in
                    TagChip(text: tag)
                }
            }
            .padding(.horizontal, 1)
        }
        .frame(height: 25)
    }
}