//
//  TagChip.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct TagChip: View {
    let text: String
    
    var body: some View {
        Text(TagTranslationManager.shared.getLocalizedTag(text))
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
            )
    }
}
