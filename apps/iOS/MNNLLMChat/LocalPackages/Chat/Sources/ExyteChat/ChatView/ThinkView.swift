//
//  ThinkView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/1/25.
//

import SwiftUI

/// A view component for displaying AI thinking process content
/// Supports collapsible/expandable functionality with smooth animations
struct ThinkView: View {
    let thinkContent: String
    @State private var isExpanded: Bool = true
    @State private var contentHeight: CGFloat = 0
    
    private let maxCollapsedHeight: CGFloat = 60
    private let animationDuration: Double = 0.3
    
    var body: some View {
        if !thinkContent.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                // Header with expand/collapse button
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.secondary)
                        .font(.system(size: 14, weight: .medium))
                    
                    Text("AI Thinking Process")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Button(action: {
                        withAnimation(.easeInOut(duration: animationDuration)) {
                            isExpanded.toggle()
                        }
                    }) {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(.secondary)
                            .font(.system(size: 12, weight: .medium))
                            .rotationEffect(.degrees(isExpanded ? 180 : 0))
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
                
                // Content area
                VStack(alignment: .leading, spacing: 0) {
                    Text(thinkContent)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundColor(.primary.opacity(0.8))
                        .lineLimit(isExpanded ? nil : 3)
                        .multilineTextAlignment(.leading)
                        .background(
                            GeometryReader { geometry in
                                Color.clear
                                    .onAppear {
                                        contentHeight = geometry.size.height
                                    }
                                    .onChange(of: thinkContent) { _ in
                                        contentHeight = geometry.size.height
                                    }
                            }
                        )
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)
                }
                .frame(maxHeight: isExpanded ? .infinity : maxCollapsedHeight)
                .clipped()
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
            )
            .padding(.horizontal, 4)
            .padding(.vertical, 4)
        }
    }
}

#if DEBUG
struct ThinkView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 16) {
            // Short content preview
            ThinkView(thinkContent: "This is a short thinking process.")
            
            // Long content preview
            ThinkView(thinkContent: "This is a much longer thinking process that demonstrates how the component handles multiple lines of text. It should show the collapsible functionality when the content exceeds the maximum collapsed height. The user can tap the chevron button to expand or collapse the content with smooth animations.")
            
            // Empty content (should not display)
            ThinkView(thinkContent: "")
        }
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
#endif
