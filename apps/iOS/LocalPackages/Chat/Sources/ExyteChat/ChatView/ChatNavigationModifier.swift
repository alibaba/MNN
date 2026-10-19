//
//  ChatNavigationModifier.swift
//  
//
//  Created by Alexandra Afonasova on 12.01.2023.
//

import SwiftUI

struct ChatNavigationModifier: ViewModifier {

    @Environment(\.presentationMode) private var presentationMode
    @Environment(\.chatTheme) private var theme
    
    let title: String
    let status: String?
    let cover: URL?
    
    func body(content: Content) -> some View {
        content
            .navigationBarBackButtonHidden()
            .toolbar {
                backButton
                infoToolbarItem
            }
    }
    
    private var backButton: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            Button { presentationMode.wrappedValue.dismiss() } label: {
                theme.images.backButton
            }
        }
    }
    
    private var infoToolbarItem: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            HStack {
                if let url = cover {
                    CachedAsyncImage(url: url, urlCache: .imageCache) { phase in
                        switch phase {
                        case .success(let image):
                            image
                                .resizable()
                                .scaledToFill()
                        default:
                            Rectangle().fill(theme.colors.mainTint)
                        }
                    }
                    .frame(width: 35, height: 35)
                    .clipShape(Circle())
                }
                
                VStack(alignment: .leading, spacing: 0) {
                    Text(title)
                        .fontWeight(.semibold)
                        .font(.headline)
                        .foregroundColor(theme.colors.mainText)
                    if let status = status {
                        Text(status)
                            .font(.footnote)
                            .foregroundColor(theme.colors.statusGray)
                    }
                }
                Spacer()
            }
            .padding(.leading, 10)
        }
    }
    
}
