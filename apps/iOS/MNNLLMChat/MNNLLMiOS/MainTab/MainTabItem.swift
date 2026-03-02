//
//  MainTabItem.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/9/3.
//

import SwiftUI

struct MainTabItem: View {
    let imageName: String
    let title: String
    let isSelected: Bool
    
    var body: some View {
        VStack {
            Image(imageName)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 10, height: 10)
            Text(title)
        }
        .foregroundColor(isSelected ? .primaryPurple : .gray)
    }
}
