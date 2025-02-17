//
//  ModelIconView.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import SwiftUI

struct ModelIconView: View {
    let modelId: String
    
    var body: some View {
        if let iconName = ModelIconManager.shared.getModelImage(with: modelId) {
            Image(iconName)
                .resizable()
                .scaledToFit()
        } else {
            Image("mnn_icon")
                .resizable()
                .scaledToFit()
        }
    }
} 
