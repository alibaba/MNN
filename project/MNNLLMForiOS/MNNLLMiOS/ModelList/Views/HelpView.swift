//
//  HelpView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/13.
//

import SwiftUI

struct HelpView: View {
    var body: some View {
        WebView(url: URL(string: "https://github.com/alibaba/MNN")!) // ?tab=readme-ov-file#intro
            .navigationTitle("Help")
            .navigationBarTitleDisplayMode(.inline)
    }
}
