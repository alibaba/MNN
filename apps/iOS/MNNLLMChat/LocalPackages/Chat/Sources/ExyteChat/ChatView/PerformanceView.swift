//
//  PerformanceView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/1/25.
//

import SwiftUI

/// Displays the final LLM performance metrics as unobtrusive plain text.
struct PerformanceView: View {
    let performanceData: String

    var body: some View {
        if !performanceData.isEmpty {
            Text(performanceData)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
        }
    }
}

#if DEBUG
struct PerformanceView_Previews: PreviewProvider {
    static var previews: some View {
        PerformanceView(performanceData: "Prefill: 42.1 tokens/s\nDecode: 15.8 tokens/s")
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
#endif
