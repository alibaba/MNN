//
//  StatusCard.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/// Reusable status display card component for benchmark interface.
/// Shows status messages and updates to provide user feedback.
struct StatusCard: View {
    let statusMessage: String
    
    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.benchmarkWarning.opacity(0.2), Color.benchmarkWarning.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 40, height: 40)
                
                Image(systemName: "info.circle")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.benchmarkWarning)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(String(localized: "Status Update"))
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                Text(statusMessage)
                    .font(.subheadline)
                    .foregroundColor(.benchmarkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.benchmarkCardBg)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.benchmarkWarning.opacity(0.3), lineWidth: 1)
                )
        )
    }
}

#Preview {
    StatusCard(statusMessage: "Initializing benchmark test environment...")
        .padding()
}