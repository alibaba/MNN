//
//  MetricCard.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/**
 * Reusable metric display card component.
 * Shows performance metrics with icon, title, and value in a compact format.
 */
struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundColor(.benchmarkAccent)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.benchmarkSecondary)
                    .lineLimit(1)
            }
            
            Text(value)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.primary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.benchmarkAccent.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.benchmarkAccent.opacity(0.1), lineWidth: 1)
                )
        )
    }
}

#Preview {
    HStack(spacing: 12) {
        MetricCard(title: "Runtime", value: "2.456s", icon: "clock")
        MetricCard(title: "Speed", value: "109.8 t/s", icon: "speedometer")
        MetricCard(title: "Memory", value: "1.2 GB", icon: "memorychip")
    }
    .padding()
}