//
//  EnhancedPerformanceMetricView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/// Enhanced performance metric display component.
/// Shows detailed performance metrics with gradient backgrounds, icons, and custom colors.
struct PerformanceMetricView: View {
    let icon: String
    let title: String
    let value: String
    let subtitle: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .center, spacing: 12) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [color.opacity(0.2), color.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 50, height: 50)
                
                Image(systemName: icon)
                    .font(.system(size: 25, weight: .semibold))
                    .foregroundColor(color)
            }
            
            VStack(alignment: .center, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.benchmarkSecondary)
            }
            
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
                .multilineTextAlignment(.center)
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(
                    LinearGradient(
                        colors: [Color.benchmarkCardBg, color.opacity(0.02)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(color.opacity(0.2), lineWidth: 1)
                )
        )
    }
}


#Preview {
    VStack(spacing: 16) {
        HStack(spacing: 12) {
            PerformanceMetricView(
                icon: "speedometer",
                title: String(localized: "Prefill Speed"),
                value: "1024.5 t/s",
                subtitle: String(localized: "Tokens per second"),
                color: .benchmarkGradientStart
            )
            
            PerformanceMetricView(
                icon: "gauge",
                title: String(localized: "Decode Speed"),
                value: "109.8 t/s",
                subtitle: String(localized: "Generation rate"),
                color: .benchmarkGradientEnd
            )
        }
        
        HStack(spacing: 12) {
            PerformanceMetricView(
                icon: "memorychip",
                title: String(localized: "Memory Usage"),
                value: "1.2 GB",
                subtitle: String(localized: "Peak memory"),
                color: .benchmarkWarning
            )
            
            PerformanceMetricView(
                icon: "clock",
                title: String(localized: "Total Time"),
                value: "2.456s",
                subtitle: String(localized: "Complete duration"),
                color: .benchmarkSuccess
            )
        }
    }
    .padding()
}
