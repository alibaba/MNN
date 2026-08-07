//
//  PerformanceView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/1/25.
//

import SwiftUI

/// A view component for displaying LLM performance metrics
/// Shows various performance indicators in a compact, readable format
struct PerformanceView: View {
    let performanceData: String
    @State private var isExpanded: Bool = true
    
    private let animationDuration: Double = 0.3
    
    var body: some View {
        if !performanceData.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                // Header with expand/collapse button
                HStack {
                    Image(systemName: "speedometer")
                        .foregroundColor(.blue)
                        .font(.system(size: 14, weight: .medium))
                    
                    Text("Performance Metrics")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.blue)
                    
                    Spacer()
                    
                    Button(action: {
                        withAnimation(.easeInOut(duration: animationDuration)) {
                            isExpanded.toggle()
                        }
                    }) {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(.blue)
                            .font(.system(size: 12, weight: .medium))
                            .rotationEffect(.degrees(isExpanded ? 180 : 0))
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
                
                // Content area
                if isExpanded {
                    VStack(alignment: .leading, spacing: 6) {
                        if let metrics = parsePerformanceData(performanceData) {
                            ForEach(metrics, id: \.key) { metric in
                                HStack {
                                    Text(metric.key)
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundColor(.secondary)
                                    
                                    Spacer()
                                    
                                    Text(metric.value)
                                        .font(.system(size: 12, weight: .regular, design: .monospaced))
                                        .foregroundColor(.primary)
                                }
                            }
                        } else {
                            Text(performanceData)
                                .font(.system(size: 12, weight: .regular, design: .monospaced))
                                .foregroundColor(.primary)
                                .multilineTextAlignment(.leading)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
                } else {
                    // Compact summary view
                    HStack {
                        if let summary = getPerformanceSummary(performanceData) {
                            Text(summary)
                                .font(.system(size: 12, weight: .regular))
                                .foregroundColor(.blue.opacity(0.8))
                        } else {
                            Text("Tap to view details")
                                .font(.system(size: 12, weight: .regular))
                                .foregroundColor(.blue.opacity(0.6))
                        }
                        
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.blue.opacity(0.05))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.blue.opacity(0.2), lineWidth: 1)
                    )
            )
            .padding(.horizontal, 4)
            .padding(.vertical, 4)
        }
    }
    
    /// Parse performance data string into key-value pairs
    /// - Parameter data: Raw performance data string
    /// - Returns: Array of key-value pairs or nil if parsing fails
    private func parsePerformanceData(_ data: String) -> [(key: String, value: String)]? {
        let lines = data.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        
        var metrics: [(key: String, value: String)] = []
        
        for line in lines {
            if line.contains(":") {
                let components = line.components(separatedBy: ":")
                if components.count >= 2 {
                    let key = components[0].trimmingCharacters(in: .whitespaces)
                    let value = components[1...].joined(separator: ":").trimmingCharacters(in: .whitespaces)
                    metrics.append((key: key, value: value))
                }
            } else if line.contains("=") {
                let components = line.components(separatedBy: "=")
                if components.count >= 2 {
                    let key = components[0].trimmingCharacters(in: .whitespaces)
                    let value = components[1...].joined(separator: "=").trimmingCharacters(in: .whitespaces)
                    metrics.append((key: key, value: value))
                }
            }
        }
        
        return metrics.isEmpty ? nil : metrics
    }
    
    /// Generate a compact summary of performance data
    /// - Parameter data: Raw performance data string
    /// - Returns: Summary string or nil if no meaningful summary can be generated
    private func getPerformanceSummary(_ data: String) -> String? {
        // Look for common performance indicators
        let lowercaseData = data.lowercased()
        
        if lowercaseData.contains("tokens/s") || lowercaseData.contains("token/s") {
            // Extract token generation speed
            let pattern = #"([0-9.]+)\s*tokens?/s"#
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: data, options: [], range: NSRange(location: 0, length: data.count)),
               let range = Range(match.range(at: 1), in: data) {
                let speed = String(data[range])
                return "\(speed) tokens/s"
            }
        }
        
        if lowercaseData.contains("ms") || lowercaseData.contains("millisecond") {
            // Extract timing information
            let pattern = #"([0-9.]+)\s*ms"#
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: data, options: [], range: NSRange(location: 0, length: data.count)),
               let range = Range(match.range(at: 1), in: data) {
                let time = String(data[range])
                return "\(time)ms"
            }
        }
        
        return nil
    }
}

#if DEBUG
struct PerformanceView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 16) {
            // Structured performance data
            PerformanceView(performanceData: "Generation Speed: 25.3 tokens/s\nLatency: 120ms\nMemory Usage: 2.1GB\nModel Size: 7B parameters")
            
            // Simple performance data
            PerformanceView(performanceData: "Processing time: 450ms")
            
            // Raw performance data
            PerformanceView(performanceData: "Total tokens: 150\nGeneration time: 2.3s\nAverage speed: 65.2 tokens/s")
            
            // Empty data (should not display)
            PerformanceView(performanceData: "")
        }
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
#endif
