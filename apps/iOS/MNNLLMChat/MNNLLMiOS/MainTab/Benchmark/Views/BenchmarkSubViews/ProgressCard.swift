//
//  ProgressCard.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/**
 * Reusable progress tracking card component for benchmark interface.
 * Displays test progress with detailed metrics and visual indicators.
 */
struct ProgressCard: View {
    let progress: BenchmarkProgress?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            if let progress = progress {
                VStack(alignment: .leading, spacing: 16) {
                    progressHeader(progress)
                    progressBar(progress)
                    
                    if progress.progressType == .runningTest && progress.totalIterations > 0 {
                        testDetails(progress)
                    }
                }
            } else {
                fallbackProgress
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.benchmarkCardBg)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.benchmarkSuccess.opacity(0.3), lineWidth: 1)
                )
        )
    }
    
    // MARK: - Private Views
    
    private func progressHeader(_ progress: BenchmarkProgress) -> some View {
        HStack {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.benchmarkAccent.opacity(0.2), Color.benchmarkGradientEnd.opacity(0.1)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 40, height: 40)
                    
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.benchmarkAccent)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Test Progress")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                    
                    Text("Running performance tests")
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(progress.progress)%")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.benchmarkAccent)
                
                Text("Complete")
                    .font(.caption)
                    .foregroundColor(.benchmarkSecondary)
            }
        }
    }
    
    private func progressBar(_ progress: BenchmarkProgress) -> some View {
        VStack(spacing: 8) {
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.2))
                    .frame(height: 8)
                
                RoundedRectangle(cornerRadius: 8)
                    .fill(
                        LinearGradient(
                            colors: [Color.benchmarkGradientStart, Color.benchmarkGradientEnd],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: CGFloat(progress.progress) / 100 * UIScreen.main.bounds.width * 0.8, height: 8)
                    .animation(.easeInOut(duration: 0.3), value: progress.progress)
            }
        }
    }
    
    private func testDetails(_ progress: BenchmarkProgress) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Test iteration info
            HStack {
                Image(systemName: "repeat")
                    .font(.caption)
                    .foregroundColor(.benchmarkAccent)
                
                Text("Test \(progress.currentIteration) of \(progress.totalIterations)")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Text("PP: \(progress.nPrompt) • TG: \(progress.nGenerate)")
                    .font(.caption)
                    .foregroundColor(.benchmarkSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.benchmarkAccent.opacity(0.1))
                    )
            }
            
            // Real-time performance metrics
            if progress.runTimeSeconds > 0 {
                VStack(spacing: 12) {
                    // Timing metrics
                    HStack(spacing: 12) {
                        MetricCard(title: "Runtime", value: String(format: "%.3fs", progress.runTimeSeconds), icon: "clock")
                        MetricCard(title: "Prefill", value: String(format: "%.3fs", progress.prefillTimeSeconds), icon: "arrow.up.circle")
                        MetricCard(title: "Decode", value: String(format: "%.3fs", progress.decodeTimeSeconds), icon: "arrow.down.circle")
                    }
                    
                    // Speed metrics
                    HStack(spacing: 12) {
                        MetricCard(title: "Prefill Speed", value: String(format: "%.2f t/s", progress.prefillSpeed), icon: "speedometer")
                        MetricCard(title: "Decode Speed", value: String(format: "%.2f t/s", progress.decodeSpeed), icon: "gauge")
                        Spacer()
                    }
                }
            }
        }
    }
    
    private var fallbackProgress: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Progress")
                .font(.headline)
            ProgressView()
                .progressViewStyle(LinearProgressViewStyle())
        }
    }
}

#Preview {
    ProgressCard(
        progress: BenchmarkProgress(
            progress: 65,
            statusMessage: "Running benchmark...",
            progressType: .runningTest,
            currentIteration: 3,
            totalIterations: 5,
            nPrompt: 128,
            nGenerate: 256,
            runTimeSeconds: 2.456,
            prefillTimeSeconds: 0.123,
            decodeTimeSeconds: 2.333,
            prefillSpeed: 1024.5,
            decodeSpeed: 109.8
        )
    )
    .padding()
}