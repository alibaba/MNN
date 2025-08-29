//
//  ResultsCard.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/// Reusable results display card component for benchmark interface.
/// Shows comprehensive benchmark results with performance metrics and statistics.
struct ResultsCard: View {
    let results: BenchmarkResults
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            resultsHeader
            infoHeader
            performanceMetrics
            detailedStats
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
    
    private var infoHeader: some View {
        
        let statistics = BenchmarkResultsHelper.shared.processTestResults(results.testResults)
        
        return VStack(alignment: .leading, spacing: 8) {
            Text(results.modelDisplayName)
                .font(.headline)
            Text(BenchmarkResultsHelper.shared.getDeviceInfo())
                .font(.subheadline)
                .foregroundColor(.secondary)
        
            Text("Benchmark Config")
                .font(.headline)
            Text(statistics.configText)
                .font(.subheadline)
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
                .foregroundColor(.secondary)
        }
    }
    
    private var resultsHeader: some View {
        HStack {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.benchmarkSuccess.opacity(0.2), Color.benchmarkSuccess.opacity(0.1)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 40, height: 40)
                    
                    Image(systemName: "chart.bar.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.benchmarkSuccess)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Benchmark Results")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                    
                    Text("Performance analysis complete")
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                }
            }
            
            Spacer()
            
            Button(action: {
                shareResults()
            }) {
                VStack(alignment: .center, spacing: 2) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.title2)
                        .foregroundColor(.benchmarkSuccess)
                    
                    Text("Share")
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                }
            }
            .buttonStyle(PlainButtonStyle())
        }
    }

    
    private var performanceMetrics: some View {
        let statistics = BenchmarkResultsHelper.shared.processTestResults(results.testResults)
        
        return VStack(spacing: 16) {
            HStack(spacing: 12) {
                if let prefillStats = statistics.prefillStats {
                    PerformanceMetricView(
                        icon: "speedometer",
                        title: "Prefill Speed",
                        value: BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(prefillStats),
                        subtitle: "Tokens per second",
                        color: .benchmarkGradientStart
                    )
                } else {
                    PerformanceMetricView(
                        icon: "speedometer",
                        title: "Prefill Speed",
                        value: "N/A",
                        subtitle: "Tokens per second",
                        color: .benchmarkGradientStart
                    )
                }
                
                if let decodeStats = statistics.decodeStats {
                    PerformanceMetricView(
                        icon: "gauge",
                        title: "Decode Speed",
                        value: BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(decodeStats),
                        subtitle: "Generation rate",
                        color: .benchmarkGradientEnd
                    )
                } else {
                    PerformanceMetricView(
                        icon: "gauge",
                        title: "Decode Speed",
                        value: "N/A",
                        subtitle: "Generation rate",
                        color: .benchmarkGradientEnd
                    )
                }
            }
            
            HStack(spacing: 12) {
                let totalMemoryKb = BenchmarkResultsHelper.shared.getTotalSystemMemoryKb()
                let memoryInfo = BenchmarkResultsHelper.shared.formatMemoryUsage(
                    maxMemoryKb: results.maxMemoryKb,
                    totalKb: totalMemoryKb
                )
                
                PerformanceMetricView(
                    icon: "memorychip",
                    title: "Memory Usage",
                    value: memoryInfo.valueText,
                    subtitle: "Peak memory",
                    color: .benchmarkWarning
                )
                
                PerformanceMetricView(
                    icon: "clock",
                    title: "Total Tokens",
                    value: "\(statistics.totalTokensProcessed)",
                    subtitle: "Complete duration",
                    color: .benchmarkSuccess
                )
            }
        }
    }
    
    private var detailedStats: some View {
        return VStack(alignment: .leading, spacing: 12) {
            VStack(spacing: 8) {
                HStack {
                    Text("Completed")
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                    Spacer()
                    Text(results.timestamp)
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                }
                
                HStack {
                    Text("Powered By MNN")
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                    Spacer()
                    Text(verbatim: "https://github.com/alibaba/MNN")
                        .font(.caption)
                        .foregroundColor(.benchmarkSecondary)
                }
            }
            .padding(.vertical, 8)
        }
    }
    
    // MARK: - Helper Functions
    
    /// Formats byte count into human-readable string
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
    
    /// Initiates sharing of benchmark results through system share sheet
    private func shareResults() {
        let viewToRender = self.body.frame(width: 390) // Adjust width as needed
        if let image = viewToRender.snapshot() {
            presentShareSheet(activityItems: [image, formatResultsForSharing()])
        } else {
            presentShareSheet(activityItems: [formatResultsForSharing()])
        }
    }

    private func presentShareSheet(activityItems: [Any]) {
        let activityViewController = UIActivityViewController(activityItems: activityItems, applicationActivities: nil)

        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootViewController = window.rootViewController {

            if let popover = activityViewController.popoverPresentationController {
                popover.sourceView = window
                popover.sourceRect = CGRect(x: window.bounds.midX, y: window.bounds.midY, width: 0, height: 0)
                popover.permittedArrowDirections = []
            }

            rootViewController.present(activityViewController, animated: true)
        }
    }
    
    /// Formats benchmark results into shareable text format with performance metrics and hashtags
    private func formatResultsForSharing() -> String {
        let statistics = BenchmarkResultsHelper.shared.processTestResults(results.testResults)
        let deviceInfo = BenchmarkResultsHelper.shared.getDeviceInfo()
        
        var shareText = """
        📱 MNN LLM Benchmark Results
        
        🤖 Model: \(results.modelDisplayName)
        📱 \(deviceInfo)
        📅 Completed: \(results.timestamp)
        
        📊 Configuration:
        \(statistics.configText)
        
        ⚡️ Performance Results:
        """
        
        if let prefillStats = statistics.prefillStats {
            shareText += "\n🔄 Prompt Processing: \(BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(prefillStats))"
        }
        
        if let decodeStats = statistics.decodeStats {
            shareText += "\n⚡️ Token Generation: \(BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(decodeStats))"
        }
        
        let totalMemoryKb = BenchmarkResultsHelper.shared.getTotalSystemMemoryKb()
        let memoryInfo = BenchmarkResultsHelper.shared.formatMemoryUsage(
            maxMemoryKb: results.maxMemoryKb,
            totalKb: totalMemoryKb
        )
        shareText += "\n💾 Peak Memory: \(memoryInfo.valueText) (\(memoryInfo.labelText))"
        
        shareText += "\n\n📈 Summary:"
        shareText += "\n• Total Tokens Processed: \(statistics.totalTokensProcessed)"
        shareText += "\n• Number of Tests: \(statistics.totalTests)"
        
        shareText += "\n\n#MNNLLMBenchmark #AIPerformance #MobileAI"
        
        return shareText
    }
}

extension View {
    func snapshot() -> UIImage? {
        let controller = UIHostingController(rootView: self)
        let view = controller.view

        let targetSize = controller.view.intrinsicContentSize
        view?.bounds = CGRect(origin: .zero, size: targetSize)
        view?.backgroundColor = .clear

        let renderer = UIGraphicsImageRenderer(size: targetSize)

        return renderer.image { _ in
            view?.drawHierarchy(in: controller.view.bounds, afterScreenUpdates: true)
        }
    }
}

#Preview {
    ResultsCard(
        results: BenchmarkResults(
            modelDisplayName: "Qwen2.5-1.5B-Instruct",
            maxMemoryKb: 1200000, // 1.2 GB in KB
            testResults: [],
            timestamp: "2025-01-21 14:30:25"
        )
    )
    .padding()
}
