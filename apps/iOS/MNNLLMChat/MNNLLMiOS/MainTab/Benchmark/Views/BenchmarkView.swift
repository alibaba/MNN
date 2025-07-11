//
//  BenchmarkView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import SwiftUI

/**
 * Main benchmark view that provides interface for running performance tests on ML models.
 * Features include model selection, progress tracking, and results visualization.
 */
struct BenchmarkView: View {
    @StateObject private var viewModel = BenchmarkViewModel()
    @State private var showStopConfirmation = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Model Selection Section
                    modelSelectionSection
                    
                    // Progress Section
                    if viewModel.showProgressBar {
                        progressSection
                    }
                    
                    // Status Section
                    if !viewModel.statusMessage.isEmpty {
                        statusSection
                    }
                    
                    // Results Section
                    if viewModel.showResults, let results = viewModel.benchmarkResults {
                        resultsSection(results)
                    }
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle("Benchmark")
        }
        .alert("Stop Benchmark", isPresented: $showStopConfirmation) {
            Button("Yes", role: .destructive) {
                viewModel.onStopBenchmarkTapped()
            }
            Button("No", role: .cancel) { }
        } message: {
            Text("Are you sure you want to stop the benchmark test?")
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK") { }
        } message: {
            Text(viewModel.errorMessage)
        }
        .onReceive(viewModel.$isRunning) { isRunning in
            if isRunning && viewModel.startButtonText.contains("Stop") {
                showStopConfirmation = false
            }
        }
    }
    
    // MARK: - UI Sections
    
    /// Model selection interface with dropdown menu and start/stop controls
    private var modelSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Select Model")
                .font(.headline)
                .foregroundColor(.primary)
            
            if viewModel.isLoading {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Loading models...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Menu {
                    if viewModel.availableModels.isEmpty {
                        Button("No models available") {
                            // Placeholder - no action
                        }
                        .disabled(true)
                    } else {
                        ForEach(viewModel.availableModels, id: \.id) { model in
                            Button(action: {
                                viewModel.onModelSelected(model)
                            }) {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(model.modelName)
                                            .font(.system(size: 14, weight: .medium))
                                        Text("Local")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                        .font(.caption)
                                }
                            }
                        }
                    }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(viewModel.selectedModel?.modelName ?? "Select a model")
                                .font(.system(size: 16, weight: .medium))
                                .foregroundColor(viewModel.selectedModel != nil ? .primary : .secondary)
                            
                            if let model = viewModel.selectedModel {
                                HStack(spacing: 8) {
                                    Text("Local Available")
                                        .font(.caption)
                                        .foregroundColor(.green)
                                    
                                    if let size = model.cachedSize {
                                        Text("• \(formatBytes(size))")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                            }
                        }
                        
                        Spacer()
                        
                        Image(systemName: "chevron.down")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(.systemGray4), lineWidth: 1)
                    )
                }
            }
            
            // Start/Stop Button
            Button(action: {
                if viewModel.startButtonText.contains("Stop") {
                    showStopConfirmation = true
                } else {
                    viewModel.onStartBenchmarkTapped()
                }
            }) {
                HStack {
                    if viewModel.isRunning && viewModel.startButtonText.contains("Stop") {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.8)
                    }
                    
                    Text(viewModel.startButtonText)
                        .font(.system(size: 16, weight: .semibold))
                }
                .frame(maxWidth: .infinity)
                .frame(height: 44)
                .background(
                    viewModel.isStartButtonEnabled ? 
                    (viewModel.startButtonText.contains("Stop") ? Color.red : Color.blue) : 
                    Color.gray
                )
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .disabled(!viewModel.isStartButtonEnabled || viewModel.selectedModel == nil)
            
            // Status messages for user guidance
            if viewModel.selectedModel == nil {
                Text("Start benchmark after selecting your model")
                    .font(.caption)
                    .foregroundColor(.orange)
            } else if viewModel.availableModels.isEmpty {
                Text("No local models found. Please download a model first.")
                    .font(.caption)
                    .foregroundColor(.orange)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        )
    }
    
    /// Progress tracking section with detailed test metrics
    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let progress = viewModel.currentProgress {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Progress")
                            .font(.headline)
                        Spacer()
                        Text("\(progress.progress)%")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    ProgressView(value: Double(progress.progress), total: 100)
                        .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                    
                    // Detailed test information for running tests
                    if progress.progressType == .runningTest && progress.totalIterations > 0 {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Test \(progress.currentIteration)/\(progress.totalIterations) (PP=\(progress.nPrompt), TG=\(progress.nGenerate))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            // Real-time performance metrics
                            if progress.runTimeSeconds > 0 {
                                HStack(spacing: 16) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Runtime")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text(String(format: "%.3fs", progress.runTimeSeconds))
                                            .font(.caption)
                                    }
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Prefill")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text(String(format: "%.3fs", progress.prefillTimeSeconds))
                                            .font(.caption)
                                    }
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Decode")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text(String(format: "%.3fs", progress.decodeTimeSeconds))
                                            .font(.caption)
                                    }
                                    
                                    Spacer()
                                }
                                
                                HStack(spacing: 16) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Prefill Speed")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text(String(format: "%.2f t/s", progress.prefillSpeed))
                                            .font(.caption)
                                    }
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Decode Speed")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text(String(format: "%.2f t/s", progress.decodeSpeed))
                                            .font(.caption)
                                    }
                                    
                                    Spacer()
                                }
                            }
                        }
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Progress")
                        .font(.headline)
                    ProgressView()
                        .progressViewStyle(LinearProgressViewStyle())
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        )
    }
    
    /// Simple status message display
    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Status")
                .font(.headline)
            Text(viewModel.statusMessage)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        )
    }
    
    /// Comprehensive results display with performance metrics and sharing options
    private func resultsSection(_ results: BenchmarkResults) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Test Result")
                    .font(.title2)
                    .fontWeight(.bold)
                Spacer()
                Menu {
                    Button("Share Results") {
                        shareResults(results)
                    }
                    Button("Delete Results", role: .destructive) {
                        viewModel.onDeleteResultTapped()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.title3)
                }
            }
            
            // Model and device information
            VStack(alignment: .leading, spacing: 8) {
                Text(results.modelDisplayName)
                    .font(.headline)
                
                Text(BenchmarkResultsHelper.shared.getDeviceInfo())
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            // Process and display statistics
            let statistics = BenchmarkResultsHelper.shared.processTestResults(results.testResults)
            
            // Configuration details
            VStack(alignment: .leading, spacing: 8) {
                Text("Benchmark Config")
                    .font(.headline)
                Text(statistics.configText)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            
            // Performance metrics display
            VStack(spacing: 16) {
                // Prompt Processing (Prefill) performance
                if let prefillStats = statistics.prefillStats {
                    performanceMetricView(
                        title: "Prompt Processing",
                        value: BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(prefillStats),
                        icon: "cpu"
                    )
                }
                
                // Token Generation (Decode) performance
                if let decodeStats = statistics.decodeStats {
                    performanceMetricView(
                        title: "Token Generation",
                        value: BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(decodeStats),
                        icon: "bolt"
                    )
                }
                
                // Memory usage statistics
                let totalMemoryKb = BenchmarkResultsHelper.shared.getTotalSystemMemoryKb()
                let memoryInfo = BenchmarkResultsHelper.shared.formatMemoryUsage(
                    maxMemoryKb: results.maxMemoryKb,
                    totalKb: totalMemoryKb
                )
                performanceMetricView(
                    title: "Peak Memory Usage",
                    value: memoryInfo.valueText,
                    subtitle: memoryInfo.labelText,
                    icon: "memorychip"
                )
            }
            
            // Test summary information
            VStack(alignment: .leading, spacing: 8) {
                Text(BenchmarkResultsHelper.shared.formatModelParams(
                    totalTokens: statistics.totalTokensProcessed,
                    totalTests: statistics.totalTests
                ))
                .font(.caption)
                .foregroundColor(.secondary)
            }
            
            // Completion timestamp
            Text("Completed: \(results.timestamp)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        )
    }
    
    /// Reusable performance metric display component
    private func performanceMetricView(title: String, value: String, subtitle: String? = nil, icon: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.title3)
                    .fontWeight(.semibold)
                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.secondarySystemBackground))
        )
    }
    
    // MARK: - Helper Functions & Sharing
    
    /// Formats byte count into human-readable string
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
    
    /// Initiates sharing of benchmark results through system share sheet
    private func shareResults(_ results: BenchmarkResults) {
        let shareText = formatResultsForSharing(results)
        let activityViewController = UIActivityViewController(
            activityItems: [shareText],
            applicationActivities: nil
        )
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootViewController = window.rootViewController {
            
            // Configure popover for iPad presentation
            if let popover = activityViewController.popoverPresentationController {
                popover.sourceView = window
                popover.sourceRect = CGRect(x: window.bounds.midX, y: window.bounds.midY, width: 0, height: 0)
                popover.permittedArrowDirections = []
            }
            
            rootViewController.present(activityViewController, animated: true)
        }
    }
    
    /// Formats benchmark results into shareable text format with performance metrics and hashtags
    private func formatResultsForSharing(_ results: BenchmarkResults) -> String {
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
            shareText += "\n🔄 \(prefillStats.label): \(BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(prefillStats))"
        }
        
        if let decodeStats = statistics.decodeStats {
            shareText += "\n⚡️ \(decodeStats.label): \(BenchmarkResultsHelper.shared.formatSpeedStatisticsLine(decodeStats))"
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

// MARK: - Preview
struct BenchmarkView_Previews: PreviewProvider {
    static var previews: some View {
        BenchmarkView()
    }
}
