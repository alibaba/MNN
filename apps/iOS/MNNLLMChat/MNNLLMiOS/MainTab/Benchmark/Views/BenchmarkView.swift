//
//  BenchmarkView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/// Main benchmark view that provides interface for running performance tests on ML models.
/// Features include model selection, progress tracking, and results visualization.
struct BenchmarkView: View {
    @StateObject private var viewModel = BenchmarkViewModel()
    
    var body: some View {
        ZStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Model Selection Section
                    ModelSelectionCard(
                        viewModel: viewModel
                    )
                    
                    // Progress Section
                    if viewModel.showProgressBar {
                        ProgressCard(progress: viewModel.currentProgress)
                            .transition(.asymmetric(
                                insertion: .scale.combined(with: .opacity),
                                removal: .opacity
                            ))
                    }
                    
                    // Status Section
                    if !viewModel.statusMessage.isEmpty {
                        StatusCard(statusMessage: viewModel.statusMessage)
                            .transition(.slide)
                    }
                    
                    // Results Section
                    if viewModel.showResults, let results = viewModel.benchmarkResults {
                        ResultsCard(results: results)
                            .transition(.asymmetric(
                                insertion: .move(edge: .bottom).combined(with: .opacity),
                                removal: .opacity
                            ))
                    }
                    
                    Spacer(minLength: 20)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
        }
        .alert(String(localized: "Stop Benchmark"), isPresented: $viewModel.showStopConfirmation) {
            Button(String(localized: "Yes"), role: .destructive) {
                viewModel.onStopBenchmarkTapped()
            }
            Button(String(localized: "No"), role: .cancel) { }
        } message: {
            Text(String(localized: "Are you sure you want to stop the benchmark test?"))
        }
        .alert(String(localized: "Error"), isPresented: $viewModel.showError) {
            Button(String(localized: "OK")) {
                viewModel.dismissError()
            }
        } message: {
            Text(viewModel.errorMessage)
        }

    }
}
