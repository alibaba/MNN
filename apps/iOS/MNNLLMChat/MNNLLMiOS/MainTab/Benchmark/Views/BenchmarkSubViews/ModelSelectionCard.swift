//
//  ModelSelectionCard.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/21.
//

import SwiftUI

/// Reusable model selection card component for benchmark interface.
/// Provides dropdown menu for model selection and start/stop controls.
struct ModelSelectionCard: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Select Model")
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                Spacer()
            }
            
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
                modelDropdownMenu
            }
            
            startStopButton
            
            statusMessages
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
    
    private var modelDropdownMenu: some View {
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
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(viewModel.selectedModel?.modelName ?? String(localized: "Choose your AI model"))
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(viewModel.isRunning ? .secondary : (viewModel.selectedModel != nil ? .primary : .benchmarkSecondary))
                        .lineLimit(1)
                    
                    if let model = viewModel.selectedModel {
                        HStack(spacing: 8) {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(Color.benchmarkSuccess)
                                    .frame(width: 6, height: 6)
                                Text("Ready")
                                    .font(.caption)
                                    .foregroundColor(.benchmarkSuccess)
                            }
                            
                            if let size = model.cachedSize {
                                Text("• \(formatBytes(size))")
                                    .font(.caption)
                                    .foregroundColor(.benchmarkSecondary)
                            }
                        }
                    } else {
                        Text("Tap to select a model for testing")
                            .font(.caption)
                            .foregroundColor(.benchmarkSecondary)
                    }
                }
                
                Spacer()
                
                Image(systemName: "chevron.down")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(viewModel.isRunning ? .secondary : .benchmarkSecondary)
                    .rotationEffect(.degrees(0))
            }
            .padding(20)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.benchmarkCardBg)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(
                                viewModel.isRunning ? 
                                Color.gray.opacity(0.1) :
                                (viewModel.selectedModel != nil ? 
                                Color.benchmarkAccent.opacity(0.3) : 
                                Color.gray.opacity(0.2)),
                                lineWidth: 1
                            )
                    ))
        }
        .disabled(viewModel.isRunning)
    }
    
    private var startStopButton: some View {
        Button(action: {
            viewModel.onStartBenchmarkTapped()
        }) {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.white.opacity(0.2))
                        .frame(width: 32, height: 32)
                    
                    if viewModel.isRunning {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.7)
                    } else {
                        Image(systemName: viewModel.isRunning ? "stop.fill" : "play.fill")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.white)
                    }
                }
                
                Text(viewModel.startButtonText)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
                
                Spacer()
                
                if !viewModel.isRunning {
                    Image(systemName: "arrow.right")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(
                        viewModel.isStartButtonEnabled ? 
                        (viewModel.isRunning ? 
                         LinearGradient(
                             colors: [Color.benchmarkError, Color.benchmarkError.opacity(0.8)],
                             startPoint: .leading,
                             endPoint: .trailing
                         ) :
                         LinearGradient(
                             colors: [Color.benchmarkGradientStart, Color.benchmarkGradientEnd],
                             startPoint: .leading,
                             endPoint: .trailing
                         )) :
                        LinearGradient(
                            colors: [Color.gray, Color.gray.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
            )
        }
        .disabled(!viewModel.isStartButtonEnabled || viewModel.selectedModel == nil)
        .animation(.easeInOut(duration: 0.2), value: viewModel.startButtonText)
        .animation(.easeInOut(duration: 0.2), value: viewModel.isStartButtonEnabled)
    }
    
    private var statusMessages: some View {
        Group {
            if viewModel.selectedModel == nil {
                Text("Start benchmark after selecting your model")
                    .font(.caption)
                    .foregroundColor(.orange)
                    .padding(.horizontal, 16)
            } else if viewModel.availableModels.isEmpty {
                Text("No local models found. Please download a model first.")
                    .font(.caption)
                    .foregroundColor(.orange)
                    .padding(.horizontal, 16)
            }
        }
    }
    
    // MARK: - Helper Functions
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

#Preview {
    ModelSelectionCard(
        viewModel: BenchmarkViewModel()
    )
    .padding()
}
