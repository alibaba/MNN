//
//  MixedSamplersView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/3/24.
//

import Foundation
import SwiftUI

struct MixedSamplersView: View {
    @Binding var selectedSamplers: Set<String>
    @Binding var samplersOrder: [String]
    let onUpdate: () -> Void

    @Binding var temperature: Double
    @Binding var topK: Double
    @Binding var topP: Double
    @Binding var minP: Double
    @Binding var tfsZ: Double
    @Binding var typical: Double
    @Binding var penalty: Double

    let onUpdateTemperature: (Double) -> Void
    let onUpdateTopK: (Int) -> Void
    let onUpdateTopP: (Double) -> Void
    let onUpdateMinP: (Double) -> Void
    let onUpdateTfsZ: (Double) -> Void
    let onUpdateTypical: (Double) -> Void
    let onUpdatePenalty: (Double) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SettingLabel("Configure mixed samplers:", help: "help.mixed")
                .font(.subheadline)
                .foregroundColor(.secondary)

            List {
                ForEach(samplersOrder, id: \.self) { sampler in
                    VStack {
                        HStack {
                            switch sampler {
                            case "temperature":
                                ParameterSliderView(
                                    title: "Temperature",
                                    value: $temperature,
                                    range: 0.0 ... 2.0,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: onUpdateTemperature,
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            case "topK":
                                ParameterSliderView(
                                    title: "Top K",
                                    value: $topK,
                                    range: 1 ... 100,
                                    format: "%.0f",
                                    intValue: true,
                                    onChanged: { onUpdateTopK(Int($0)) },
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            case "topP":
                                ParameterSliderView(
                                    title: "Top P",
                                    value: $topP,
                                    range: 0.0 ... 1.0,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: onUpdateTopP,
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            case "minP":
                                ParameterSliderView(
                                    title: "Min P",
                                    value: $minP,
                                    range: 0.05 ... 0.3,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: onUpdateMinP,
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            case "tfs":
                                ParameterSliderView(
                                    title: "TFS-Z",
                                    value: $tfsZ,
                                    range: 0.9 ... 0.99,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: onUpdateTfsZ,
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            case "typical":
                                ParameterSliderView(
                                    title: "Typical",
                                    value: $typical,
                                    range: 0.8 ... 0.95,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: onUpdateTypical,
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            case "penalty":
                                ParameterSliderView(
                                    title: "Penalty",
                                    value: $penalty,
                                    range: SamplerType.penaltyRange,
                                    format: "%.2f",
                                    intValue: false,
                                    onChanged: onUpdatePenalty,
                                    isEnabled: selectedSamplers.contains(sampler)
                                )
                            default:
                                EmptyView()
                            }

                            // Toggle switch
                            Toggle("", isOn: Binding(
                                get: { selectedSamplers.contains(sampler) },
                                set: { isSelected in
                                    if isSelected {
                                        selectedSamplers.insert(sampler)
                                    } else {
                                        selectedSamplers.remove(sampler)
                                    }
                                    onUpdate()
                                }
                            ))
                            .labelsHidden()
                        }
                        .padding(.vertical, 4)
                    }
                }
                .onMove { from, to in
                    samplersOrder.move(fromOffsets: from, toOffset: to)
                    onUpdate()
                }
            }
            .environment(\.editMode, .constant(.active))
        }
    }
}
