
import SwiftUI

struct MixedSamplersView: View {
    @Binding var selectedSamplers: Set<String>
    @Binding var samplersOrder: [String]
    let onUpdate: () -> Void
    
    // 添加所需的参数状态
    @Binding var temperature: Double
    @Binding var topK: Double
    @Binding var topP: Double
    @Binding var minP: Double
    @Binding var tfsZ: Double
    @Binding var typical: Double
    @Binding var penalty: Double
    
    // 添加更新参数的回调
    let onUpdateTemperature: (Double) -> Void
    let onUpdateTopK: (Int) -> Void
    let onUpdateTopP: (Double) -> Void
    let onUpdateMinP: (Double) -> Void
    let onUpdateTfsZ: (Double) -> Void
    let onUpdateTypical: (Double) -> Void
    let onUpdatePenalty: (Double) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Select and arrange samplers:")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            List {
                ForEach(samplersOrder, id: \.self) { sampler in
                    HStack {
                        Image(systemName: "line.3.horizontal")
                            .foregroundColor(.gray)
                        Toggle(SamplerType(rawValue: sampler)?.displayName ?? sampler,
                               isOn: Binding(
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
