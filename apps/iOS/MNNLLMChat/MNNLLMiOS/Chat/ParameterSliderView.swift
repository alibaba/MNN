//
//  ParameterSliderView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/3/19.
//

import SwiftUI

struct ParameterSliderView: View {
    let title: String
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let format: String
    let intValue: Bool
    let onChanged: (Double) -> Void
    
    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(intValue ? "\(Int(value.wrappedValue))" : String(format: format, value.wrappedValue))
        }
        Slider(value: value, in: range)
            .onChange(of: value.wrappedValue) { _, newValue in
                onChanged(newValue)
            }
    }
}
