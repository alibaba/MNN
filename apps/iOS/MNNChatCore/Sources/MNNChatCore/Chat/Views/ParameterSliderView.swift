//
//  ParameterSliderView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/3/19.
//

import Foundation
import SwiftUI

struct ParameterSliderView: View {
    let title: String
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let format: String
    let intValue: Bool
    let onChanged: (Double) -> Void
    /// Only the slider is disabled, so the "?" can still explain the setting.
    var isEnabled = true

    var body: some View {
        // One block (label row above the slider) so a containing HStack does
        // not squeeze the label and its "?" beside the slider.
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                if let helpKey = SettingHelp.key(forTitle: title) {
                    SettingLabel(title, help: helpKey, isDimmed: !isEnabled)
                } else {
                    Text(title).foregroundStyle(isEnabled ? .primary : .secondary)
                }
                Spacer()
                Text(intValue ? "\(Int(value.wrappedValue))" : String(format: format, value.wrappedValue))
                    .foregroundStyle(isEnabled ? .primary : .secondary)
            }
            Slider(value: value, in: range)
                .disabled(!isEnabled)
                .onChange(of: value.wrappedValue) { _, newValue in
                    onChanged(newValue)
                }
        }
    }
}
