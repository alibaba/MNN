//
//  ButtonStyle.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 10.07.2023.
//

import SwiftUI

struct BlueButton: ButtonStyle {

    @Environment(\.isEnabled) private var isEnabled: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(17, .white, .medium)
            .frame(height: 48)
            .frame(maxWidth: .infinity)
            .background {
                RoundedRectangle(cornerRadius: 14)
                    .foregroundColor(isEnabled ? .exampleBlue : .exampleMidGray)
            }
    }
}
