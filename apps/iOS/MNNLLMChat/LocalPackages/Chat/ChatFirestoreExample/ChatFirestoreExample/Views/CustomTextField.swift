//
//  CustomTextField.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 11.07.2023.
//

import SwiftUI

struct CustomTextField: View {

    var placeholder: String
    @Binding var text: String

    @FocusState private var isFocused: Bool

    var body: some View {
        TextField(placeholder, text: $text)
            .focused($isFocused)
            .foregroundColor(.black)
            .tint(.exampleBlue)
            .padding(10)
            .background(Color.exampleLightGray.cornerRadius(10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(isFocused ? Color.exampleBlue : Color.exampleFieldBorder, lineWidth: 1)
            )
    }
}
