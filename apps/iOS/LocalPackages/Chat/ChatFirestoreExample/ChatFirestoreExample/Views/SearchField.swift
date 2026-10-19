//
//  SearchField.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 11.07.2023.
//

import SwiftUI

struct SearchField: View {

    @Binding var text: String

    var body: some View {
        ZStack {
            Color.exampleSearchField
                .cornerRadius(8)
            HStack {
                Image(.searchIcon)
                TextField("Search", text: $text)
                if !text.isEmpty {
                    Image(.searchCancel)
                        .onTapGesture {
                            text = ""
                        }
                }
            }
            .padding(.horizontal, 8)
        }
        .frame(height: 36)
    }
}
