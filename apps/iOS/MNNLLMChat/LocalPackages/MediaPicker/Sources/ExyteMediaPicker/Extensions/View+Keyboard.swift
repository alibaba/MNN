//
//  File.swift
//  
//
//  Created by Alisa Mylnikova on 04.09.2023.
//

import SwiftUI

extension View {
    func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
