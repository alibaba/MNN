//
//  RecordIndicator.swift
//  
//
//  Created by Alisa Mylnikova on 10.03.2023.
//

import SwiftUI

struct RecordIndicator: View {

    let count = 2

    var body: some View {
        let animation = Animation
            .linear(duration: 1)
            .repeatForever(autoreverses: false)

        ForEach(0..<count, id: \.self) { index in
            GrowingCircleIndicatorView(animation: animation.delay(CGFloat(index) * 0.5))
        }
    }
}

struct GrowingCircleIndicatorView: View {

    let animation: Animation

    @State private var scale: CGFloat = 0.6
    @State private var opacity: Double = 1

    var body: some View {

        return Circle()
            .scaleEffect(scale)
            .opacity(opacity)
            .onAppear {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.01) {
                    withAnimation(animation) {
                        scale = 1
                        opacity = 0
                    }
                }
            }
    }
}

