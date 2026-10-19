//
//  OpacityDotsIndicatorView.swift
//  ActivityIndicatorView
//
//  Created by Daniil Manin on 10/7/20.
//  Copyright © 2020 Exyte. All rights reserved.
//

import SwiftUI

struct OpacityDotsIndicatorView: View {

    let count: Int
    let inset: Int

    var body: some View {
        GeometryReader { geometry in
            ForEach(0..<count, id: \.self) { index in
                OpacityDotsIndicatorItemView(index: index, count: count, inset: inset, size: geometry.size)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }
}

struct OpacityDotsIndicatorItemView: View {

    let index: Int
    let count: Int
    let inset: Int
    let size: CGSize

    @State private var scale: CGFloat = 0
    @State private var opacity: Double = 0

    var body: some View {
        let itemSize = (size.width - CGFloat(inset) * CGFloat(count - 1)) / CGFloat(count)

        let animation = Animation.easeOut
            .repeatForever(autoreverses: true)
            .delay(index % 2 == 0 ? 0.2 : 0)

        return Circle()
            .frame(width: itemSize, height: itemSize)
            .scaleEffect(scale)
            .opacity(opacity)
            .onAppear {
                scale = 1
                opacity = 1
                withAnimation(animation) {
                    scale = 0.9
                    opacity = 0.3
                }
            }
            .offset(x: (itemSize + CGFloat(inset)) * CGFloat(index) - size.width / 2 + itemSize / 2)
    }
}
