//
//  View+StatusBar.swift
//  
//
//  Created by Alisa Mylnikova on 02.06.2023.
//

import SwiftUI
import UIKit

public extension View {

    /// for this to work make sure all the other scrollViews have scrollsToTop = false
    func onStatusBarTap(onTap: @escaping () -> ()) -> some View {
        self.overlay {
            StatusBarTabDetector(onTap: onTap)
                .offset(x: UIScreen.main.bounds.width)
        }
    }
}

private struct StatusBarTabDetector: UIViewRepresentable {

    var onTap: () -> ()

    func makeUIView(context: Context) -> UIView {
        let fakeScrollView = UIScrollView()
        fakeScrollView.contentOffset = CGPoint(x: 0, y: 10)
        fakeScrollView.delegate = context.coordinator
        fakeScrollView.scrollsToTop = true
        fakeScrollView.contentSize = CGSize(width: 100, height: UIScreen.main.bounds.height * 2)
        return fakeScrollView
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onTap: onTap)
    }

    class Coordinator: NSObject, UIScrollViewDelegate {

        var onTap: () -> ()

        init(onTap: @escaping () -> ()) {
            self.onTap = onTap
        }

        func scrollViewShouldScrollToTop(_ scrollView: UIScrollView) -> Bool {
            onTap()
            return false
        }
    }
}
