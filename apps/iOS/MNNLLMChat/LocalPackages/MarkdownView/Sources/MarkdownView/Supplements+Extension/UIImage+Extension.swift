//
//  UIImage+Extension.swift
//  MarkdownView
//
//  Created by 秋星桥 on 8/1/25.
//

import UIKit

extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsImageRenderer(size: size).image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
