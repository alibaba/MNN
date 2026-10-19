//
//  RenderedTextContent.swift
//  MarkdownView
//
//  Created by 秋星桥 on 6/3/25.
//

import UIKit

public struct RenderedTextContent {
    public let image: UIImage?
    public let text: String

    public typealias Map = [String: RenderedTextContent]

    public init(image: UIImage?, text: String) {
        self.image = image
        self.text = text
    }
}
