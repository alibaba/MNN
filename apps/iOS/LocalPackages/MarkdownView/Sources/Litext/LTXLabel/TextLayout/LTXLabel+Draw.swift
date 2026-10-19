//
//  LTXLabel+Draw.swift
//  Litext
//
//  Created by 秋星桥 on 3/27/25.
//

import UIKit

public extension LTXLabel {
    override func draw(_: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }
        UIGraphicsPushContext(context)
        textLayout.draw(in: context)
        UIGraphicsPopContext()
    }
}
