
//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by 安志钢.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation

#if os(macOS)

extension MTBezierPath {
    func addLine(to point: CGPoint) {
        self.line(to: point)
    }
}

extension MTView {
    
    var backgroundColor:MTColor? {
        get {
            MTColor(cgColor: self.layer?.backgroundColor ?? MTColor.clear.cgColor)
        }
        set {
            self.layer?.backgroundColor = MTColor.clear.cgColor
            self.wantsLayer = true
        }
    }
    
}

#endif

