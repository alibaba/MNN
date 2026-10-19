//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by 安志钢.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation
import SwiftUI

#if os(macOS)

public class MTLabel : NSTextField {
    
    init() {
        super.init(frame: .zero)
        self.stringValue = ""
        self.isBezeled = false
        self.drawsBackground = false
        self.isEditable = false
        self.isSelectable = false
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    // MARK: - Customized getter and setter methods for property text.
    var text:String? {
        get { super.stringValue }
        set { super.stringValue = newValue! }
    }
    
}

#endif
