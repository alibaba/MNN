//
//  MTFontV2.swift
//
//
//  Created by Peter Tang on 15/9/2023.
//

import Foundation
import CoreGraphics
import CoreText

extension MathFont {
    public func mtfont(size: CGFloat) -> MTFontV2 {
        MTFontV2(font: self, size: size)
    }
}
public final class MTFontV2: MTFont {
    let font: MathFont
    let size: CGFloat
    private let _cgFont: CGFont
    private let _ctFont: CTFont
    private let unitsPerEm: UInt
    private var _mathTab: MTFontMathTableV2?
    init(font: MathFont = .latinModernFont, size: CGFloat) {
        self.font = font
        self.size = size
        // MathFont cgfont and ctfont are fast & threadsafe, keep a local copy is cheaper than
        // handling via NSLock
        self._cgFont = font.cgFont()
        self._ctFont = font.ctFont(withSize: size)
        self.unitsPerEm = self._ctFont.unitsPerEm
        super.init()
        
        super.defaultCGFont = nil
        super.ctFont = nil
        super.mathTable = nil
        super.rawMathTable = nil
    }
    override var defaultCGFont: CGFont! {
        set { fatalError("\(#function): change to \(font.fontName) not allowed.") }
        get { _cgFont }
    }
    override var ctFont: CTFont! {
        set { fatalError("\(#function): change to \(font.fontName) not allowed.") }
        get { _ctFont }
    }
    private let mtfontV2LockOnMathTable = NSLock()
    override var mathTable: MTFontMathTable? {
        set { fatalError("\(#function): change to \(font.rawValue) not allowed.") }
        get {
            guard _mathTab == nil else { return _mathTab }
            //Note: lazy _mathTab initialization is now threadsafe.
            mtfontV2LockOnMathTable.lock()
            defer { mtfontV2LockOnMathTable.unlock() }
            if _mathTab == nil {
                _mathTab = MTFontMathTableV2(mathFont: font, size: size, unitsPerEm: unitsPerEm)
            }
            return _mathTab
        }
    }
    override var rawMathTable: NSDictionary? {
        set { fatalError("\(#function): change to \(font.rawValue) not allowed.") }
        get { fatalError("\(#function): access to \(font.rawValue) not allowed.") }
    }
    public override func copy(withSize size: CGFloat) -> MTFont {
        MTFontV2(font: font, size: size)
    }
}
