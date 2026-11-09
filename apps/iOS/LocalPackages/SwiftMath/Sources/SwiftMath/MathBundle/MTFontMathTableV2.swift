//
//  MTFontMathTableV2.swift
//
//
//  Created by Peter Tang on 15/9/2023.
//

import Foundation
import CoreGraphics
import CoreText

internal class MTFontMathTableV2: MTFontMathTable {
    private let mathFont: MathFont
    private let fontSize: CGFloat
    private let unitsPerEm: UInt
    private let mTable: NSDictionary
    init(mathFont: MathFont, size: CGFloat, unitsPerEm: UInt) {
        self.mathFont = mathFont
        self.fontSize = size
        self.unitsPerEm = unitsPerEm
        mTable = mathFont.rawMathTable()
        super.init(withFont: mathFont.mtfont(size: fontSize), mathTable: mTable)
        super._mathTable = nil
        // disable all possible access to _mathTable in superclass!
    }
    override var _mathTable: NSDictionary? {
        set { fatalError("\(#function) change to _mathTable \(mathFont.rawValue) not allowed.") }
        get { mTable }
    }
    override var muUnit: CGFloat { fontSize/18 }
    
    override func fontUnitsToPt(_ fontUnits:Int) -> CGFloat {
        CGFloat(fontUnits) * fontSize / CGFloat(unitsPerEm)
    }
    override func constantFromTable(_ constName: String) -> CGFloat {
        guard let consts = mTable[kConstants] as? NSDictionary, let val = consts[constName] as? NSNumber else {
            return .zero
        }
        return fontUnitsToPt(val.intValue)
    }
    override func percentFromTable(_ percentName: String) -> CGFloat {
        guard let consts = mTable[kConstants] as? NSDictionary, let val = consts[percentName] as? NSNumber else {
            return .zero
        }
        return CGFloat(val.floatValue) / 100
    }
    /** Returns an Array of all the vertical variants of the glyph if any. If
     there are no variants for the glyph, the array contains the given glyph. */
    override func getVerticalVariantsForGlyph(_ glyph: CGGlyph) -> [NSNumber?] {
        guard let variants = mTable[kVertVariants] as? NSDictionary else { return [] }
        return self.getVariantsForGlyph(glyph, inDictionary: variants)
    }
    /** Returns an Array of all the horizontal variants of the glyph if any. If
     there are no variants for the glyph, the array contains the given glyph. */
    override func getHorizontalVariantsForGlyph(_ glyph: CGGlyph) -> [NSNumber?] {
        guard let variants = mTable[kHorizVariants] as? NSDictionary else { return [] }
        return self.getVariantsForGlyph(glyph, inDictionary:variants)
    }
    override func getVariantsForGlyph(_ glyph: CGGlyph, inDictionary variants: NSDictionary) -> [NSNumber?] {
        let font = mathFont.mtfont(size: fontSize)
        let glyphName = font.get(nameForGlyph: glyph)
        
        var glyphArray = [NSNumber]()
        let variantGlyphs = variants[glyphName] as? NSArray

        guard let variantGlyphs = variantGlyphs, variantGlyphs.count != .zero else {
            // There are no extra variants, so just add the current glyph to it.
            let glyph = font.get(glyphWithName: glyphName)
            glyphArray.append(NSNumber(value:glyph))
            return glyphArray
        }
        for gvn in variantGlyphs {
            if let glyphVariantName = gvn as? String {
                let variantGlyph = font.get(glyphWithName: glyphVariantName)
                glyphArray.append(NSNumber(value:variantGlyph))
            }
        }
        return glyphArray
    }
    /** Returns a larger vertical variant of the given glyph if any.
     If there is no larger version, this returns the current glyph.
     */
    override func getLargerGlyph(_ glyph: CGGlyph) -> CGGlyph {
        let font = mathFont.mtfont(size: fontSize)
        let glyphName = font.get(nameForGlyph: glyph)

        guard let variants = mTable[kVertVariants] as? NSDictionary,
                let variantGlyphs = variants[glyphName] as? NSArray, variantGlyphs.count != .zero else {
            // There are no extra variants, so just returnt the current glyph.
            return glyph
        }
        // Find the first variant with a different name.
        for gvn in variantGlyphs {
            if let glyphVariantName = gvn as? String, glyphVariantName != glyphName {
                let variantGlyph = font.get(glyphWithName: glyphVariantName)
                return variantGlyph
            }
        }
        // We did not find any variants of this glyph so return it.
        return glyph
    }
    /** Returns the italic correction for the given glyph if any. If there
     isn't any this returns 0. */
    override func getItalicCorrection(_ glyph: CGGlyph) -> CGFloat {
        let font = mathFont.mtfont(size: fontSize)
        let glyphName = font.get(nameForGlyph: glyph)

        guard let italics = mTable[kItalic] as? NSDictionary, let val = italics[glyphName] as? NSNumber else {
            return .zero
        }
        // if val is nil, this returns 0.
        return fontUnitsToPt(val.intValue)
    }
    override func getTopAccentAdjustment(_ glyph: CGGlyph) -> CGFloat {
        let font = mathFont.mtfont(size: fontSize)
        let glyphName = font.get(nameForGlyph: glyph)
        
        guard let accents = mTable[kAccents] as? NSDictionary, let val = accents[glyphName] as? NSNumber else {
            // If no top accent is defined then it is the center of the advance width.
            var glyph = glyph
            var advances = CGSize.zero
            CTFontGetAdvancesForGlyphs(font.ctFont, .horizontal, &glyph, &advances, 1)
            return advances.width/2
        }
        return fontUnitsToPt(val.intValue)
    }
    override func getVerticalGlyphAssembly(forGlyph glyph: CGGlyph) -> [GlyphPart] {
        let font = mathFont.mtfont(size: fontSize)
        let glyphName = font.get(nameForGlyph: glyph)
        
        guard let assemblyTable = mTable[kVertAssembly] as? NSDictionary,
              let assemblyInfo = assemblyTable[glyphName] as? NSDictionary,
              let parts = assemblyInfo[kAssemblyParts] as? NSArray else {
            // No vertical assembly defined for glyph
            // parts should always have been defined, but if it isn't return nil
            return []
        }
        
        var rv = [GlyphPart]()
        for part in parts {
            guard let partInfo = part as? NSDictionary,
                let adv = partInfo["advance"] as? NSNumber,
                let end = partInfo["endConnector"] as? NSNumber,
                let start = partInfo["startConnector"] as? NSNumber,
                let ext = partInfo["extender"] as? NSNumber,
                let glyphName = partInfo["glyph"] as? String else { continue }
            let fullAdvance = fontUnitsToPt(adv.intValue)
            let endConnectorLength = fontUnitsToPt(end.intValue)
            let startConnectorLength = fontUnitsToPt(start.intValue)
            let isExtender = ext.boolValue
            let glyph = font.get(glyphWithName: glyphName)
            let part = GlyphPart(glyph: glyph, fullAdvance: fullAdvance,
                                 startConnectorLength: startConnectorLength,
                                 endConnectorLength: endConnectorLength,
                                 isExtender: isExtender)
            rv.append(part)
        }
        return rv
    }
}
