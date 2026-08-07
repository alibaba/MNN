//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by Kostub Deshmukh.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation
import CoreText

struct GlyphPart {
    /// The glyph that represents this part
    var glyph: CGGlyph!

    /// Full advance width/height for this part, in the direction of the extension in points.
    var fullAdvance: CGFloat = 0

    /// Advance width/ height of the straight bar connector material at the beginning of the glyph in points.
    var startConnectorLength: CGFloat = 0

    /// Advance width/ height of the straight bar connector material at the end of the glyph in points.
    var endConnectorLength: CGFloat = 0

    /// If this part is an extender. If set, the part can be skipped or repeated.
    var isExtender: Bool = false
}

/** This class represents the Math table of an open type font.
 
 The math table is documented here: https://www.microsoft.com/typography/otspec/math.htm
 
 How the constants in this class affect the display is documented here:
 http://www.tug.org/TUGboat/tb30-1/tb94vieth.pdf

 Note: We don't parse the math table from the open type font. Rather we parse it
 in python and convert it to a .plist file which is easily consumed by this class.
 This approach is preferable to spending an inordinate amount of time figuring out
 how to parse the returned NSData object using the open type rules.
 
 Remark: This class is not meant to be used outside of this library.
 */
class MTFontMathTable {
    
    // The font for this math table.
    public private(set) weak var font:MTFont? // @property (nonatomic, readonly, weak) MTFont* font;
    
    var _unitsPerEm: UInt
    var _fontSize: CGFloat
    var _mathTable: NSDictionary!
    
    let kConstants = "constants"
    
    /** MU unit in points */
    var muUnit:CGFloat { _fontSize/18 }
    
    func fontUnitsToPt(_ fontUnits:Int) -> CGFloat {
        CGFloat(fontUnits) * _fontSize / CGFloat(_unitsPerEm)
    }
    
    init(withFont font: MTFont?, mathTable:NSDictionary) {
        assert(font != nil, "font has nil value")
        assert(font!.ctFont != nil, "font.ctFont has nil value")
        self.font = font
        // do domething with font
        _unitsPerEm = UInt(CTFontGetUnitsPerEm(font!.ctFont))
        _fontSize = font!.fontSize;
        _mathTable = mathTable
        let version = _mathTable["version"] as! String
        if version != "1.3" {
            NSException(name: NSExceptionName.internalInconsistencyException, reason: "Invalid version of math table plist: \(version)").raise()
        }
    }
    
    func constantFromTable(_ constName:String) -> CGFloat {
        let consts = _mathTable[kConstants] as! NSDictionary?
        let val = consts![constName] as! NSNumber?
        return fontUnitsToPt(val!.intValue)
    }
    
    func percentFromTable(_ percentName:String) -> CGFloat {
        let consts = _mathTable[kConstants] as! NSDictionary?
        let val = consts![percentName] as! NSNumber?
        return CGFloat(val!.floatValue) / 100
    }
    
    /// Math Font Metrics from the opentype specification
    // MARK: - Fractions
    var fractionNumeratorDisplayStyleShiftUp:CGFloat { constantFromTable("FractionNumeratorDisplayStyleShiftUp") }          // \sigma_8 in TeX
    var fractionNumeratorShiftUp:CGFloat { constantFromTable("FractionNumeratorShiftUp") }                      // \sigma_9 in TeX
    var fractionDenominatorDisplayStyleShiftDown:CGFloat { constantFromTable("FractionDenominatorDisplayStyleShiftDown") }      // \sigma_11 in TeX
    var fractionDenominatorShiftDown:CGFloat { constantFromTable("FractionDenominatorShiftDown") }                  // \sigma_12 in TeX
    var fractionNumeratorDisplayStyleGapMin:CGFloat { constantFromTable("FractionNumDisplayStyleGapMin") }           // 3 * \xi_8 in TeX
    var fractionNumeratorGapMin:CGFloat { constantFromTable("FractionNumeratorGapMin") }                       // \xi_8 in TeX
    var fractionDenominatorDisplayStyleGapMin:CGFloat { constantFromTable("FractionDenomDisplayStyleGapMin") }         // 3 * \xi_8 in TeX
    var fractionDenominatorGapMin:CGFloat { constantFromTable("FractionDenominatorGapMin") }                     // \xi_8 in TeX
    var fractionRuleThickness:CGFloat { constantFromTable("FractionRuleThickness") }                         // \xi_8 in TeX
    var skewedFractionHorizonalGap:CGFloat { constantFromTable("SkewedFractionHorizontalGap") }             // \sigma_20 in TeX
    var skewedFractionVerticalGap:CGFloat { constantFromTable("SkewedFractionVerticalGap") }                         // \sigma_21 in TeX
    
    // MARK: - Non-standard
    /// FractionDelimiterSize and FractionDelimiterDisplayStyleSize are not constants
    /// specified in the OpenType Math specification. Rather these are proposed LuaTeX extensions
    /// for the TeX parameters \sigma_20 (delim1) and \sigma_21 (delim2). Since these do not
    /// exist in the fonts that we have, we use the same approach as LuaTeX and use the fontSize
    /// to determine these values. The constants used are the same as LuaTeX and KaTeX and match the
    /// metrics values of the original TeX fonts.
    /// Note: An alternative approach is to use DelimitedSubFormulaMinHeight for \sigma21 and use a factor
    /// of 2 to get \sigma 20 as proposed in Vieth paper.
    /// The XeTeX implementation sets \sigma21 = fontSize and \sigma20 = DelimitedSubFormulaMinHeight which
    /// will produce smaller delimiters.
    /// Of all the approaches we've implemented LuaTeX's approach since it mimics LaTeX most accurately.
    var fractionDelimiterSize: CGFloat { 1.01 * _fontSize }
    
    /// Modified constant from 2.4 to 2.39, it matches KaTeX and looks better.
    var fractionDelimiterDisplayStyleSize: CGFloat { 2.39 * _fontSize }

    // MARK: - Stacks
    var stackTopDisplayStyleShiftUp:CGFloat { constantFromTable("StackTopDisplayStyleShiftUp")  }                   // \sigma_8 in TeX
    var stackTopShiftUp:CGFloat { constantFromTable("StackTopShiftUp")  }                               // \sigma_10 in TeX
    var stackDisplayStyleGapMin:CGFloat { constantFromTable("StackDisplayStyleGapMin")  }                       // 7 \xi_8 in TeX
    var stackGapMin:CGFloat { constantFromTable("StackGapMin")  }                                   // 3 \xi_8 in TeX
    var stackBottomDisplayStyleShiftDown:CGFloat { constantFromTable("StackBottomDisplayStyleShiftDown")  }              // \sigma_11 in TeX
    var stackBottomShiftDown:CGFloat { constantFromTable("StackBottomShiftDown")  } // \sigma_12 in TeX

   var stretchStackBottomShiftDown:CGFloat { constantFromTable("StretchStackBottomShiftDown") }
   var stretchStackGapAboveMin:CGFloat { constantFromTable("StretchStackGapAboveMin") }
   var stretchStackGapBelowMin:CGFloat { constantFromTable("StretchStackGapBelowMin") }
   var stretchStackTopShiftUp:CGFloat { constantFromTable("StretchStackTopShiftUp") }
    
    // MARK: - super/sub scripts

    var superscriptShiftUp:CGFloat { constantFromTable("SuperscriptShiftUp")  }                            // \sigma_13, \sigma_14 in TeX
    var superscriptShiftUpCramped:CGFloat { constantFromTable("SuperscriptShiftUpCramped")  }                     // \sigma_15 in TeX
    var subscriptShiftDown:CGFloat { constantFromTable("SubscriptShiftDown")  }                            // \sigma_16, \sigma_17 in TeX
    var superscriptBaselineDropMax:CGFloat { constantFromTable("SuperscriptBaselineDropMax")  }                    // \sigma_18 in TeX
    var subscriptBaselineDropMin:CGFloat { constantFromTable("SubscriptBaselineDropMin")  }                      // \sigma_19 in TeX
    var superscriptBottomMin:CGFloat { constantFromTable("SuperscriptBottomMin")  }                          // 1/4 \sigma_5 in TeX
    var subscriptTopMax:CGFloat { constantFromTable("SubscriptTopMax")  }                               // 4/5 \sigma_5 in TeX
    var subSuperscriptGapMin:CGFloat { constantFromTable("SubSuperscriptGapMin")  }                          // 4 \xi_8 in TeX
    var superscriptBottomMaxWithSubscript:CGFloat { constantFromTable("SuperscriptBottomMaxWithSubscript")  }             // 4/5 \sigma_5 in TeX

    var spaceAfterScript:CGFloat { constantFromTable("SpaceAfterScript")  }

    // MARK: - radicals
    var radicalExtraAscender:CGFloat { constantFromTable("RadicalExtraAscender")  }                          // \xi_8 in Tex
    var radicalRuleThickness:CGFloat { constantFromTable("RadicalRuleThickness")  }                          // \xi_8 in Tex
    var radicalDisplayStyleVerticalGap:CGFloat { constantFromTable("RadicalDisplayStyleVerticalGap")  }                // \xi_8 + 1/4 \sigma_5 in Tex
    var radicalVerticalGap:CGFloat { constantFromTable("RadicalVerticalGap")  }                            // 5/4 \xi_8 in Tex
    var radicalKernBeforeDegree:CGFloat { constantFromTable("RadicalKernBeforeDegree")  }                       // 5 mu in Tex
    var radicalKernAfterDegree:CGFloat { constantFromTable("RadicalKernAfterDegree")  }                        // -10 mu in Tex
    var radicalDegreeBottomRaisePercent:CGFloat { percentFromTable("RadicalDegreeBottomRaisePercent")  }               // 60% in Tex

    // MARK: - Limits
    var upperLimitBaselineRiseMin:CGFloat { constantFromTable("UpperLimitBaselineRiseMin")  }                     // \xi_11 in TeX
    var upperLimitGapMin:CGFloat { constantFromTable("UpperLimitGapMin")  }                              // \xi_9 in TeX
    var lowerLimitGapMin:CGFloat { constantFromTable("LowerLimitGapMin")  }                              // \xi_10 in TeX
    var lowerLimitBaselineDropMin:CGFloat { constantFromTable("LowerLimitBaselineDropMin")  }                     // \xi_12 in TeX
    var limitExtraAscenderDescender:CGFloat { 0 }                   // \xi_13 in TeX, not present in OpenType so we always set it to 0.

    // MARK: - Underline
    var underbarVerticalGap:CGFloat { constantFromTable("UnderbarVerticalGap")  }                           // 3 \xi_8 in TeX
    var underbarRuleThickness:CGFloat { constantFromTable("UnderbarRuleThickness")  }                         // \xi_8 in TeX
    var underbarExtraDescender:CGFloat { constantFromTable("UnderbarExtraDescender")  }                        // \xi_8 in TeX

    // MARK: - Overline
    var overbarVerticalGap:CGFloat { constantFromTable("OverbarVerticalGap")  }                            // 3 \xi_8 in TeX
    var overbarRuleThickness:CGFloat { constantFromTable("OverbarRuleThickness")  }                          // \xi_8 in TeX
    var overbarExtraAscender:CGFloat { constantFromTable("OverbarExtraAscender")  }                          // \xi_8 in TeX

    // MARK: - Constants

    var axisHeight:CGFloat { constantFromTable("AxisHeight")  }                                    // \sigma_22 in TeX
    var scriptScaleDown:CGFloat { percentFromTable("ScriptPercentScaleDown")  }
    var scriptScriptScaleDown:CGFloat { percentFromTable("ScriptScriptPercentScaleDown")  }
    var mathLeading:CGFloat { constantFromTable("MathLeading")  }
    var delimitedSubFormulaMinHeight:CGFloat { constantFromTable("DelimitedSubFormulaMinHeight")  }

    // MARK: - Accent

    var accentBaseHeight:CGFloat { constantFromTable("AccentBaseHeight")  } // \fontdimen5 in TeX (x-height)
    var flattenedAccentBaseHeight:CGFloat { constantFromTable("FlattenedAccentBaseHeight")  }
    
    // MARK: - Variants
    
    let kVertVariants = "v_variants"
    let kHorizVariants = "h_variants"

    /** Returns an Array of all the vertical variants of the glyph if any. If
     there are no variants for the glyph, the array contains the given glyph. */
    func getVerticalVariantsForGlyph( _ glyph:CGGlyph) -> [NSNumber?] {
        let variants = _mathTable[kVertVariants] as! NSDictionary?
        return self.getVariantsForGlyph(glyph, inDictionary: variants!)
    }

    /** Returns an Array of all the horizontal variants of the glyph if any. If
     there are no variants for the glyph, the array contains the given glyph. */
    func getHorizontalVariantsForGlyph( _ glyph:CGGlyph) -> [NSNumber?] {
        let variants = _mathTable[kHorizVariants] as! NSDictionary
        return self.getVariantsForGlyph(glyph, inDictionary:variants)
    }
    
    func getVariantsForGlyph(_ glyph: CGGlyph, inDictionary variants:NSDictionary) -> [NSNumber?] {
        let glyphName = self.font!.get(nameForGlyph: glyph)
        let variantGlyphs = variants[glyphName] as! NSArray?
        var glyphArray = [NSNumber]()
        if variantGlyphs == nil || variantGlyphs?.count == 0 {
            // There are no extra variants, so just add the current glyph to it.
            let glyph = self.font!.get(glyphWithName: glyphName)
            glyphArray.append(NSNumber(value:glyph))
            return glyphArray
        }
        for gvn in variantGlyphs! {
            let glyphVariantName = gvn as! String?
            let variantGlyph = self.font?.get(glyphWithName: glyphVariantName!)
            glyphArray.append(NSNumber(value:variantGlyph!))
        }
        return glyphArray
    }

    /** Returns a larger vertical variant of the given glyph if any.
     If there is no larger version, this returns the current glyph.
     */
    func getLargerGlyph(_ glyph:CGGlyph) -> CGGlyph {
        let variants = _mathTable[kVertVariants] as! NSDictionary?
        let glyphName = self.font?.get(nameForGlyph: glyph)
        let variantGlyphs = variants![glyphName!] as! NSArray?
        if variantGlyphs == nil || variantGlyphs?.count == 0 {
            // There are no extra variants, so just returnt the current glyph.
            return glyph
        }
        // Find the first variant with a different name.
        for gvn in variantGlyphs! {
            let glyphVariantName = gvn as! String?
            if glyphVariantName != glyphName {
                let variantGlyph = self.font?.get(glyphWithName: glyphVariantName!)
                return variantGlyph!
            }
        }
        // We did not find any variants of this glyph so return it.
        return glyph;
    }

    // MARK: - Italic Correction
    
    let kItalic = "italic"

    /** Returns the italic correction for the given glyph if any. If there
     isn't any this returns 0. */
    func getItalicCorrection(_ glyph: CGGlyph) -> CGFloat {
        let italics = _mathTable[kItalic] as! NSDictionary?
        let glyphName = self.font?.get(nameForGlyph: glyph)
        let val = italics![glyphName!] as! NSNumber?
        // if val is nil, this returns 0.
        return self.fontUnitsToPt(val?.intValue ?? 0)
    }

    // MARK: - Accents
    
    let kAccents = "accents"

    /** Returns the adjustment to the top accent for the given glyph if any.
     If there isn't any this returns -1. */
    func getTopAccentAdjustment(_ glyph: CGGlyph) -> CGFloat {
        var glyph = glyph
        let accents = _mathTable[kAccents] as! NSDictionary?
        let glyphName = self.font?.get(nameForGlyph: glyph)
        let val = accents![glyphName!] as! NSNumber?
        if let val = val {
            return self.fontUnitsToPt(val.intValue)
        } else {
            // If no top accent is defined then it is the center of the advance width.
            var advances = CGSize.zero
            CTFontGetAdvancesForGlyphs(self.font!.ctFont, .horizontal, &glyph, &advances, 1)
            return advances.width/2
        }
    }

    // MARK: - Glyph Construction

    /** Minimum overlap of connecting glyphs during glyph construction */
    var minConnectorOverlap:CGFloat { constantFromTable("MinConnectorOverlap") }
    
    let kVertAssembly = "v_assembly"
    let kAssemblyParts = "parts"

    /** Returns an array of the glyph parts to be used for constructing vertical variants
     of this glyph. If there is no glyph assembly defined, returns an empty array. */
    func getVerticalGlyphAssembly(forGlyph glyph:CGGlyph) -> [GlyphPart] {
        let assemblyTable = _mathTable[kVertAssembly] as! NSDictionary?
        let glyphName = self.font?.get(nameForGlyph: glyph)
        let assemblyInfo = assemblyTable![glyphName!] as! NSDictionary?
        if assemblyInfo == nil {
            // No vertical assembly defined for glyph
            return []
        }
        let parts = assemblyInfo![kAssemblyParts] as! NSArray?
        if parts == nil {
            // parts should always have been defined, but if it isn't return nil
            return []
        }
        var rv = [GlyphPart]()
        for part in parts! {
            let partInfo = part as! NSDictionary?
            var part = GlyphPart()
            let adv = partInfo!["advance"] as! NSNumber?
            part.fullAdvance = self.fontUnitsToPt(adv!.intValue)
            let end = partInfo!["endConnector"] as! NSNumber?
            part.endConnectorLength = self.fontUnitsToPt(end!.intValue)
            let start = partInfo!["startConnector"] as! NSNumber?
            part.startConnectorLength = self.fontUnitsToPt(start!.intValue)
            let ext = partInfo!["extender"] as! NSNumber?
            part.isExtender = ext!.boolValue
            let glyphName = partInfo!["glyph"] as! String?
            part.glyph = self.font?.get(glyphWithName: glyphName!)
            rv.append(part)
        }
        return rv
    }

    
}
