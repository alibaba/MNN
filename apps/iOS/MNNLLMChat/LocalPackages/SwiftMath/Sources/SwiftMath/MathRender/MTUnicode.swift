//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by Kostub Deshmukh.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation

public struct UnicodeSymbol {
    static let multiplication              = "\u{00D7}"
    static let division                    = "\u{00F7}"
    static let fractionSlash               = "\u{2044}"
    static let whiteSquare                 = "\u{25A1}"
    static let blackSquare                 = "\u{25A0}"
    static let lessEqual                   = "\u{2264}"
    static let greaterEqual                = "\u{2265}"
    static let notEqual                    = "\u{2260}"
    static let squareRoot                  = "\u{221A}" // \sqrt
    static let cubeRoot                    = "\u{221B}"
    static let infinity                    = "\u{221E}" // \infty
    static let angle                       = "\u{2220}" // \angle
    static let degree                      = "\u{00B0}" // \circ
    
    static let capitalGreekStart           = UInt32(0x0391)
    static let capitalGreekEnd             = UInt32(0x03A9)
    static let lowerGreekStart             = UInt32(0x03B1)
    static let lowerGreekEnd               = UInt32(0x03C9)
    static let planksConstant              = UInt32(0x210e)
    static let lowerItalicStart            = UInt32(0x1D44E)
    static let capitalItalicStart          = UInt32(0x1D434)
    static let greekLowerItalicStart       = UInt32(0x1D6FC)
    static let greekCapitalItalicStart     = UInt32(0x1D6E2)
    static let greekSymbolItalicStart      = UInt32(0x1D716)
    
    static let mathCapitalBoldStart        = UInt32(0x1D400)
    static let mathLowerBoldStart          = UInt32(0x1D41A)
    static let greekCapitalBoldStart       = UInt32(0x1D6A8)
    static let greekLowerBoldStart         = UInt32(0x1D6C2)
    static let greekSymbolBoldStart        = UInt32(0x1D6DC)
    static let numberBoldStart             = UInt32(0x1D7CE)
    
    static let mathCapitalBoldItalicStart  = UInt32(0x1D468)
    static let mathLowerBoldItalicStart    = UInt32(0x1D482)
    static let greekCapitalBoldItalicStart = UInt32(0x1D71C)
    static let greekLowerBoldItalicStart   = UInt32(0x1D736)
    static let greekSymbolBoldItalicStart  = UInt32(0x1D750)
    
    static let mathCapitalScriptStart      = UInt32(0x1D49C)
    static let mathCapitalTTStart          = UInt32(0x1D670)
    static let mathLowerTTStart            = UInt32(0x1D68A)
    static let numberTTStart               = UInt32(0x1D7F6)
    static let mathCapitalSansSerifStart   = UInt32(0x1D5A0)
    static let mathLowerSansSerifStart     = UInt32(0x1D5BA)
    static let numberSansSerifStart        = UInt32(0x1D7E2)
    static let mathCapitalFrakturStart     = UInt32(0x1D504)
    static let mathLowerFrakturStart       = UInt32(0x1D51E)
    static let mathCapitalBlackboardStart  = UInt32(0x1D538)
    static let mathLowerBlackboardStart    = UInt32(0x1D552)
    static let numberBlackboardStart       = UInt32(0x1D7D8)
}

extension Character {
    
    var utf32Char: UTF32Char { self.unicodeScalars.map { $0.value }.reduce(0, +) }
    var isLowerEnglish : Bool { self >= "a" && self <= "z" }
    var isUpperEnglish : Bool { self >= "A" && self <= "Z" }
    var isNumber : Bool { self >= "0" && self <= "9" }

    var isLowerGreek : Bool {
        let uch = self.utf32Char
        return uch >= UnicodeSymbol.lowerGreekStart && uch <= UnicodeSymbol.lowerGreekEnd
    }

    var isCapitalGreek : Bool {
        let uch = self.utf32Char
        return uch >= UnicodeSymbol.capitalGreekStart && uch <= UnicodeSymbol.capitalGreekEnd
    }

    var greekSymbolOrder : UInt32? {
        let greekSymbols : [UTF32Char] = [0x03F5, 0x03D1, 0x03F0, 0x03D5, 0x03F1, 0x03D6]
        let index = greekSymbols.firstIndex(of: self.utf32Char)
        if let pos = index { return UInt32(pos) }
        return nil
    }

    var isGreekSymbol : Bool { self.greekSymbolOrder != nil }
}
