//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by Kostub Deshmukh.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation
import CoreText

// MARK: - Inter Element Spacing

enum InterElementSpaceType : Int {
    case invalid = -1
    case none = 0
    case thin
    case nsThin    // Thin but not in script mode
    case nsMedium
    case nsThick
}

var interElementSpaceArray = [[InterElementSpaceType]]()
private let interElementLock = NSLock()

func getInterElementSpaces() -> [[InterElementSpaceType]] {
    if interElementSpaceArray.isEmpty {
        
        interElementLock.lock()
        defer { interElementLock.unlock() }
        guard interElementSpaceArray.isEmpty else { return interElementSpaceArray }
        
        interElementSpaceArray =
        //   ordinary   operator   binary     relation  open       close     punct     fraction
        [  [.none,     .thin,     .nsMedium, .nsThick, .none,     .none,    .none,    .nsThin],    // ordinary
           [.thin,     .thin,     .invalid,  .nsThick, .none,     .none,    .none,    .nsThin],    // operator
           [.nsMedium, .nsMedium, .invalid,  .invalid, .nsMedium, .invalid, .invalid, .nsMedium],  // binary
           [.nsThick,  .nsThick,  .invalid,  .none,    .nsThick,  .none,    .none,    .nsThick],   // relation
           [.none,     .none,     .invalid,  .none,    .none,     .none,    .none,    .none],      // open
           [.none,     .thin,     .nsMedium, .nsThick, .none,     .none,    .none,    .nsThin],    // close
           [.nsThin,   .nsThin,   .invalid,  .nsThin,  .nsThin,   .nsThin,  .nsThin,  .nsThin],    // punct
           [.nsThin,   .thin,     .nsMedium, .nsThick, .nsThin,   .none,    .nsThin,  .nsThin],    // fraction
           [.nsMedium, .nsThin,   .nsMedium, .nsThick, .none,     .none,    .none,    .nsThin]]    // radical
    }
    return interElementSpaceArray
}


// Get's the index for the given type. If row is true, the index is for the row (i.e. left element) otherwise it is for the column (right element)
func getInterElementSpaceArrayIndexForType(_ type:MTMathAtomType, row:Bool) -> Int {
    switch type {
        case .color, .textcolor, .colorBox, .ordinary, .placeholder:   // A placeholder is treated as ordinary
            return 0
        case .largeOperator:
            return 1
        case .binaryOperator:
            return 2;
        case .relation:
            return 3;
        case .open:
            return 4;
        case .close:
            return 5;
        case .punctuation:
            return 6;
        case .fraction,  // Fraction and inner are treated the same.
             .inner:
            return 7;
        case .radical:
            if row {
                // Radicals have inter element spaces only when on the left side.
                // Note: This is a departure from latex but we don't want \sqrt{4}4 to look weird so we put a space in between.
                // They have the same spacing as ordinary except with ordinary.
                return 8;
            } else {
                assert(false, "Interelement space undefined for radical on the right. Treat radical as ordinary.")
                return Int.max
            }
        default:
            assert(false, "Interelement space undefined for type \(type)")
            return Int.max
    }
}

// MARK: - Italics
// mathit
func getItalicized(_ ch:Character) -> UTF32Char  {
    var unicode = ch.utf32Char
    
    // Special cases for italics
    if ch == "h" { return UnicodeSymbol.planksConstant }
    
    if ch.isUpperEnglish {
        unicode = UnicodeSymbol.capitalItalicStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        unicode = UnicodeSymbol.lowerItalicStart + (ch.utf32Char - Character("a").utf32Char)
    } else if ch.isCapitalGreek {
        // Capital Greek characters
        unicode = UnicodeSymbol.greekCapitalItalicStart + (ch.utf32Char - UnicodeSymbol.capitalGreekStart)
    } else if ch.isLowerGreek {
        // Greek characters
        unicode = UnicodeSymbol.greekLowerItalicStart + (ch.utf32Char - UnicodeSymbol.lowerGreekStart)
    } else if ch.isGreekSymbol {
        return UnicodeSymbol.greekSymbolItalicStart + ch.greekSymbolOrder!
    }
    // Note there are no italicized numbers in unicode so we don't support italicizing numbers.
    return unicode
}

// mathbf
func getBold(_ ch:Character) -> UTF32Char {
    var unicode = ch.utf32Char
    if ch.isUpperEnglish {
        unicode = UnicodeSymbol.mathCapitalBoldStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        unicode = UnicodeSymbol.mathLowerBoldStart + (ch.utf32Char - Character("a").utf32Char)
    } else if ch.isCapitalGreek {
        // Capital Greek characters
        unicode = UnicodeSymbol.greekCapitalBoldStart + (ch.utf32Char - UnicodeSymbol.capitalGreekStart);
    } else if ch.isLowerGreek {
        // Greek characters
        unicode = UnicodeSymbol.greekLowerBoldStart + (ch.utf32Char - UnicodeSymbol.lowerGreekStart);
    } else if ch.isGreekSymbol {
        return UnicodeSymbol.greekSymbolBoldStart + ch.greekSymbolOrder!
    } else if ch.isNumber {
        unicode = UnicodeSymbol.numberBoldStart + (ch.utf32Char - Character("0").utf32Char)
    }
    return unicode
}

// mathbfit
func getBoldItalic(_ ch:Character) -> UTF32Char {
    var unicode = ch.utf32Char
    if ch.isUpperEnglish {
        unicode = UnicodeSymbol.mathCapitalBoldItalicStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        unicode = UnicodeSymbol.mathLowerBoldItalicStart + (ch.utf32Char - Character("a").utf32Char)
    } else if ch.isCapitalGreek {
        // Capital Greek characters
        unicode = UnicodeSymbol.greekCapitalBoldItalicStart + (ch.utf32Char - UnicodeSymbol.capitalGreekStart);
    } else if ch.isLowerGreek {
        // Greek characters
        unicode = UnicodeSymbol.greekLowerBoldItalicStart + (ch.utf32Char - UnicodeSymbol.lowerGreekStart);
    } else if ch.isGreekSymbol {
        return UnicodeSymbol.greekSymbolBoldItalicStart + ch.greekSymbolOrder!
    } else if ch.isNumber {
        // No bold italic for numbers so we just bold them.
        unicode = getBold(ch);
    }
    return unicode;
}

// LaTeX default
func getDefaultStyle(_ ch:Character) -> UTF32Char {
    if ch.isLowerEnglish || ch.isUpperEnglish || ch.isLowerGreek || ch.isGreekSymbol {
        return getItalicized(ch);
    } else if ch.isNumber || ch.isCapitalGreek {
        // In the default style numbers and capital greek is roman
        return ch.utf32Char
    } else if ch == "." {
        // . is treated as a number in our code, but it doesn't change fonts.
        return ch.utf32Char
    } else {
        NSException(name: NSExceptionName("IllegalCharacter"), reason: "Unknown character \(ch) for default style.").raise()
    }
    return ch.utf32Char
}

// mathcal/mathscr (caligraphic or script)
func getCaligraphic(_ ch:Character) -> UTF32Char {
    // Caligraphic has lots of exceptions:
    switch ch {
        case "B":
            return 0x212C;   // Script B (bernoulli)
        case "E":
            return 0x2130;   // Script E (emf)
        case "F":
            return 0x2131;   // Script F (fourier)
        case "H":
            return 0x210B;   // Script H (hamiltonian)
        case "I":
            return 0x2110;   // Script I
        case "L":
            return 0x2112;   // Script L (laplace)
        case "M":
            return 0x2133;   // Script M (M-matrix)
        case "R":
            return 0x211B;   // Script R (Riemann integral)
        case "e":
            return 0x212F;   // Script e (Natural exponent)
        case "g":
            return 0x210A;   // Script g (real number)
        case "o":
            return 0x2134;   // Script o (order)
        default:
            break;
    }
    var unicode:UTF32Char
    if ch.isUpperEnglish {
        unicode = UnicodeSymbol.mathCapitalScriptStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        // Latin Modern Math does not have lower case caligraphic characters, so we use
        // the default style instead of showing a ?
        unicode = getDefaultStyle(ch)
    } else {
        // Caligraphic characters don't exist for greek or numbers, we give them the
        // default treatment.
        unicode = getDefaultStyle(ch)
    }
    return unicode;
}

// mathtt (monospace)
func getTypewriter(_ ch:Character) -> UTF32Char {
    if ch.isUpperEnglish {
        return UnicodeSymbol.mathCapitalTTStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        return UnicodeSymbol.mathLowerTTStart + (ch.utf32Char - Character("a").utf32Char)
    } else if ch.isNumber {
        return UnicodeSymbol.numberTTStart + (ch.utf32Char - Character("0").utf32Char)
    }
    // Monospace characters don't exist for greek, we give them the
    // default treatment.
    return getDefaultStyle(ch);
}

// mathsf
func getSansSerif(_ ch:Character) -> UTF32Char {
    if ch.isUpperEnglish {
        return UnicodeSymbol.mathCapitalSansSerifStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        return UnicodeSymbol.mathLowerSansSerifStart + (ch.utf32Char - Character("a").utf32Char)
    } else if ch.isNumber {
        return UnicodeSymbol.numberSansSerifStart + (ch.utf32Char - Character("0").utf32Char)
    }
    // Sans-serif characters don't exist for greek, we give them the
    // default treatment.
    return getDefaultStyle(ch);
}

// mathfrak
func getFraktur(_ ch:Character) -> UTF32Char {
    // Fraktur has exceptions:
    switch ch {
        case "C":
            return 0x212D;   // C Fraktur
        case "H":
            return 0x210C;   // Hilbert space
        case "I":
            return 0x2111;   // Imaginary
        case "R":
            return 0x211C;   // Real
        case "Z":
            return 0x2128;   // Z Fraktur
        default:
            break;
    }
    if ch.isUpperEnglish {
        return UnicodeSymbol.mathCapitalFrakturStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        return UnicodeSymbol.mathLowerFrakturStart + (ch.utf32Char - Character("a").utf32Char)
    }
    // Fraktur characters don't exist for greek & numbers, we give them the
    // default treatment.
    return getDefaultStyle(ch);
}

// mathbb (double struck)
func getBlackboard(_ ch:Character) -> UTF32Char {
    // Blackboard has lots of exceptions:
    switch(ch) {
        case "C":
            return 0x2102;   // Complex numbers
        case "H":
            return 0x210D;   // Quarternions
        case "N":
            return 0x2115;   // Natural numbers
        case "P":
            return 0x2119;   // Primes
        case "Q":
            return 0x211A;   // Rationals
        case "R":
            return 0x211D;   // Reals
        case "Z":
            return 0x2124;   // Integers
        default:
            break;
    }
    if ch.isUpperEnglish {
        return UnicodeSymbol.mathCapitalBlackboardStart + (ch.utf32Char - Character("A").utf32Char)
    } else if ch.isLowerEnglish {
        return UnicodeSymbol.mathLowerBlackboardStart + (ch.utf32Char - Character("a").utf32Char)
    } else if ch.isNumber {
        return UnicodeSymbol.numberBlackboardStart + (ch.utf32Char - Character("0").utf32Char)
    }
    // Blackboard characters don't exist for greek, we give them the
    // default treatment.
    return getDefaultStyle(ch);
}

func styleCharacter(_ ch:Character, fontStyle:MTFontStyle) -> UTF32Char {
    switch fontStyle {
        case .defaultStyle:
            return getDefaultStyle(ch);
        case .roman:
            return ch.utf32Char
        case .bold:
            return getBold(ch);
        case .italic:
            return getItalicized(ch);
        case .boldItalic:
            return getBoldItalic(ch);
        case .caligraphic:
            return getCaligraphic(ch);
        case .typewriter:
            return getTypewriter(ch);
        case .sansSerif:
            return getSansSerif(ch);
        case .fraktur:
            return getFraktur(ch);
        case .blackboard:
            return getBlackboard(ch);
    }
}

func changeFont(_ str:String, fontStyle:MTFontStyle) -> String {
    var retval = ""
    let codes = Array(str)
    for i in 0..<str.count {
        let ch = codes[i]
        var unicode = styleCharacter(ch, fontStyle: fontStyle);
        unicode = NSSwapHostIntToLittle(unicode)
        let charStr = String(UnicodeScalar(unicode)!)
        retval.append(charStr)
    }
    return retval
}

func getBboxDetails(_ bbox:CGRect, ascent:inout CGFloat, descent:inout CGFloat) {
    ascent = max(0, CGRectGetMaxY(bbox) - 0)
    
    // Descent is how much the line goes below the origin. However if the line is all above the origin, then descent can't be negative.
    descent = max(0, 0 - CGRectGetMinY(bbox))
}

// MARK: - MTTypesetter

class MTTypesetter {
    var font:MTFont!
    var displayAtoms = [MTDisplay]()
    var currentPosition = CGPoint.zero
    var currentLine:NSMutableAttributedString!
    var currentAtoms = [MTMathAtom]()   // List of atoms that make the line
    var currentLineIndexRange = NSMakeRange(0, 0)
    var style:MTLineStyle { didSet { _styleFont = nil } }
    private var _styleFont:MTFont?
    var styleFont:MTFont {
        if _styleFont == nil {
            _styleFont = font.copy(withSize: Self.getStyleSize(style, font: font))
        }
        return _styleFont!
    }
    var cramped = false
    var spaced = false
    
    static func createLineForMathList(_ mathList:MTMathList?, font:MTFont?, style:MTLineStyle) -> MTMathListDisplay? {
        let finalizedList = mathList?.finalized
        // default is not cramped
        return self.createLineForMathList(finalizedList, font:font, style:style, cramped:false)
    }
    
    // Internal
    static func createLineForMathList(_ mathList:MTMathList?, font:MTFont?, style:MTLineStyle, cramped:Bool) -> MTMathListDisplay? {
        return self.createLineForMathList(mathList, font:font, style:style, cramped:cramped, spaced:false)
    }
    
    // Internal
    static func createLineForMathList(_ mathList:MTMathList?, font:MTFont?, style:MTLineStyle, cramped:Bool, spaced:Bool) -> MTMathListDisplay? {
        assert(font != nil)
        let preprocessedAtoms = self.preprocessMathList(mathList)
        let typesetter = MTTypesetter(withFont:font, style:style, cramped:cramped, spaced:spaced)
        typesetter.createDisplayAtoms(preprocessedAtoms)
        let lastAtom = mathList!.atoms.last
        let last = lastAtom?.indexRange ?? NSMakeRange(0, 0)
        let line = MTMathListDisplay(withDisplays: typesetter.displayAtoms, range: NSMakeRange(0, NSMaxRange(last)))
        return line
    }
    
    static var placeholderColor: MTColor { MTColor.blue }
    
    init(withFont font:MTFont?, style:MTLineStyle, cramped:Bool, spaced:Bool) {
        self.font = font
        self.displayAtoms = [MTDisplay]()
        self.currentPosition = CGPoint.zero
        self.cramped = cramped
        self.spaced = spaced
        self.currentLine = NSMutableAttributedString()
        self.currentAtoms = [MTMathAtom]()
        self.style = style
        self.currentLineIndexRange = NSMakeRange(NSNotFound, NSNotFound);
    }
    
    static func preprocessMathList(_ ml:MTMathList?) -> [MTMathAtom] {
        // Note: Some of the preprocessing described by the TeX algorithm is done in the finalize method of MTMathList.
        // Specifically rules 5 & 6 in Appendix G are handled by finalize.
        // This function does not do a complete preprocessing as specified by TeX either. It removes any special atom types
        // that are not included in TeX and applies Rule 14 to merge ordinary characters.
        var preprocessed = [MTMathAtom]() //  arrayWithCapacity:ml.atoms.count)
        var prevNode:MTMathAtom! = nil
        preprocessed.reserveCapacity(ml!.atoms.count)
        for atom in ml!.atoms {
            if atom.type == .variable || atom.type == .number {
                // This is not a TeX type node. TeX does this during parsing the input.
                // switch to using the italic math font
                // We convert it to ordinary
                let newFont = changeFont(atom.nucleus, fontStyle: atom.fontStyle) // mathItalicize(atom.nucleus)
                atom.type = .ordinary
                atom.nucleus = newFont
            } else if atom.type == .unaryOperator {
                // Neither of these are TeX nodes. TeX treats these as Ordinary. So will we.
                atom.type = .ordinary
            }
            
            if atom.type == .ordinary {
                // This is Rule 14 to merge ordinary characters.
                // combine ordinary atoms together
                if prevNode != nil && prevNode.type == .ordinary && prevNode.subScript == nil && prevNode.superScript == nil {
                    prevNode.fuse(with: atom)
                    // skip the current node, we are done here.
                    continue
                }
            }
            
            // TODO: add italic correction here or in second pass?
            prevNode = atom
            preprocessed.append(atom)
        }
        return preprocessed
    }
    
    // returns the size of the font in this style
    static func getStyleSize(_ style:MTLineStyle, font:MTFont?) -> CGFloat {
        let original = font!.fontSize
        switch style {
            case .display, .text:
                return original
            case .script:
                return original * font!.mathTable!.scriptScaleDown
            case .scriptOfScript:
                return original * font!.mathTable!.scriptScriptScaleDown
        }
    }
    
    func addInterElementSpace(_ prevNode:MTMathAtom?, currentType type:MTMathAtomType) {
        var interElementSpace = CGFloat(0)
        if prevNode != nil {
            interElementSpace = getInterElementSpace(prevNode!.type, right:type)
        } else if self.spaced {
            // For the first atom of a spaced list, treat it as if it is preceded by an open.
            interElementSpace = getInterElementSpace(.open, right:type)
        }
        self.currentPosition.x += interElementSpace
    }
    
    func createDisplayAtoms(_ preprocessed:[MTMathAtom]) {
        // items should contain all the nodes that need to be layed out.
        // convert to a list of DisplayAtoms
        var prevNode:MTMathAtom? = nil
        var lastType:MTMathAtomType!
        for atom in preprocessed {
            switch atom.type {
                case .number, .variable,. unaryOperator:
                    // These should never appear as they should have been removed by preprocessing
                    assertionFailure("These types should never show here as they are removed by preprocessing.")
                    
                case .boundary:
                    assertionFailure("A boundary atom should never be inside a mathlist.")
                    
                case .space:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let space = atom as! MTMathSpace
                    // add the desired space
                    currentPosition.x += space.space * styleFont.mathTable!.muUnit;
                    // Since this is extra space, the desired interelement space between the prevAtom
                    // and the next node is still preserved. To avoid resetting the prevAtom and lastType
                    // we skip to the next node.
                    continue
                    
                case .style:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let style = atom as! MTMathStyle
                    self.style = style.style
                    // We need to preserve the prevNode for any interelement space changes.
                    // so we skip to the next node.
                    continue
                    
                case .color:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let colorAtom = atom as! MTMathColor
                    let display = MTTypesetter.createLineForMathList(colorAtom.innerList, font: font, style: style)
                    display!.localTextColor = MTColor(fromHexString: colorAtom.colorString)
                    display!.position = currentPosition
                    currentPosition.x += display!.width
                    displayAtoms.append(display!)

                case .textcolor:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let colorAtom = atom as! MTMathTextColor
                    let display = MTTypesetter.createLineForMathList(colorAtom.innerList, font: font, style: style)
                    display!.localTextColor = MTColor(fromHexString: colorAtom.colorString)

                    if prevNode != nil {
                        let subDisplay: MTDisplay = display!.subDisplays[0]
                        let subDisplayAtom = (subDisplay as? MTCTLineDisplay)!.atoms[0]
                        let interElementSpace = self.getInterElementSpace(prevNode!.type, right:subDisplayAtom.type)
                        if currentLine.length > 0 {
                            if interElementSpace > 0 {
                                // add a kerning of that space to the previous character
                                currentLine.addAttribute(kCTKernAttributeName as NSAttributedString.Key,
                                                         value:NSNumber(floatLiteral: interElementSpace),
                                                         range:currentLine.mutableString.rangeOfComposedCharacterSequence(at: currentLine.length-1))
                            }
                        } else {
                            // increase the space
                            currentPosition.x += interElementSpace
                        }
                    }

                    display!.position = currentPosition
                    currentPosition.x += display!.width
                    displayAtoms.append(display!)

                case .colorBox:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let colorboxAtom =  atom as! MTMathColorbox
                    let display = MTTypesetter.createLineForMathList(colorboxAtom.innerList, font:font, style:style)
                    
                    display!.localBackgroundColor = MTColor(fromHexString: colorboxAtom.colorString)
                    display!.position = currentPosition
                    currentPosition.x += display!.width;
                    displayAtoms.append(display!)
                    
                case .radical:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let rad = atom as! MTRadical
                    // Radicals are considered as Ord in rule 16.
                    self.addInterElementSpace(prevNode, currentType:.ordinary)
                    let displayRad = self.makeRadical(rad.radicand, range:rad.indexRange)
                    if rad.degree != nil {
                        // add the degree to the radical
                        let degree = MTTypesetter.createLineForMathList(rad.degree, font:font, style:.scriptOfScript)
                        displayRad!.setDegree(degree, fontMetrics:styleFont.mathTable)
                    }
                    displayAtoms.append(displayRad!)
                    currentPosition.x += displayRad!.width
                    
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        self.makeScripts(atom, display:displayRad, index:UInt(rad.indexRange.location), delta:0)
                    }
                    // change type to ordinary
                    //atom.type = .ordinary;
                    
                case .fraction:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    let frac = atom as! MTFraction?
                    self.addInterElementSpace(prevNode, currentType:atom.type)
                    let display = self.makeFraction(frac)
                    displayAtoms.append(display!)
                    currentPosition.x += display!.width;
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        self.makeScripts(atom, display:display, index:UInt(frac!.indexRange.location), delta:0)
                    }
                    
                case .largeOperator:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    self.addInterElementSpace(prevNode, currentType:atom.type)
                    let op = atom as! MTLargeOperator?
                    let display = self.makeLargeOp(op)
                    displayAtoms.append(display!)
                    
                case .inner:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    self.addInterElementSpace(prevNode, currentType:atom.type)
                    let inner =  atom as! MTInner?
                    var display : MTDisplay? = nil
                    if inner!.leftBoundary != nil || inner!.rightBoundary != nil {
                        display = self.makeLeftRight(inner)
                    } else {
                        display = MTTypesetter.createLineForMathList(inner!.innerList, font:font, style:style, cramped:cramped)
                    }
                    display!.position = currentPosition
                    currentPosition.x += display!.width
                    displayAtoms.append(display!)
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        self.makeScripts(atom, display:display, index:UInt(atom.indexRange.location), delta:0)
                    }
                    
                case .underline:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    // Underline is considered as Ord in rule 16.
                    self.addInterElementSpace(prevNode, currentType:.ordinary)
                    atom.type = .ordinary;
                    
                    let under = atom as! MTUnderLine?
                    let display = self.makeUnderline(under)
                    displayAtoms.append(display!)
                    currentPosition.x += display!.width;
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        self.makeScripts(atom, display:display, index:UInt(atom.indexRange.location), delta:0)
                    }
                    
                case .overline:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    // Overline is considered as Ord in rule 16.
                    self.addInterElementSpace(prevNode, currentType:.ordinary)
                    atom.type = .ordinary;
                    
                    let over = atom as! MTOverLine?
                    let display = self.makeOverline(over)
                    displayAtoms.append(display!)
                    currentPosition.x += display!.width;
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        self.makeScripts(atom, display:display, index:UInt(atom.indexRange.location), delta:0)
                    }
                    
                case .accent:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    // Accent is considered as Ord in rule 16.
                    self.addInterElementSpace(prevNode, currentType:.ordinary)
                    atom.type = .ordinary;
                    
                    let accent = atom as! MTAccent?
                    let display = self.makeAccent(accent)
                    displayAtoms.append(display!)
                    currentPosition.x += display!.width;
                    
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        self.makeScripts(atom, display:display, index:UInt(atom.indexRange.location), delta:0)
                    }
                    
                case .table:
                    // stash the existing layout
                    if currentLine.length > 0 {
                        self.addDisplayLine()
                    }
                    // We will consider tables as inner
                    self.addInterElementSpace(prevNode, currentType:.inner)
                    atom.type = .inner;
                    
                    let table = atom as! MTMathTable?
                    let display = self.makeTable(table)
                    displayAtoms.append(display!)
                    currentPosition.x += display!.width
                    // A table doesn't have subscripts or superscripts
                    
                case .ordinary, .binaryOperator, .relation, .open, .close, .placeholder, .punctuation:
                    // the rendering for all the rest is pretty similar
                    // All we need is render the character and set the interelement space.
                    if prevNode != nil {
                        let interElementSpace = self.getInterElementSpace(prevNode!.type, right:atom.type)
                        if currentLine.length > 0 {
                            if interElementSpace > 0 {
                                // add a kerning of that space to the previous character
                                currentLine.addAttribute(kCTKernAttributeName as NSAttributedString.Key,
                                                         value:NSNumber(floatLiteral: interElementSpace),
                                                         range:currentLine.mutableString.rangeOfComposedCharacterSequence(at: currentLine.length-1))
                            }
                        } else {
                            // increase the space
                            currentPosition.x += interElementSpace
                        }
                    }
                    var current:NSAttributedString? = nil
                    if atom.type == .placeholder {
                        let color = MTTypesetter.placeholderColor
                        current = NSAttributedString(string:atom.nucleus,
                                                     attributes:[kCTForegroundColorAttributeName as NSAttributedString.Key : color.cgColor])
                    } else {
                        current = NSAttributedString(string:atom.nucleus)
                    }
                    currentLine.append(current!)
                    // add the atom to the current range
                    if currentLineIndexRange.location == NSNotFound {
                        currentLineIndexRange = atom.indexRange
                    } else {
                        currentLineIndexRange.length += atom.indexRange.length
                    }
                    // add the fused atoms
                    if !atom.fusedAtoms.isEmpty {
                        currentAtoms.append(contentsOf: atom.fusedAtoms)  //.addObjectsFromArray:atom.fusedAtoms)
                    } else {
                        currentAtoms.append(atom)
                    }
                    
                    // add super scripts || subscripts
                    if atom.subScript != nil || atom.superScript != nil {
                        // stash the existing line
                        // We don't check currentLine.length here since we want to allow empty lines with super/sub scripts.
                        let line = self.addDisplayLine()
                        var delta = CGFloat(0)
                        if !atom.nucleus.isEmpty {
                            // Use the italic correction of the last character.
                            let index = atom.nucleus.index(before: atom.nucleus.endIndex)
                            let glyph = self.findGlyphForCharacterAtIndex(index, inString:atom.nucleus)
                            delta = styleFont.mathTable!.getItalicCorrection(glyph)
                        }
                        if delta > 0 && atom.subScript == nil {
                            // Add a kern of delta
                            currentPosition.x += delta;
                        }
                        self.makeScripts(atom, display:line, index:UInt(NSMaxRange(atom.indexRange) - 1), delta:delta)
                    }
            } // switch
            lastType = atom.type
            prevNode = atom
        } // node loop
        if currentLine.length > 0 {
            self.addDisplayLine()
        }
        if spaced && lastType != nil {
            // If spaced then add an interelement space between the last type and close
            let display = displayAtoms.last
            let interElementSpace = self.getInterElementSpace(lastType, right:.close)
            display?.width += interElementSpace
        }
    }
    
    @discardableResult
    func addDisplayLine() -> MTCTLineDisplay? {
        // add the font
        currentLine.addAttribute(kCTFontAttributeName as NSAttributedString.Key, value:styleFont.ctFont as Any, range:NSMakeRange(0, currentLine.length))
        /*assert(currentLineIndexRange.length == numCodePoints(currentLine.string),
         "The length of the current line: %@ does not match the length of the range (%d, %d)",
         currentLine, currentLineIndexRange.location, currentLineIndexRange.length);*/
        
        let displayAtom = MTCTLineDisplay(withString:currentLine, position:currentPosition, range:currentLineIndexRange, font:styleFont, atoms:currentAtoms)
        self.displayAtoms.append(displayAtom)
        // update the position
        currentPosition.x += displayAtom.width;
        // clear the string and the range
        currentLine = NSMutableAttributedString()
        currentAtoms = [MTMathAtom]()
        currentLineIndexRange = NSMakeRange(NSNotFound, NSNotFound)
        return displayAtom
    }
    
    // MARK: - Spacing
    
    // Returned in units of mu = 1/18 em.
    func getSpacingInMu(_ type: InterElementSpaceType) -> Int {
        // let valid = [MTLineStyle.display, .text]
        switch type {
            case .invalid:  return -1
            case .none:     return 0
            case .thin:     return 3
            case .nsThin:   return style.isNotScript ? 3 : 0;
            case .nsMedium: return style.isNotScript ? 4 : 0;
            case .nsThick:  return style.isNotScript ? 5 : 0;
        }
    }
    
    func getInterElementSpace(_ left: MTMathAtomType, right:MTMathAtomType) -> CGFloat {
        let leftIndex = getInterElementSpaceArrayIndexForType(left, row: true)
        let rightIndex = getInterElementSpaceArrayIndexForType(right, row: false)
        let spaceArray = getInterElementSpaces()[Int(leftIndex)]
        let spaceTypeObj = spaceArray[Int(rightIndex)]
        let spaceType = spaceTypeObj
        assert(spaceType != .invalid, "Invalid space between \(left) and \(right)")
        
        let spaceMultipler = self.getSpacingInMu(spaceType)
        if spaceMultipler > 0 {
            // 1 em = size of font in pt. space multipler is in multiples mu or 1/18 em
            return CGFloat(spaceMultipler) * styleFont.mathTable!.muUnit
        }
        return 0
    }
    
    
    // MARK: - Subscript/Superscript
    
    func scriptStyle() -> MTLineStyle {
        switch style {
            case .display, .text:          return .script
            case .script, .scriptOfScript: return .scriptOfScript
        }
    }
    
    // subscript is always cramped
    func subscriptCramped() -> Bool { true }
    
    // superscript is cramped only if the current style is cramped
    func superScriptCramped() -> Bool { cramped }
    
    func superScriptShiftUp() -> CGFloat {
        if cramped {
            return styleFont.mathTable!.superscriptShiftUpCramped;
        } else {
            return styleFont.mathTable!.superscriptShiftUp;
        }
    }
    
    // make scripts for the last atom
    // index is the index of the element which is getting the sub/super scripts.
    func makeScripts(_ atom: MTMathAtom?, display:MTDisplay?, index:UInt, delta:CGFloat) {
        assert(atom!.subScript != nil || atom!.superScript != nil)
        
        var superScriptShiftUp = 0.0
        var subscriptShiftDown = 0.0
        
        display?.hasScript = true
        if !(display is MTCTLineDisplay) {
            // get the font in script style
            let scriptFontSize = Self.getStyleSize(self.scriptStyle(), font:font)
            let scriptFont = font.copy(withSize: scriptFontSize)
            let scriptFontMetrics = scriptFont.mathTable
            
            // if it is not a simple line then
            superScriptShiftUp = display!.ascent - scriptFontMetrics!.superscriptBaselineDropMax
            subscriptShiftDown = display!.descent + scriptFontMetrics!.subscriptBaselineDropMin
        }
        
        if atom!.superScript == nil {
            assert(atom!.subScript != nil)
            let _subscript = MTTypesetter.createLineForMathList(atom!.subScript, font:font, style:self.scriptStyle(), cramped:self.subscriptCramped())
            _subscript?.type = .ssubscript
            _subscript?.index = Int(index)
            
            subscriptShiftDown = fmax(subscriptShiftDown, styleFont.mathTable!.subscriptShiftDown);
            subscriptShiftDown = fmax(subscriptShiftDown, _subscript!.ascent - styleFont.mathTable!.subscriptTopMax);
            // add the subscript
            _subscript?.position = CGPointMake(currentPosition.x, currentPosition.y - subscriptShiftDown);
            displayAtoms.append(_subscript!)
            // update the position
            currentPosition.x += _subscript!.width + styleFont.mathTable!.spaceAfterScript;
            return;
        }
        
        let superScript = MTTypesetter.createLineForMathList(atom!.superScript, font:font, style:self.scriptStyle(), cramped:self.superScriptCramped())
        superScript!.type = .superscript
        superScript!.index = Int(index);
        superScriptShiftUp = fmax(superScriptShiftUp, self.superScriptShiftUp());
        superScriptShiftUp = fmax(superScriptShiftUp, superScript!.descent + styleFont.mathTable!.superscriptBottomMin);
        
        if atom!.subScript == nil {
            superScript!.position = CGPointMake(currentPosition.x, currentPosition.y + superScriptShiftUp);
            displayAtoms.append(superScript!)
            // update the position
            currentPosition.x += superScript!.width + styleFont.mathTable!.spaceAfterScript;
            return;
        }
        let ssubscript = MTTypesetter.createLineForMathList(atom!.subScript, font:font, style:self.scriptStyle(), cramped:self.subscriptCramped())
        ssubscript!.type = .ssubscript
        ssubscript!.index = Int(index)
        subscriptShiftDown = fmax(subscriptShiftDown, styleFont.mathTable!.subscriptShiftDown);
        
        // joint positioning of subscript & superscript
        let subSuperScriptGap = (superScriptShiftUp - superScript!.descent) + (subscriptShiftDown - ssubscript!.ascent);
        if (subSuperScriptGap < styleFont.mathTable!.subSuperscriptGapMin) {
            // Set the gap to atleast as much
            subscriptShiftDown += styleFont.mathTable!.subSuperscriptGapMin - subSuperScriptGap;
            let superscriptBottomDelta = styleFont.mathTable!.superscriptBottomMaxWithSubscript - (superScriptShiftUp - superScript!.descent);
            if (superscriptBottomDelta > 0) {
                // superscript is lower than the max allowed by the font with a subscript.
                superScriptShiftUp += superscriptBottomDelta;
                subscriptShiftDown -= superscriptBottomDelta;
            }
        }
        // The delta is the italic correction above that shift superscript position
        superScript?.position = CGPointMake(currentPosition.x + delta, currentPosition.y + superScriptShiftUp);
        displayAtoms.append(superScript!)
        ssubscript?.position = CGPointMake(currentPosition.x, currentPosition.y - subscriptShiftDown);
        displayAtoms.append(ssubscript!)
        currentPosition.x += max(superScript!.width + delta, ssubscript!.width) + styleFont.mathTable!.spaceAfterScript;
    }
    
    // MARK: - Fractions
    
    func numeratorShiftUp(_ hasRule:Bool) -> CGFloat {
        if hasRule {
            if style == .display {
                return styleFont.mathTable!.fractionNumeratorDisplayStyleShiftUp
            } else {
                return styleFont.mathTable!.fractionNumeratorShiftUp
            }
        } else {
            if style == .display {
                return styleFont.mathTable!.stackTopDisplayStyleShiftUp
            } else {
                return styleFont.mathTable!.stackTopShiftUp
            }
        }
    }
    
    func numeratorGapMin() -> CGFloat {
        if style == .display {
            return styleFont.mathTable!.fractionNumeratorDisplayStyleGapMin;
        } else {
            return styleFont.mathTable!.fractionNumeratorGapMin;
        }
    }
    
    func denominatorShiftDown(_ hasRule:Bool) -> CGFloat {
        if hasRule {
            if style == .display {
                return styleFont.mathTable!.fractionDenominatorDisplayStyleShiftDown;
            } else {
                return styleFont.mathTable!.fractionDenominatorShiftDown;
            }
        } else {
            if style == .display {
                return styleFont.mathTable!.stackBottomDisplayStyleShiftDown;
            } else {
                return styleFont.mathTable!.stackBottomShiftDown;
            }
        }
    }
    
    func denominatorGapMin() -> CGFloat {
        if style == .display {
            return styleFont.mathTable!.fractionDenominatorDisplayStyleGapMin;
        } else {
            return styleFont.mathTable!.fractionDenominatorGapMin;
        }
    }
    
    func stackGapMin() -> CGFloat {
        if style == .display {
            return styleFont.mathTable!.stackDisplayStyleGapMin;
        } else {
            return styleFont.mathTable!.stackGapMin;
        }
    }
    
    func fractionDelimiterHeight()-> CGFloat {
        if style == .display {
            return styleFont.mathTable!.fractionDelimiterDisplayStyleSize;
        } else {
            return styleFont.mathTable!.fractionDelimiterSize;
        }
    }
    
    func fractionStyle() -> MTLineStyle {
        if style == .scriptOfScript {
            return .scriptOfScript
        }
        return style.inc()
    }
    
    func makeFraction(_ frac:MTFraction?) -> MTDisplay? {
        // lay out the parts of the fraction
        let fractionStyle = self.fractionStyle;
        let numeratorDisplay = MTTypesetter.createLineForMathList(frac!.numerator, font:font, style:fractionStyle(), cramped:false)
        let denominatorDisplay = MTTypesetter.createLineForMathList(frac!.denominator, font:font, style:fractionStyle(), cramped:true)
        
        // determine the location of the numerator
        var numeratorShiftUp = self.numeratorShiftUp(frac!.hasRule)
        var denominatorShiftDown = self.denominatorShiftDown(frac!.hasRule)
        let barLocation = styleFont.mathTable!.axisHeight
        let barThickness = frac!.hasRule ? styleFont.mathTable!.fractionRuleThickness : 0
        
        if frac!.hasRule {
            // This is the difference between the lowest edge of the numerator and the top edge of the fraction bar
            let distanceFromNumeratorToBar = (numeratorShiftUp - numeratorDisplay!.descent) - (barLocation + barThickness/2);
            // The distance should at least be displayGap
            let minNumeratorGap = self.numeratorGapMin;
            if distanceFromNumeratorToBar < minNumeratorGap() {
                // This makes the distance between the bottom of the numerator and the top edge of the fraction bar
                // at least minNumeratorGap.
                numeratorShiftUp += (minNumeratorGap() - distanceFromNumeratorToBar);
            }
            
            // Do the same for the denominator
            // This is the difference between the top edge of the denominator and the bottom edge of the fraction bar
            let distanceFromDenominatorToBar = (barLocation - barThickness/2) - (denominatorDisplay!.ascent - denominatorShiftDown);
            // The distance should at least be denominator gap
            let minDenominatorGap = self.denominatorGapMin;
            if distanceFromDenominatorToBar < minDenominatorGap() {
                // This makes the distance between the top of the denominator and the bottom of the fraction bar to be exactly
                // minDenominatorGap
                denominatorShiftDown += (minDenominatorGap() - distanceFromDenominatorToBar);
            }
        } else {
            // This is the distance between the numerator and the denominator
            let clearance = (numeratorShiftUp - numeratorDisplay!.descent) - (denominatorDisplay!.ascent - denominatorShiftDown);
            // This is the minimum clearance between the numerator and denominator.
            let minGap = self.stackGapMin()
            if clearance < minGap {
                numeratorShiftUp += (minGap - clearance)/2;
                denominatorShiftDown += (minGap - clearance)/2;
            }
        }
        
        let display = MTFractionDisplay(withNumerator: numeratorDisplay, denominator: denominatorDisplay, position: currentPosition, range: frac!.indexRange)
        
        display.numeratorUp = numeratorShiftUp;
        display.denominatorDown = denominatorShiftDown;
        display.lineThickness = barThickness;
        display.linePosition = barLocation;
        if frac!.leftDelimiter.isEmpty && frac!.rightDelimiter.isEmpty {
            return display
        } else {
            return self.addDelimitersToFractionDisplay(display, forFraction:frac)
        }
    }
    
    func addDelimitersToFractionDisplay(_ display:MTFractionDisplay?, forFraction frac:MTFraction?) -> MTDisplay? {
        assert(!frac!.leftDelimiter.isEmpty || !frac!.rightDelimiter.isEmpty, "Fraction should have a delimiters to call this function");
        
        var innerElements = [MTDisplay]()
        let glyphHeight = self.fractionDelimiterHeight
        var position = CGPoint.zero
        if !frac!.leftDelimiter.isEmpty {
            let leftGlyph = self.findGlyphForBoundary(frac!.leftDelimiter, withHeight:glyphHeight())
            leftGlyph!.position = position
            position.x += leftGlyph!.width
            innerElements.append(leftGlyph!)
        }
        
        display!.position = position
        position.x += display!.width
        innerElements.append(display!)
        
        if !frac!.rightDelimiter.isEmpty {
            let rightGlyph = self.findGlyphForBoundary(frac!.rightDelimiter, withHeight:glyphHeight())
            rightGlyph!.position = position
            position.x += rightGlyph!.width
            innerElements.append(rightGlyph!)
        }
        let innerDisplay = MTMathListDisplay(withDisplays: innerElements, range: frac!.indexRange)
        innerDisplay.position = currentPosition
        return innerDisplay
    }
    
    // MARK: - Radicals
    
    func radicalVerticalGap() -> CGFloat {
        if style == .display {
            return styleFont.mathTable!.radicalDisplayStyleVerticalGap
        } else {
            return styleFont.mathTable!.radicalVerticalGap
        }
    }
    
    func getRadicalGlyphWithHeight(_ radicalHeight:CGFloat) -> MTDisplayDS? {
        var glyphAscent=CGFloat(0), glyphDescent=CGFloat(0), glyphWidth=CGFloat(0)
        
        let radicalGlyph = self.findGlyphForCharacterAtIndex("\u{221A}".startIndex, inString:"\u{221A}")
        let glyph = self.findGlyph(radicalGlyph, withHeight:radicalHeight, glyphAscent:&glyphAscent, glyphDescent:&glyphDescent, glyphWidth:&glyphWidth)
        
        var glyphDisplay:MTDisplayDS?
        if glyphAscent + glyphDescent < radicalHeight {
            // the glyphs is not as large as required. A glyph needs to be constructed using the extenders.
            glyphDisplay = self.constructGlyph(radicalGlyph, withHeight:radicalHeight)
        }
        
        if glyphDisplay == nil {
            // No constructed display so use the glyph we got.
            glyphDisplay = MTGlyphDisplay(withGlpyh: glyph, range: NSMakeRange(NSNotFound, 0), font:styleFont)
            glyphDisplay!.ascent = glyphAscent;
            glyphDisplay!.descent = glyphDescent;
            glyphDisplay!.width = glyphWidth;
        }
        return glyphDisplay;
    }
    
    func makeRadical(_ radicand:MTMathList?, range:NSRange) -> MTRadicalDisplay? {
        let innerDisplay = MTTypesetter.createLineForMathList(radicand, font:font, style:style, cramped:true)!
        var clearance = self.radicalVerticalGap()
        let radicalRuleThickness = styleFont.mathTable!.radicalRuleThickness
        let radicalHeight = innerDisplay.ascent + innerDisplay.descent + clearance + radicalRuleThickness
        
        let glyph = self.getRadicalGlyphWithHeight(radicalHeight)!
        
        // Note this is a departure from Latex. Latex assumes that glyphAscent == thickness.
        // Open type math makes no such assumption, and ascent and descent are independent of the thickness.
        // Latex computes delta as descent - (h(inner) + d(inner) + clearance)
        // but since we may not have ascent == thickness, we modify the delta calculation slightly.
        // If the font designer followes Latex conventions, it will be identical.
        let delta = (glyph.descent + glyph.ascent) - (innerDisplay.ascent + innerDisplay.descent + clearance + radicalRuleThickness)
        if delta > 0 {
            clearance += delta/2  // increase the clearance to center the radicand inside the sign.
        }
        
        // we need to shift the radical glyph up, to coincide with the baseline of inner.
        // The new ascent of the radical glyph should be thickness + adjusted clearance + h(inner)
        let radicalAscent = radicalRuleThickness + clearance + innerDisplay.ascent
        let shiftUp = radicalAscent - glyph.ascent  // Note: if the font designer followed latex conventions, this is the same as glyphAscent == thickness.
        glyph.shiftDown = -shiftUp
        
        let radical = MTRadicalDisplay(withRadicand: innerDisplay, glyph: glyph, position: currentPosition, range: range)
        radical.ascent = radicalAscent + styleFont.mathTable!.radicalExtraAscender
        radical.topKern = styleFont.mathTable!.radicalExtraAscender
        radical.lineThickness = radicalRuleThickness
        // Note: Until we have radical construction from parts, it is possible that glyphAscent+glyphDescent is less
        // than the requested height of the glyph (i.e. radicalHeight), so in the case the innerDisplay has a larger
        // descent we use the innerDisplay's descent.
        radical.descent = max(glyph.ascent + glyph.descent - radicalAscent, innerDisplay.descent)
        radical.width = glyph.width + innerDisplay.width
        return radical
    }
    
    // MARK: - Glyphs
    
    func findGlyph(_ glyph:CGGlyph, withHeight height:CGFloat, glyphAscent:inout CGFloat, glyphDescent:inout CGFloat, glyphWidth:inout CGFloat) -> CGGlyph {
        let variants = styleFont.mathTable!.getVerticalVariantsForGlyph(glyph)
        let numVariants = variants.count;
        var glyphs = [CGGlyph]()// numVariants)
        glyphs.reserveCapacity(numVariants)
        for i in 0 ..< numVariants {
            let glyph = variants[i]!.uint16Value
            glyphs.append(glyph)
        }
        
        var bboxes = [CGRect](repeating: CGRect.zero, count: numVariants)
        var advances = [CGSize](repeating: CGSize.zero, count: numVariants)
        
        // Get the bounds for these glyphs
        CTFontGetBoundingRectsForGlyphs(styleFont.ctFont, .horizontal, glyphs, &bboxes, numVariants)
        CTFontGetAdvancesForGlyphs(styleFont.ctFont, .horizontal, glyphs, &advances, numVariants);
        var ascent=CGFloat(0), descent=CGFloat(0), width=CGFloat(0)
        for i in 0..<numVariants {
            let bounds = bboxes[i]
            width = advances[i].width;
            getBboxDetails(bounds, ascent: &ascent, descent: &descent);
            
            if (ascent + descent >= height) {
                glyphAscent = ascent;
                glyphDescent = descent;
                glyphWidth = width;
                return glyphs[i]
            }
        }
        glyphAscent = ascent;
        glyphDescent = descent;
        glyphWidth = width;
        return glyphs[numVariants - 1]
    }
    
    func constructGlyph(_ glyph:CGGlyph, withHeight glyphHeight:CGFloat) -> MTGlyphConstructionDisplay? {
        let parts = styleFont.mathTable!.getVerticalGlyphAssembly(forGlyph: glyph)
        if parts.count == 0 {
            return nil
        }
        var glyphs = [NSNumber](), offsets = [NSNumber]()
        var height:CGFloat=0
        self.constructGlyphWithParts(parts, glyphHeight:glyphHeight, glyphs:&glyphs, offsets:&offsets, height:&height)
        var first = glyphs[0].uint16Value
        let width = CTFontGetAdvancesForGlyphs(styleFont.ctFont, .horizontal, &first, nil, 1);
        let display = MTGlyphConstructionDisplay(withGlyphs: glyphs, offsets: offsets, font: styleFont)
        display.width = width;
        display.ascent = height;
        display.descent = 0;   // it's upto the rendering to adjust the display up or down.
        return display;
    }
    
    func constructGlyphWithParts(_ parts:[GlyphPart], glyphHeight:CGFloat, glyphs:inout [NSNumber], offsets:inout [NSNumber], height:inout CGFloat) {
        // Loop forever until the glyph height is valid
        for numExtenders in 0..<Int.max {
            var glyphsRv = [NSNumber]()
            var offsetsRv = [NSNumber]()
            
            var prev:GlyphPart? = nil;
            let minDistance = styleFont.mathTable!.minConnectorOverlap;
            var minOffset = CGFloat(0)
            var maxDelta = CGFloat.greatestFiniteMagnitude  // the maximum amount we can increase the offsets by
            
            for part in parts {
                var repeats = 1;
                if part.isExtender {
                    repeats = numExtenders;
                }
                // add the extender num extender times
                for _ in 0 ..< repeats {
                    glyphsRv.append(NSNumber(value: part.glyph)) // addObject:[NSNumber numberWithShort:part.glyph])
                    if prev != nil {
                        let maxOverlap = min(prev!.endConnectorLength, part.startConnectorLength);
                        // the minimum amount we can add to the offset
                        let minOffsetDelta = prev!.fullAdvance - maxOverlap;
                        // The maximum amount we can add to the offset.
                        let maxOffsetDelta = prev!.fullAdvance - minDistance;
                        // we can increase the offsets by at most max - min.
                        maxDelta = min(maxDelta, maxOffsetDelta - minOffsetDelta);
                        minOffset = minOffset + minOffsetDelta;
                    }
                    offsetsRv.append(NSNumber(floatLiteral: minOffset))  // addObject:[NSNumber numberWithFloat:minOffset])
                    prev = part
                }
            }
            
            assert(glyphsRv.count == offsetsRv.count, "Offsets should match the glyphs");
            if prev == nil {
                continue;   // maybe only extenders
            }
            let minHeight = minOffset + prev!.fullAdvance
            let maxHeight = minHeight + maxDelta * CGFloat(glyphsRv.count - 1)
            if (minHeight >= glyphHeight) {
                // we are done
                glyphs = glyphsRv;
                offsets = offsetsRv;
                height = minHeight;
                return;
            } else if (glyphHeight <= maxHeight) {
                // spread the delta equally between all the connectors
                let delta = glyphHeight - minHeight;
                let deltaIncrease = Float(delta) / Float(glyphsRv.count - 1)
                var lastOffset = CGFloat(0)
                for i in 0..<offsetsRv.count {
                    let offset = offsetsRv[i].floatValue + Float(i)*deltaIncrease;
                    offsetsRv[i] = NSNumber(value:offset)
                    lastOffset = CGFloat(offset)
                }
                // we are done
                glyphs = glyphsRv
                offsets = offsetsRv
                height = lastOffset + prev!.fullAdvance;
                return;
            }
        }
    }
    
    func findGlyphForCharacterAtIndex(_ index:String.Index, inString str:String) -> CGGlyph {
        // Get the character at index taking into account UTF-32 characters
        var chars = Array(str[index].utf16)
        
        // Get the glyph from the font
        var glyph = [CGGlyph](repeating: CGGlyph.zero, count: chars.count)
        let found = CTFontGetGlyphsForCharacters(styleFont.ctFont, &chars, &glyph, chars.count)
        if !found {
            // the font did not contain a glyph for our character, so we just return 0 (notdef)
            return 0
        }
        return glyph[0]
    }
    
    // MARK: - Large Operators
    
    func makeLargeOp(_ op:MTLargeOperator!) -> MTDisplay?  {
        let limits = op.limits && style == .display
        var delta = CGFloat(0)
        if op.nucleus.count == 1 {
            var glyph = self.findGlyphForCharacterAtIndex(op.nucleus.startIndex, inString:op.nucleus)
            if style == .display && glyph != 0 {
                // Enlarge the character in display style.
                glyph = styleFont.mathTable!.getLargerGlyph(glyph)
            }
            // This is be the italic correction of the character.
            delta = styleFont.mathTable!.getItalicCorrection(glyph)

            // vertically center
            let bbox = CTFontGetBoundingRectsForGlyphs(styleFont.ctFont, .horizontal, &glyph, nil, 1);
            let width = CTFontGetAdvancesForGlyphs(styleFont.ctFont, .horizontal, &glyph, nil, 1);
            var ascent=CGFloat(0), descent=CGFloat(0)
            getBboxDetails(bbox, ascent: &ascent, descent: &descent)
            let shiftDown = 0.5*(ascent - descent) - styleFont.mathTable!.axisHeight;
            let glyphDisplay = MTGlyphDisplay(withGlpyh: glyph, range: op.indexRange, font: styleFont)
            glyphDisplay.ascent = ascent;
            glyphDisplay.descent = descent;
            glyphDisplay.width = width;
            if (op.subScript != nil) && !limits {
                // Remove italic correction from the width of the glyph if
                // there is a subscript and limits is not set.
                glyphDisplay.width -= delta;
            }
            glyphDisplay.shiftDown = shiftDown;
            glyphDisplay.position = currentPosition;
            return self.addLimitsToDisplay(glyphDisplay, forOperator:op, delta:delta)
        } else {
            // Create a regular node
            let line = NSMutableAttributedString(string: op.nucleus)
            // add the font
            line.addAttribute(kCTFontAttributeName as NSAttributedString.Key, value:styleFont.ctFont!, range:NSMakeRange(0, line.length))
            let displayAtom = MTCTLineDisplay(withString: line, position: currentPosition, range: op.indexRange, font: styleFont, atoms: [op])
            return self.addLimitsToDisplay(displayAtom, forOperator:op, delta:0)
        }
    }
    
    func addLimitsToDisplay(_ display:MTDisplay?, forOperator op:MTLargeOperator, delta:CGFloat) -> MTDisplay? {
        // If there is no subscript or superscript, just return the current display
        if op.subScript == nil && op.superScript == nil {
            currentPosition.x += display!.width
            return display;
        }
        if op.limits && style == .display {
            // make limits
            var superScript:MTMathListDisplay? = nil, subScript:MTMathListDisplay? = nil
            if op.superScript != nil {
                superScript = MTTypesetter.createLineForMathList(op.superScript, font:font, style:self.scriptStyle(), cramped:self.superScriptCramped())
            }
            if op.subScript != nil {
                subScript = MTTypesetter.createLineForMathList(op.subScript, font:font, style:self.scriptStyle(), cramped:self.subscriptCramped())
            }
            assert((superScript != nil) || (subScript != nil), "At least one of superscript or subscript should have been present.");
            let opsDisplay = MTLargeOpLimitsDisplay(withNucleus:display, upperLimit:superScript, lowerLimit:subScript, limitShift:delta/2, extraPadding:0)
            if superScript != nil {
                let upperLimitGap = max(styleFont.mathTable!.upperLimitGapMin, styleFont.mathTable!.upperLimitBaselineRiseMin - superScript!.descent);
                opsDisplay.upperLimitGap = upperLimitGap;
            }
            if subScript != nil {
                let lowerLimitGap = max(styleFont.mathTable!.lowerLimitGapMin, styleFont.mathTable!.lowerLimitBaselineDropMin - subScript!.ascent);
                opsDisplay.lowerLimitGap = lowerLimitGap;
            }
            opsDisplay.position = currentPosition;
            opsDisplay.range = op.indexRange;
            currentPosition.x += opsDisplay.width;
            return opsDisplay;
        } else {
            currentPosition.x += display!.width;
            self.makeScripts(op, display:display, index:UInt(op.indexRange.location), delta:delta)
            return display;
        }
    }
    
    // MARK: - Large delimiters
    
    // Delimiter shortfall from plain.tex
    static let kDelimiterFactor = CGFloat(901)
    static let kDelimiterShortfallPoints = CGFloat(5)
    
    func makeLeftRight(_ inner: MTInner?) -> MTDisplay? {
        assert(inner!.leftBoundary != nil || inner!.rightBoundary != nil, "Inner should have a boundary to call this function");
        
        let innerListDisplay = MTTypesetter.createLineForMathList(inner!.innerList, font:font, style:style, cramped:cramped, spaced:true)
        let axisHeight = styleFont.mathTable!.axisHeight
        // delta is the max distance from the axis
        let delta = max(innerListDisplay!.ascent - axisHeight, innerListDisplay!.descent + axisHeight);
        let d1 = (delta / 500) * MTTypesetter.kDelimiterFactor;  // This represents atleast 90% of the formula
        let d2 = 2 * delta - MTTypesetter.kDelimiterShortfallPoints;  // This represents a shortfall of 5pt
        // The size of the delimiter glyph should cover at least 90% of the formula or
        // be at most 5pt short.
        let glyphHeight = max(d1, d2);
        
        var innerElements = [MTDisplay]()
        var position = CGPoint.zero
        if inner!.leftBoundary != nil && !inner!.leftBoundary!.nucleus.isEmpty {
            let leftGlyph = self.findGlyphForBoundary(inner!.leftBoundary!.nucleus, withHeight:glyphHeight)
            leftGlyph!.position = position
            position.x += leftGlyph!.width
            innerElements.append(leftGlyph!)
        }
        
        innerListDisplay!.position = position;
        position.x += innerListDisplay!.width;
        innerElements.append(innerListDisplay!)
        
        if inner!.rightBoundary != nil && !inner!.rightBoundary!.nucleus.isEmpty {
            let rightGlyph = self.findGlyphForBoundary(inner!.rightBoundary!.nucleus, withHeight:glyphHeight)
            rightGlyph!.position = position;
            position.x += rightGlyph!.width;
            innerElements.append(rightGlyph!)
        }
        let innerDisplay = MTMathListDisplay(withDisplays: innerElements, range: inner!.indexRange)
        return innerDisplay
    }
    
    func findGlyphForBoundary(_ delimiter:String, withHeight glyphHeight:CGFloat) -> MTDisplay? {
        var glyphAscent=CGFloat(0), glyphDescent=CGFloat(0), glyphWidth=CGFloat(0)
        let leftGlyph = self.findGlyphForCharacterAtIndex(delimiter.startIndex, inString:delimiter)
        let glyph = self.findGlyph(leftGlyph, withHeight:glyphHeight, glyphAscent:&glyphAscent, glyphDescent:&glyphDescent, glyphWidth:&glyphWidth)
        
        var glyphDisplay:MTDisplayDS?
        if (glyphAscent + glyphDescent < glyphHeight) {
            // we didn't find a pre-built glyph that is large enough
            glyphDisplay = self.constructGlyph(leftGlyph, withHeight:glyphHeight)
        }
        
        if glyphDisplay == nil {
            // Create a glyph display
            glyphDisplay = MTGlyphDisplay(withGlpyh: glyph, range: NSMakeRange(NSNotFound, 0), font:styleFont)
            glyphDisplay!.ascent = glyphAscent;
            glyphDisplay!.descent = glyphDescent;
            glyphDisplay!.width = glyphWidth;
        }
        // Center the glyph on the axis
        let shiftDown = 0.5*(glyphDisplay!.ascent - glyphDisplay!.descent) - styleFont.mathTable!.axisHeight;
        glyphDisplay!.shiftDown = shiftDown;
        return glyphDisplay;
    }
    
    // MARK: - Underline/Overline
    
    func makeUnderline(_ under:MTUnderLine?) -> MTDisplay? {
        let innerListDisplay = MTTypesetter.createLineForMathList(under!.innerList, font:font, style:style, cramped:cramped)
        let underDisplay = MTLineDisplay(withInner: innerListDisplay, position: currentPosition, range: under!.indexRange)
        // Move the line down by the vertical gap.
        underDisplay.lineShiftUp = -(innerListDisplay!.descent + styleFont.mathTable!.underbarVerticalGap);
        underDisplay.lineThickness = styleFont.mathTable!.underbarRuleThickness;
        underDisplay.ascent = innerListDisplay!.ascent
        underDisplay.descent = innerListDisplay!.descent + styleFont.mathTable!.underbarVerticalGap + styleFont.mathTable!.underbarRuleThickness + styleFont.mathTable!.underbarExtraDescender;
        underDisplay.width = innerListDisplay!.width;
        return underDisplay;
    }
    
    func makeOverline(_ over:MTOverLine?) -> MTDisplay? {
        let innerListDisplay = MTTypesetter.createLineForMathList(over!.innerList, font:font, style:style, cramped:true)
        let overDisplay = MTLineDisplay(withInner:innerListDisplay, position:currentPosition, range:over!.indexRange)
        overDisplay.lineShiftUp = innerListDisplay!.ascent + styleFont.mathTable!.overbarVerticalGap;
        overDisplay.lineThickness = styleFont.mathTable!.underbarRuleThickness;
        overDisplay.ascent = innerListDisplay!.ascent + styleFont.mathTable!.overbarVerticalGap + styleFont.mathTable!.overbarRuleThickness + styleFont.mathTable!.overbarExtraAscender;
        overDisplay.descent = innerListDisplay!.descent;
        overDisplay.width = innerListDisplay!.width;
        return overDisplay;
    }
    
    // MARK: - Accents
    
    func isSingleCharAccentee(_ accent:MTAccent?) -> Bool {
        guard let accent = accent else { return false }
        if accent.innerList!.atoms.count != 1 {
            // Not a single char list.
            return false
        }
        let innerAtom = accent.innerList!.atoms[0]
        if innerAtom.nucleus.count != 1 {
            // A complex atom, not a simple char.
            return false
        }
        if innerAtom.subScript != nil || innerAtom.superScript != nil {
            return false
        }
        return true
    }
    
    // The distance the accent must be moved from the beginning.
    func getSkew(_ accent: MTAccent?, accenteeWidth width:CGFloat, accentGlyph:CGGlyph) -> CGFloat {
        guard let accent = accent else { return 0 }
        if accent.nucleus.isEmpty {
            // No accent
            return 0
        }
        let accentAdjustment = styleFont.mathTable!.getTopAccentAdjustment(accentGlyph)
        var accenteeAdjustment = CGFloat(0)
        if !self.isSingleCharAccentee(accent) {
            // use the center of the accentee
            accenteeAdjustment = width/2
        } else {
            let innerAtom = accent.innerList!.atoms[0]
            let accenteeGlyph = self.findGlyphForCharacterAtIndex(innerAtom.nucleus.index(innerAtom.nucleus.endIndex, offsetBy:-1), inString:innerAtom.nucleus)
            accenteeAdjustment = styleFont.mathTable!.getTopAccentAdjustment(accenteeGlyph)
        }
        // The adjustments need to aligned, so skew is just the difference.
        return (accenteeAdjustment - accentAdjustment)
    }
    
    // Find the largest horizontal variant if exists, with width less than max width.
    func findVariantGlyph(_ glyph:CGGlyph, withMaxWidth maxWidth:CGFloat, maxWidth glyphAscent:inout CGFloat, glyphDescent:inout CGFloat, glyphWidth:inout CGFloat) -> CGGlyph {
        let variants = styleFont.mathTable!.getHorizontalVariantsForGlyph(glyph)
        let numVariants = variants.count
        assert(numVariants > 0, "A glyph is always it's own variant, so number of variants should be > 0");
        var glyphs = [CGGlyph]() // [numVariants)
        glyphs.reserveCapacity(numVariants)
        for i in 0 ..< numVariants {
            let glyph = variants[i]!.uint16Value
            glyphs.append(glyph)
        }

        var curGlyph = glyphs[0]  // if no other glyph is found, we'll return the first one.
        var bboxes = [CGRect](repeating: CGRect.zero, count: numVariants) // [numVariants)
        var advances = [CGSize](repeating: CGSize.zero, count:numVariants)
        // Get the bounds for these glyphs
        CTFontGetBoundingRectsForGlyphs(styleFont.ctFont, .horizontal, &glyphs, &bboxes, numVariants);
        CTFontGetAdvancesForGlyphs(styleFont.ctFont, .horizontal, &glyphs, &advances, numVariants);
        for i in 0..<numVariants {
            let bounds = bboxes[i]
            var ascent=CGFloat(0), descent=CGFloat(0)
            let width = CGRectGetMaxX(bounds);
            getBboxDetails(bounds, ascent: &ascent, descent: &descent);

            if (width > maxWidth) {
                if (i == 0) {
                    // glyph dimensions are not yet set
                    glyphWidth = advances[i].width;
                    glyphAscent = ascent;
                    glyphDescent = descent;
                }
                return curGlyph;
            } else {
                curGlyph = glyphs[i]
                glyphWidth = advances[i].width;
                glyphAscent = ascent;
                glyphDescent = descent;
            }
        }
        // We exhausted all the variants and none was larger than the width, so we return the largest
        return curGlyph;
    }
    
    func makeAccent(_ accent:MTAccent?) -> MTDisplay? {
        var accentee = MTTypesetter.createLineForMathList(accent!.innerList, font:font, style:style, cramped:true)
        if accent!.nucleus.isEmpty {
            // no accent!
            return accentee
        }
        let end = accent!.nucleus.index(before: accent!.nucleus.endIndex)
        var accentGlyph = self.findGlyphForCharacterAtIndex(end, inString:accent!.nucleus)
        let accenteeWidth = accentee!.width;
        var glyphAscent=CGFloat(0), glyphDescent=CGFloat(0), glyphWidth=CGFloat(0)
        accentGlyph = self.findVariantGlyph(accentGlyph, withMaxWidth:accenteeWidth, maxWidth:&glyphAscent, glyphDescent:&glyphDescent, glyphWidth:&glyphWidth)
        let delta = min(accentee!.ascent, styleFont.mathTable!.accentBaseHeight);
        let skew = self.getSkew(accent, accenteeWidth:accenteeWidth, accentGlyph:accentGlyph)
        let height = accentee!.ascent - delta;  // This is always positive since delta <= height.
        let accentPosition = CGPointMake(skew, height);
        let accentGlyphDisplay = MTGlyphDisplay(withGlpyh: accentGlyph, range: accent!.indexRange, font: styleFont)
        accentGlyphDisplay.ascent = glyphAscent;
        accentGlyphDisplay.descent = glyphDescent;
        accentGlyphDisplay.width = glyphWidth;
        accentGlyphDisplay.position = accentPosition;

        if self.isSingleCharAccentee(accent) && (accent!.subScript != nil || accent!.superScript != nil) {
            // Attach the super/subscripts to the accentee instead of the accent.
            let innerAtom = accent!.innerList!.atoms[0]
            innerAtom.superScript = accent!.superScript;
            innerAtom.subScript = accent!.subScript;
            accent?.superScript = nil;
            accent?.subScript = nil;
            // Remake the accentee (now with sub/superscripts)
            // Note: Latex adjusts the heights in case the height of the char is different in non-cramped mode. However this shouldn't be the case since cramping
            // only affects fractions and superscripts. We skip adjusting the heights.
            accentee = MTTypesetter.createLineForMathList(accent!.innerList, font:font, style:style, cramped:cramped)
        }

        let display = MTAccentDisplay(withAccent:accentGlyphDisplay, accentee:accentee, range:accent!.indexRange)
        display.width = accentee!.width;
        display.descent = accentee!.descent;
        let ascent = accentee!.ascent - delta + glyphAscent;
        display.ascent = max(accentee!.ascent, ascent);
        display.position = currentPosition;

        return display;
    }
    
    // MARK: - Table
    
    let kBaseLineSkipMultiplier = CGFloat(1.2)  // default base line stretch is 12 pt for 10pt font.
    let kLineSkipMultiplier = CGFloat(0.1)  // default is 1pt for 10pt font.
    let kLineSkipLimitMultiplier = CGFloat(0)
    let kJotMultiplier = CGFloat(0.3) // A jot is 3pt for a 10pt font.
    
    func makeTable(_ table:MTMathTable?) -> MTDisplay? {
        let numColumns = table!.numColumns;
        if numColumns == 0 || table!.numRows == 0 {
            // Empty table
            return MTMathListDisplay(withDisplays: [MTDisplay](), range: table!.indexRange)
        }

        var columnWidths = [CGFloat](repeating: 0, count: numColumns)
        let displays = self.typesetCells(table, columnWidths:&columnWidths)

        // Position all the columns in each row
        var rowDisplays = [MTDisplay]()
        for row in displays {
            let rowDisplay = self.makeRowWithColumns(row, forTable:table, columnWidths:columnWidths)
            rowDisplays.append(rowDisplay!)
        }

        // Position all the rows
        self.positionRows(rowDisplays, forTable:table)
        let tableDisplay = MTMathListDisplay(withDisplays: rowDisplays, range: table!.indexRange)
        tableDisplay.position = currentPosition;
        return tableDisplay;
    }
    
    // Typeset every cell in the table. As a side-effect calculate the max column width of each column.
    func typesetCells(_ table:MTMathTable?, columnWidths: inout [CGFloat]) -> [[MTDisplay]] {
        var displays = [[MTDisplay]]()
        for row in table!.cells {
            var colDisplays = [MTDisplay]()
            for i in 0..<row.count {
                let disp = MTTypesetter.createLineForMathList(row[i], font:font, style:style)
                columnWidths[i] = max(disp!.width, columnWidths[i])
                colDisplays.append(disp!)
            }
            displays.append(colDisplays)
        }
        return displays
    }
    
    func makeRowWithColumns(_ cols:[MTDisplay], forTable table:MTMathTable?, columnWidths:[CGFloat]) -> MTMathListDisplay? {
        var columnStart = CGFloat(0)
        var rowRange = NSMakeRange(NSNotFound, 0);
        for i in 0..<cols.count {
            let col = cols[i]
            let colWidth = columnWidths[i]
            let alignment = table?.get(alignmentForColumn: i)
            var cellPos = columnStart;
            switch alignment {
                case .right:
                    cellPos += colWidth - col.width
                case .center:
                    cellPos += (colWidth - col.width) / 2;
                case .left, .none:
                    // No changes if left aligned
                    cellPos += 0  // no op
            }
            if (rowRange.location != NSNotFound) {
                rowRange = NSUnionRange(rowRange, col.range);
            } else {
                rowRange = col.range;
            }

            col.position = CGPointMake(cellPos, 0);
            columnStart += colWidth + table!.interColumnSpacing * styleFont.mathTable!.muUnit;
        }
        // Create a display for the row
        let rowDisplay = MTMathListDisplay(withDisplays: cols, range:rowRange)
        return rowDisplay
    }
    
    func positionRows(_ rows:[MTDisplay], forTable table:MTMathTable?) {
        // Position the rows
        // We will first position the rows starting from 0 and then in the second pass center the whole table vertically.
        var currPos = CGFloat(0)
        let openup = table!.interRowAdditionalSpacing * kJotMultiplier * styleFont.fontSize;
        let baselineSkip = openup + kBaseLineSkipMultiplier * styleFont.fontSize;
        let lineSkip = openup + kLineSkipMultiplier * styleFont.fontSize;
        let lineSkipLimit = openup + kLineSkipLimitMultiplier * styleFont.fontSize;
        var prevRowDescent = CGFloat(0)
        var ascent = CGFloat(0)
        var first = true
        for row in rows {
            if first {
                row.position = CGPointZero;
                ascent += row.ascent;
                first = false;
            } else {
                var skip = baselineSkip;
                if (skip - (prevRowDescent + row.ascent) < lineSkipLimit) {
                    // rows are too close to each other. Space them apart further
                    skip = prevRowDescent + row.ascent + lineSkip;
                }
                // We are going down so we decrease the y value.
                currPos -= skip;
                row.position = CGPointMake(0, currPos);
            }
            prevRowDescent = row.descent;
        }

        // Vertically center the whole structure around the axis
        // The descent of the structure is the position of the last row
        // plus the descent of the last row.
        let descent =  -currPos + prevRowDescent;
        let shiftDown = 0.5*(ascent - descent) - styleFont.mathTable!.axisHeight;

        for row in rows {
            row.position = CGPointMake(row.position.x, row.position.y - shiftDown);
        }
    }
}
