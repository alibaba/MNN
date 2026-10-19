import XCTest
@testable import SwiftMath

//
//  MathTypesetterTests.swift
//  MathTypesetterTests
//
//  Created by Mike Griebling on 2023-01-02.
//

extension CGPoint {
    
    func isEqual(to p:CGPoint, accuracy:CGFloat) -> Bool {
        abs(self.x - p.x) < accuracy && abs(self.y - p.y) < accuracy
    }
    
}

final class MTTypesetterTests: XCTestCase {
    
    var font:MTFont!
    
    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
        try super.setUpWithError()
        self.font = MTFontManager.fontManager.defaultFont
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        try super.tearDownWithError()
    }

    func testSimpleVariable() throws {
        let mathList = MTMathList()
        mathList.add(MTMathAtomFactory.atom(forCharacter: "x"))
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay)
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        // The x is italicized
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, line.ascent);
        XCTAssertEqual(display.descent, line.descent);
        XCTAssertEqual(display.width, line.width);
        
        XCTAssertEqual(display.ascent, 8.834, accuracy: 0.01)
        XCTAssertEqual(display.descent, 0.22, accuracy: 0.01)
        XCTAssertEqual(display.width, 11.44, accuracy: 0.01)
    }

    func testMultipleVariables() throws {
        let mathList = MTMathAtomFactory.mathListForCharacters("xyzw")
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 4)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 4);
        XCTAssertEqual(line.attributedString?.string, "𝑥𝑦𝑧𝑤");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 4)));
        XCTAssertFalse(line.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, line.ascent);
        XCTAssertEqual(display.descent, line.descent);
        XCTAssertEqual(display.width, line.width);
        
        XCTAssertEqual(display.ascent, 8.834, accuracy: 0.01)
        XCTAssertEqual(display.descent, 4.10, accuracy: 0.01)
        XCTAssertEqual(display.width, 44.86, accuracy: 0.01)
    }

    func testVariablesAndNumbers() throws {
        let mathList = MTMathAtomFactory.mathListForCharacters("xy2w")
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular)
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero))
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 4)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript)
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 4);
        XCTAssertEqual(line.attributedString?.string, "𝑥𝑦2𝑤");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 4)));
        XCTAssertFalse(line.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, line.ascent);
        XCTAssertEqual(display.descent, line.descent);
        XCTAssertEqual(display.width, line.width);
        
        XCTAssertEqual(display.ascent, 13.32, accuracy: 0.01)
        XCTAssertEqual(display.descent, 4.10, accuracy: 0.01)
        XCTAssertEqual(display.width, 45.56, accuracy: 0.01)
    }

    func testEquationWithOperatorsAndRelations() throws {
        let mathList = MTMathAtomFactory.mathListForCharacters("2x+3=y")
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 6)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 6);
        XCTAssertEqual(line.attributedString?.string, "2𝑥+3=𝑦");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 6)));
        XCTAssertFalse(line.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, line.ascent);
        XCTAssertEqual(display.descent, line.descent);
        XCTAssertEqual(display.width, line.width);
        
        XCTAssertEqual(display.ascent, 13.32, accuracy: 0.01)
        XCTAssertEqual(display.descent, 4.10, accuracy: 0.01)
        XCTAssertEqual(display.width, 92.36, accuracy: 0.01)
    }

//    #define XCTAssertTrue(CGPointEqualToPoint(p1, p2, accuracy, ...) \
//        XCTAssertEqual(p1.x, p2.x, accuracy, __VA_ARGS__); \
//        XCTAssertEqual(p1.y, p2.y, accuracy, __VA_ARGS__)
//
//
//    #define XCTAssertTrue(NSEqualRanges(r1, r2, ...) \
//        XCTAssertEqual(r1.location, r2.location, __VA_ARGS__); \
//        XCTAssertEqual(r1.length, r2.length, __VA_ARGS__)

    func testSuperscript() throws {
        let mathList = MTMathList()
        let x = MTMathAtomFactory.atom(forCharacter: "x")
        let supersc = MTMathList()
        supersc.add(MTMathAtomFactory.atom(forCharacter: "2"))
        x?.superScript = supersc;
        mathList.add(x)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        // The x is italicized
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(line.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTMathListDisplay)
        let display2 = sub1 as! MTMathListDisplay
        XCTAssertEqual(display2.type, .superscript)
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointMake(11.44, 7.26)))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, 0);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let sub1sub0 = display2.subDisplays[0];
        XCTAssertTrue(sub1sub0 is MTCTLineDisplay);
        let line2 = sub1sub0 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "2");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertFalse(line2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 16.584, accuracy: 0.01)
        XCTAssertEqual(display.descent, 0.22, accuracy: 0.01)
        XCTAssertEqual(display.width, 18.44, accuracy: 0.01)
    }

    func testSubscript() throws {
        let mathList = MTMathList()
        let x = MTMathAtomFactory.atom(forCharacter: "x")
        let subsc = MTMathList()
        subsc.add(MTMathAtomFactory.atom(forCharacter: "1"))
        x?.subScript = subsc
        mathList.add(x)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        // The x is italicized
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(line.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTMathListDisplay);
        let display2 = sub1 as! MTMathListDisplay
        XCTAssertEqual(display2.type, .ssubscript);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointMake(11.44, -4.94)))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)))
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, 0);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let sub1sub0 = display2.subDisplays[0];
        XCTAssertTrue(sub1sub0 is MTCTLineDisplay);
        let line2 = sub1sub0 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertFalse(line2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 8.834, accuracy: 0.01)
        XCTAssertEqual(display.descent, 4.940, accuracy: 0.01)
        XCTAssertEqual(display.width, 18.44, accuracy: 0.01)
    }

    func testSupersubscript() throws {
        let mathList = MTMathList()
        let x = MTMathAtomFactory.atom(forCharacter: "x")
        let supersc = MTMathList()
        supersc.add(MTMathAtomFactory.atom(forCharacter: "2"))
        let subsc = MTMathList()
        subsc.add(MTMathAtomFactory.atom(forCharacter: "1"))
        x?.subScript = subsc;
        x?.superScript = supersc;
        mathList.add(x)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 3);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        // The x is italicized
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(line.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTMathListDisplay);
        let display2 = sub1 as! MTMathListDisplay
        XCTAssertEqual(display2.type, .superscript);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointMake(11.44, 7.26)))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, 0);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let sub1sub0 = display2.subDisplays[0];
        XCTAssertTrue(sub1sub0 is MTCTLineDisplay);
        let line2 = sub1sub0 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "2");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertFalse(line2.hasScript);
        
        let sub2 = display.subDisplays[2];
        XCTAssertTrue(sub2 is MTMathListDisplay);
        let display3 = sub2 as! MTMathListDisplay
        XCTAssertEqual(display3.type, .ssubscript);
        // Positioned differently when both subscript and superscript present.
        XCTAssertTrue(CGPointEqualToPoint(display3.position, CGPointMake(11.44, -5.264)))
        XCTAssertTrue(NSEqualRanges(display3.range, NSMakeRange(0, 1)))
        XCTAssertFalse(display3.hasScript);
        XCTAssertEqual(display3.index, 0);
        XCTAssertEqual(display3.subDisplays.count, 1);
        
        let sub2sub0 = display3.subDisplays[0];
        XCTAssertTrue(sub2sub0 is MTCTLineDisplay)
        let line3 = sub2sub0 as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertFalse(line3.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 16.584, accuracy: 0.01)
        XCTAssertEqual(display.descent, 5.264, accuracy: 0.01)
        XCTAssertEqual(display.width, 18.44, accuracy: 0.01)
    }

    func testRadical() throws {
        let mathList = MTMathList()
        let rad = MTRadical()
        let radicand = MTMathList()
        radicand.add(MTMathAtomFactory.atom(forCharacter: "1"))
        rad.radicand = radicand;
        mathList.add(rad)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTRadicalDisplay);
        let radical = sub0 as! MTRadicalDisplay
        XCTAssertTrue(NSEqualRanges(radical.range, NSMakeRange(0, 1)));
        XCTAssertFalse(radical.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(radical.position, CGPointZero));
        XCTAssertNotNil(radical.radicand);
        XCTAssertNil(radical.degree);

        let display2 = radical.radicand!
        XCTAssertEqual(display2.type, .regular)
        XCTAssertTrue(CGPointMake(16.66, 0).isEqual(to: display2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subrad = display2.subDisplays[0];
        XCTAssertTrue(subrad is MTCTLineDisplay);
        let line2 = subrad as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 19.34, accuracy: 0.01)
        XCTAssertEqual(display.descent, 1.46, accuracy: 0.01)
        XCTAssertEqual(display.width, 26.66, accuracy: 0.01)
    }

    func testRadicalWithDegree() throws {
        let mathList = MTMathList()
        let rad = MTRadical()
        let radicand = MTMathList()
        radicand.add(MTMathAtomFactory.atom(forCharacter: "1"))
        let degree = MTMathList()
        degree.add(MTMathAtomFactory.atom(forCharacter: "3"))
        rad.radicand = radicand;
        rad.degree = degree;
        mathList.add(rad)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTRadicalDisplay);
        let radical = sub0 as! MTRadicalDisplay
        XCTAssertTrue(NSEqualRanges(radical.range, NSMakeRange(0, 1)));
        XCTAssertFalse(radical.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(radical.position, CGPointZero));
        XCTAssertNotNil(radical.radicand);
        XCTAssertNotNil(radical.degree);
        
        let display2 = radical.radicand!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointMake(16.66, 0)))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subrad = display2.subDisplays[0];
        XCTAssertTrue(subrad is MTCTLineDisplay);
        let line2 = subrad as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        let display3 = radical.degree!
        XCTAssertEqual(display3.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display3.position, CGPointMake(6.12, 10.728)))
        XCTAssertTrue(NSEqualRanges(display3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display3.hasScript);
        XCTAssertEqual(display3.index, NSNotFound);
        XCTAssertEqual(display3.subDisplays.count, 1);
        
        let subdeg = display3.subDisplays[0];
        XCTAssertTrue(subdeg is MTCTLineDisplay);
        let line3 = subdeg as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "3");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line3.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 19.34, accuracy: 0.01)
        XCTAssertEqual(display.descent, 1.46, accuracy: 0.01)
        XCTAssertEqual(display.width, 26.66, accuracy: 0.01)
    }

    func testFraction() throws {
        let mathList = MTMathList()
        let frac = MTFraction(hasRule: true)
        let num = MTMathList()
        num.add(MTMathAtomFactory.atom(forCharacter: "1"))
        let denom = MTMathList()
        denom.add(MTMathAtomFactory.atom(forCharacter: "3"))
        frac.numerator = num;
        frac.denominator = denom;
        mathList.add(frac)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTFractionDisplay)
        let fraction = sub0 as! MTFractionDisplay
        XCTAssertTrue(NSEqualRanges(fraction.range, NSMakeRange(0, 1)));
        XCTAssertFalse(fraction.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(fraction.position, CGPointZero));
        XCTAssertNotNil(fraction.numerator);
        XCTAssertNotNil(fraction.denominator);
        
        let display2 = fraction.numerator!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointMake(0, 13.54)))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subnum = display2.subDisplays[0];
        XCTAssertTrue(subnum is MTCTLineDisplay)
        let line2 = subnum as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        let display3 = fraction.denominator!
        XCTAssertEqual(display3.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display3.position, CGPointMake(0, -13.72)))
        XCTAssertTrue(NSEqualRanges(display3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display3.hasScript);
        XCTAssertEqual(display3.index, NSNotFound);
        XCTAssertEqual(display3.subDisplays.count, 1);
        
        let subdenom = display3.subDisplays[0];
        XCTAssertTrue(subdenom is MTCTLineDisplay);
        let line3 = subdenom as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "3");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line3.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 26.86, accuracy: 0.01)
        XCTAssertEqual(display.descent, 14.16, accuracy: 0.01)
        XCTAssertEqual(display.width, 10, accuracy: 0.01)
    }

    func testAtop() throws {
        let mathList = MTMathList()
        let frac = MTFraction(hasRule: false)
        let num = MTMathList()
        num.add(MTMathAtomFactory.atom(forCharacter: "1"))
        let denom = MTMathList()
        denom.add(MTMathAtomFactory.atom(forCharacter: "3"))
        frac.numerator = num;
        frac.denominator = denom;
        mathList.add(frac)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTFractionDisplay)
        let fraction = sub0 as! MTFractionDisplay
        XCTAssertTrue(NSEqualRanges(fraction.range, NSMakeRange(0, 1)));
        XCTAssertFalse(fraction.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(fraction.position, CGPointZero));
        XCTAssertNotNil(fraction.numerator);
        XCTAssertNotNil(fraction.denominator);
        
        let display2 = fraction.numerator!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointMake(0, 13.54)))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subnum = display2.subDisplays[0];
        XCTAssertTrue(subnum is MTCTLineDisplay);
        let line2 = subnum as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        let display3 = fraction.denominator!
        XCTAssertEqual(display3.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display3.position, CGPointMake(0, -13.72)))
        XCTAssertTrue(NSEqualRanges(display3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display3.hasScript);
        XCTAssertEqual(display3.index, NSNotFound);
        XCTAssertEqual(display3.subDisplays.count, 1);
        
        let subdenom = display3.subDisplays[0];
        XCTAssertTrue(subdenom is MTCTLineDisplay);
        let line3 = subdenom as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "3");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line3.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 26.86, accuracy: 0.01)
        XCTAssertEqual(display.descent, 14.16, accuracy: 0.01)
        XCTAssertEqual(display.width, 10, accuracy: 0.01)
    }

    func testBinomial() throws {
        let mathList = MTMathList()
        let frac = MTFraction(hasRule: false)
        let num = MTMathList()
        num.add(MTMathAtomFactory.atom(forCharacter: "1"))
        let denom = MTMathList()
        denom.add(MTMathAtomFactory.atom(forCharacter: "3"))
        frac.numerator = num;
        frac.denominator = denom;
        frac.leftDelimiter = "(";
        frac.rightDelimiter = ")";
        mathList.add(frac)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTMathListDisplay);
        let display0 = sub0 as! MTMathListDisplay
        XCTAssertNotNil(display0);
        XCTAssertEqual(display0.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display0.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display0.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display0.hasScript);
        XCTAssertEqual(display0.index, NSNotFound);
        XCTAssertEqual(display0.subDisplays.count, 3);
        
        let subLeft = display0.subDisplays[0];
        XCTAssertTrue(subLeft is MTGlyphDisplay);
        let glyph = subLeft;
        XCTAssertTrue(CGPointEqualToPoint(glyph.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(NSNotFound, 0)));
        XCTAssertFalse(glyph.hasScript);
        
        let subFrac = display0.subDisplays[1];
        XCTAssertTrue(subFrac is MTFractionDisplay)
        let fraction = subFrac as! MTFractionDisplay
        XCTAssertTrue(NSEqualRanges(fraction.range, NSMakeRange(0, 1)));
        XCTAssertFalse(fraction.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(fraction.position, CGPointMake(14.72, 0)))
        XCTAssertNotNil(fraction.numerator);
        XCTAssertNotNil(fraction.denominator);
        
        let display2 = fraction.numerator!
        XCTAssertEqual(display2.type, .regular)
        XCTAssertTrue(CGPointMake(14.72, 13.54).isEqual(to: display2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subnum = display2.subDisplays[0];
        XCTAssertTrue(subnum is MTCTLineDisplay);
        let line2 = subnum as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        let display3 = fraction.denominator!
        XCTAssertEqual(display3.type, .regular)
        XCTAssertTrue(CGPointMake(14.72, -13.72).isEqual(to: display3.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(display3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display3.hasScript);
        XCTAssertEqual(display3.index, NSNotFound);
        XCTAssertEqual(display3.subDisplays.count, 1);
        
        let subdenom = display3.subDisplays[0];
        XCTAssertTrue(subdenom is MTCTLineDisplay);
        let line3 = subdenom as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "3");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line3.hasScript);
        
        let subRight = display0.subDisplays[2];
        XCTAssertTrue(subRight is MTGlyphDisplay);
        let glyph2 = subRight as! MTGlyphDisplay
        XCTAssertTrue(CGPointEqualToPoint(glyph2.position, CGPointMake(24.72, 0)))
        XCTAssertTrue(NSEqualRanges(glyph2.range, NSMakeRange(NSNotFound, 0)), "Got \(glyph2.range) instead")
        XCTAssertFalse(glyph2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 28.92, accuracy: 0.001);
        XCTAssertEqual(display.descent, 18.92, accuracy: 0.001);
        XCTAssertEqual(display.width, 39.44, accuracy: 0.001);
    }

    func testLargeOpNoLimitsText() throws {
        let mathList = MTMathList()
        mathList.add(MTMathAtomFactory.atom(forLatexSymbol: "sin"))
        mathList.add(MTMathAtomFactory.atom(forCharacter: "x"))
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 2)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        XCTAssertEqual(line.attributedString?.string, "sin");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTCTLineDisplay);
        let line2 = sub1 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointMake(27.893, 0).isEqual(to: line2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(1, 1)), "Got \(line2.range) instead")
        XCTAssertFalse(line2.hasScript);
        
        XCTAssertEqual(display.ascent, 13.14, accuracy: 0.01)
        XCTAssertEqual(display.descent, 0.22, accuracy: 0.01)
        XCTAssertEqual(display.width, 39.33, accuracy: 0.01)
    }

    func testLargeOpNoLimitsSymbol() throws {
        let mathList = MTMathList()
        // Integral
        mathList.add(MTMathAtomFactory.atom(forLatexSymbol:"int"))
        mathList.add(MTMathAtomFactory.atom(forCharacter: "x"))
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 2)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTGlyphDisplay);
        let glyph = sub0;
        XCTAssertTrue(CGPointEqualToPoint(glyph.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(0, 1)));
        XCTAssertFalse(glyph.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTCTLineDisplay);
        let line2 = sub1 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointMake(23.313, 0).isEqual(to: line2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(1, 1)), "Got \(line2.range) instead")
        XCTAssertFalse(line2.hasScript);
        
        XCTAssertEqual(display.ascent, 27.22, accuracy: 0.01)
        XCTAssertEqual(display.descent, 17.22, accuracy: 0.01)
        XCTAssertEqual(display.width, 34.753, accuracy: 0.01)
    }

    func testLargeOpNoLimitsSymbolWithScripts() throws {
        let mathList = MTMathList()
        // Integral
        let op = MTMathAtomFactory.atom(forLatexSymbol:"int")!
        op.superScript = MTMathList()
        op.superScript?.add(MTMathAtomFactory.atom(forCharacter: "1"))
        op.subScript = MTMathList()
        op.subScript?.add(MTMathAtomFactory.atom(forCharacter: "0"))
        mathList.add(op)
        mathList.add(MTMathAtomFactory.atom(forCharacter: "x"))
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 2)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 4);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTMathListDisplay);
        let display0 = sub0 as! MTMathListDisplay
        XCTAssertEqual(display0.type, .superscript);
        XCTAssertTrue(CGPointEqualToPoint(display0.position, CGPointMake(19.98, 23.72)))
        XCTAssertTrue(NSEqualRanges(display0.range, NSMakeRange(0, 1)))
        XCTAssertFalse(display0.hasScript);
        XCTAssertEqual(display0.index, 0);
        XCTAssertEqual(display0.subDisplays.count, 1);
        
        let sub0sub0 = display0.subDisplays[0];
        XCTAssertTrue(sub0sub0 is MTCTLineDisplay);
        let line1 = sub0sub0 as! MTCTLineDisplay
        XCTAssertEqual(line1.atoms.count, 1);
        XCTAssertEqual(line1.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line1.position, CGPointZero));
        XCTAssertFalse(line1.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTMathListDisplay);
        let display1 = sub1 as! MTMathListDisplay
        XCTAssertEqual(display1.type, .ssubscript);
        // Due to italic correction, positioned before subscript.
        XCTAssertTrue(CGPointEqualToPoint(display1.position, CGPointMake(8.16, -20.02)))
        XCTAssertTrue(NSEqualRanges(display1.range, NSMakeRange(0, 1)))
        XCTAssertFalse(display1.hasScript);
        XCTAssertEqual(display1.index, 0);
        XCTAssertEqual(display1.subDisplays.count, 1);
        
        let sub1sub0 = display1.subDisplays[0];
        XCTAssertTrue(sub1sub0 is MTCTLineDisplay);
        let line3 = sub1sub0 as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "0");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertFalse(line3.hasScript);
        
        let sub2 = display.subDisplays[2];
        XCTAssertTrue(sub2 is MTGlyphDisplay);
        let glyph = sub2;
        XCTAssertTrue(CGPointEqualToPoint(glyph.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(0, 1)));
        XCTAssertTrue(glyph.hasScript); // There are subscripts and superscripts
        
        let sub3 = display.subDisplays[3];
        XCTAssertTrue(sub3 is MTCTLineDisplay);
        let line2 = sub3 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointMake(31.433, 0).isEqual(to: line2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(1, 1)), "Got \(line2.range) instead")
        XCTAssertFalse(line1.hasScript);
        
        XCTAssertEqual(display.ascent, 33.044, accuracy: 0.001);
        XCTAssertEqual(display.descent, 20.328, accuracy: 0.001);
        XCTAssertEqual(display.width, 42.873, accuracy: 0.001);
    }


    func testLargeOpWithLimitsTextWithScripts() throws {
        let mathList = MTMathList()
        let op = MTMathAtomFactory.atom(forLatexSymbol:"lim")!
        op.subScript = MTMathList()
        op.subScript?.add(MTMathAtomFactory.atom(forLatexSymbol:"infty"))
        mathList.add(op)
        mathList.add(MTMathAtom(type: .variable, value:"x"))
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 2)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTLargeOpLimitsDisplay)
        let largeOp = sub0 as! MTLargeOpLimitsDisplay
        XCTAssertTrue(CGPointEqualToPoint(largeOp.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(largeOp.range, NSMakeRange(0, 1)));
        XCTAssertFalse(largeOp.hasScript);
        XCTAssertNotNil(largeOp.lowerLimit);
        XCTAssertNil(largeOp.upperLimit);
        
        let display2 = largeOp.lowerLimit!
        XCTAssertEqual(display2.type, .regular)
        XCTAssertTrue(CGPointMake(6.89, -12.00).isEqual(to: display2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let sub0sub0 = display2.subDisplays[0];
        XCTAssertTrue(sub0sub0 is MTCTLineDisplay);
        let line1 = sub0sub0 as! MTCTLineDisplay
        XCTAssertEqual(line1.atoms.count, 1);
        XCTAssertEqual(line1.attributedString?.string, "∞");
        XCTAssertTrue(CGPointEqualToPoint(line1.position, CGPointZero));
        XCTAssertFalse(line1.hasScript);
        
        let sub3 = display.subDisplays[1];
        XCTAssertTrue(sub3 is MTCTLineDisplay);
        let line2 = sub3 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointMake(31.1133, 0).isEqual(to: line2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(1, 1)), "Got \(line2.range) instead")
        XCTAssertFalse(line1.hasScript);
        
        XCTAssertEqual(display.ascent, 13.88, accuracy: 0.01)
        XCTAssertEqual(display.descent, 12.154, accuracy: 0.01)
        XCTAssertEqual(display.width, 42.553, accuracy: 0.01)
    }

    func testLargeOpWithLimitsSymboltWithScripts() throws {
        let mathList = MTMathList()
        let op = MTMathAtomFactory.atom(forLatexSymbol:"sum")!
        op.superScript = MTMathList()
        op.superScript?.add(MTMathAtomFactory.atom(forLatexSymbol:"infty"))
        op.subScript = MTMathList()
        op.subScript?.add(MTMathAtomFactory.atom(forCharacter: "0"))
        mathList.add(op)
        mathList.add(MTMathAtom(type: .variable, value:"x"))
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 2)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTLargeOpLimitsDisplay);
        let largeOp = sub0 as! MTLargeOpLimitsDisplay
        XCTAssertTrue(CGPointEqualToPoint(largeOp.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(largeOp.range, NSMakeRange(0, 1)));
        XCTAssertFalse(largeOp.hasScript);
        XCTAssertNotNil(largeOp.lowerLimit);
        XCTAssertNotNil(largeOp.upperLimit);
        
        let display2 = largeOp.lowerLimit!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointMake(10.94, -21.664).isEqual(to: display2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)))
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let sub0sub0 = display2.subDisplays[0];
        XCTAssertTrue(sub0sub0 is MTCTLineDisplay);
        let line1 = sub0sub0 as! MTCTLineDisplay
        XCTAssertEqual(line1.atoms.count, 1);
        XCTAssertEqual(line1.attributedString?.string, "0");
        XCTAssertTrue(CGPointEqualToPoint(line1.position, CGPointZero));
        XCTAssertFalse(line1.hasScript);
        
        let displayU = largeOp.upperLimit!
        XCTAssertEqual(displayU.type, .regular);
        XCTAssertTrue(CGPointMake(7.44, 23.154).isEqual(to: displayU.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(displayU.range, NSMakeRange(0, 1)))
        XCTAssertFalse(displayU.hasScript);
        XCTAssertEqual(displayU.index, NSNotFound);
        XCTAssertEqual(displayU.subDisplays.count, 1);
        
        let sub0subU = displayU.subDisplays[0];
        XCTAssertTrue(sub0subU is MTCTLineDisplay);
        let line3 = sub0subU as! MTCTLineDisplay
        XCTAssertEqual(line3.atoms.count, 1);
        XCTAssertEqual(line3.attributedString?.string, "∞");
        XCTAssertTrue(CGPointEqualToPoint(line3.position, CGPointZero));
        XCTAssertFalse(line3.hasScript);
        
        let sub3 = display.subDisplays[1];
        XCTAssertTrue(sub3 is MTCTLineDisplay);
        let line2 = sub3 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointMake(32.2133, 0).isEqual(to: line2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(1, 1)), "Got \(line2.range) instead")
        XCTAssertFalse(line2.hasScript);
        
        XCTAssertEqual(display.ascent, 29.342, accuracy: 0.001);
        XCTAssertEqual(display.descent, 21.972, accuracy: 0.001);
        XCTAssertEqual(display.width, 43.653, accuracy: 0.001);
    }

    func testInner() throws {
        let innerList = MTMathList()
        innerList.add(MTMathAtomFactory.atom(forCharacter: "x"))
        let inner = MTInner()
        inner.innerList = innerList;
        inner.leftBoundary = MTMathAtom(type: .boundary, value:"(")
        inner.rightBoundary = MTMathAtom(type: .boundary, value:")")
        
        let mathList = MTMathList()
        mathList.add(inner)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTMathListDisplay);
        let display2 = sub0 as! MTMathListDisplay
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 3);
        
        let subLeft = display2.subDisplays[0];
        XCTAssertTrue(subLeft is MTGlyphDisplay);
        let glyph = subLeft;
        XCTAssertTrue(CGPointEqualToPoint(glyph.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(NSNotFound, 0)));
        XCTAssertFalse(glyph.hasScript);
        
        let sub3 = display2.subDisplays[1];
        XCTAssertTrue(sub3 is MTMathListDisplay);
        let display3 = sub3 as! MTMathListDisplay
        XCTAssertEqual(display3.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display3.position, CGPointMake(7.78, 0)))
        XCTAssertTrue(NSEqualRanges(display3.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display3.hasScript);
        XCTAssertEqual(display3.index, NSNotFound);
        XCTAssertEqual(display3.subDisplays.count, 1);
        
        let subsub3 = display3.subDisplays[0];
        XCTAssertTrue(subsub3 is MTCTLineDisplay);
        let line = subsub3 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        // The x is italicized
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertFalse(line.hasScript);
        
        let subRight = display2.subDisplays[2];
        XCTAssertTrue(subRight is MTGlyphDisplay);
        let glyph2 = subRight as! MTGlyphDisplay
        XCTAssertTrue(CGPointEqualToPoint(glyph2.position, CGPointMake(19.22, 0)))
        XCTAssertTrue(NSEqualRanges(glyph2.range, NSMakeRange(NSNotFound, 0)), "Got \(glyph2.range) instead");
        XCTAssertFalse(glyph2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, display2.ascent);
        XCTAssertEqual(display.descent, display2.descent);
        XCTAssertEqual(display.width, display2.width);
        
        XCTAssertEqual(display.ascent, 14.96, accuracy: 0.001);
        XCTAssertEqual(display.descent, 4.96, accuracy: 0.001);
        XCTAssertEqual(display.width, 27, accuracy: 0.01)
    }

    func testOverline() throws {
        let mathList = MTMathList()
        let over = MTOverLine()
        let inner = MTMathList()
        inner.add(MTMathAtomFactory.atom(forCharacter: "1"))
        over.innerList = inner;
        mathList.add(over)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTLineDisplay);
        let overline = sub0 as! MTLineDisplay
        XCTAssertTrue(NSEqualRanges(overline.range, NSMakeRange(0, 1)));
        XCTAssertFalse(overline.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(overline.position, CGPointZero));
        XCTAssertNotNil(overline.inner);
        
        let display2 = overline.inner!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointZero))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subover = display2.subDisplays[0];
        XCTAssertTrue(subover is MTCTLineDisplay);
        let line2 = subover as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 17.32, accuracy: 0.01)
        XCTAssertEqual(display.descent, 0.00, accuracy: 0.01)
        XCTAssertEqual(display.width, 10, accuracy: 0.01)
    }

    func testUnderline() throws {
        let mathList = MTMathList()
        let under = MTUnderLine()
        let inner = MTMathList()
        inner.add(MTMathAtomFactory.atom(forCharacter: "1"))
        under.innerList = inner;
        mathList.add(under)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTLineDisplay)
        let underline = sub0 as! MTLineDisplay
        XCTAssertTrue(NSEqualRanges(underline.range, NSMakeRange(0, 1)));
        XCTAssertFalse(underline.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(underline.position, CGPointZero));
        XCTAssertNotNil(underline.inner);
        
        let display2 = underline.inner!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointZero))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);
        
        let subover = display2.subDisplays[0];
        XCTAssertTrue(subover is MTCTLineDisplay);
        let line2 = subover as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "1");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);
        
        // dimensions
        XCTAssertEqual(display.ascent, 13.32, accuracy: 0.01)
        XCTAssertEqual(display.descent, 4.00, accuracy: 0.01)
        XCTAssertEqual(display.width, 10, accuracy: 0.01)
    }

    func testSpacing() throws {
        let mathList = MTMathList()
        mathList.add(MTMathAtomFactory.atom(forCharacter: "x"))
        mathList.add(MTMathSpace(space: 9))
        mathList.add(MTMathAtomFactory.atom(forCharacter: "y"))
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 3)), "Got \(display.range) instead")
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 2);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        // The x is italicized
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTCTLineDisplay);
        let line2 = sub1 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        // The y is italicized
        XCTAssertEqual(line2.attributedString?.string, "𝑦")
        XCTAssertTrue(CGPointMake(21.44, 0).isEqual(to: line2.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(2, 1)), "Got \(line2.range) instead")
        XCTAssertFalse(line2.hasScript);
        
        let noSpace = MTMathList()
        noSpace.add(MTMathAtomFactory.atom(forCharacter: "x"))
        noSpace.add(MTMathAtomFactory.atom(forCharacter: "y"))
        
        let noSpaceDisplay = MTTypesetter.createLineForMathList(noSpace, font:self.font, style:.display)!
        
        // dimensions
        XCTAssertEqual(display.ascent, noSpaceDisplay.ascent, accuracy: 0.01)
        XCTAssertEqual(display.descent, noSpaceDisplay.descent, accuracy: 0.01)
        XCTAssertEqual(display.width, noSpaceDisplay.width + 10, accuracy: 0.01)
    }

    // For issue: https://github.com/kostub/iosMath/issues/5
    func testLargeRadicalDescent() throws {
        let list = MTMathListBuilder.build(fromString: "\\sqrt{\\frac{\\sqrt{\\frac{1}{2}} + 3}{\\sqrt{5}^x}}")
        let display = MTTypesetter.createLineForMathList(list, font:self.font, style:.display)!
        
        // dimensions
        XCTAssertEqual(display.ascent, 49.16, accuracy: 0.01)
        XCTAssertEqual(display.descent, 21.288, accuracy: 0.01)
        XCTAssertEqual(display.width, 82.569, accuracy: 0.01)
    }

    func testMathTable() throws {
        let c00 = MTMathAtomFactory.mathListForCharacters("1")
        let c01 = MTMathAtomFactory.mathListForCharacters("y+z")
        let c02 = MTMathAtomFactory.mathListForCharacters("y")
        
        let c11 = MTMathList()
        c11.add(MTMathAtomFactory.fraction(withNumeratorString: "1", denominatorString:"2x"))
        let c12 = MTMathAtomFactory.mathListForCharacters("x-y")
        
        let c20 = MTMathAtomFactory.mathListForCharacters("x+5")
        let c22 = MTMathAtomFactory.mathListForCharacters("12")

        let table = MTMathTable()
        table.set(cell: c00!, forRow:0, column:0)
        table.set(cell: c01!, forRow:0, column:1)
        table.set(cell: c02!, forRow:0, column:2)
        table.set(cell: c11,  forRow:1, column:1)
        table.set(cell: c12!, forRow:1, column:2)
        table.set(cell: c20!, forRow:2, column:0)
        table.set(cell: c22!, forRow:2, column:2)
        
        // alignments
        table.set(alignment: .right, forColumn:0)
        table.set(alignment: .left, forColumn:2)
        
        table.interColumnSpacing = 18; // 1 quad
        
        let mathList = MTMathList()
        mathList.add(table)
        
        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTMathListDisplay);
        
        let display2 = sub0 as! MTMathListDisplay
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointZero))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)))
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 3);
        let rowPos = [ 30.28, -2.68, -31.95 ]
        // alignment is right, center, left.
        let cellPos = [ [ 35.89, 65.89, 129.438 ], [ 45.89, 76.94, 129.438 ], [ 0, 87.66, 129.438] ]
        // check the 3 rows of the matrix
        for i in 0..<3 {
            let sub0i = display2.subDisplays[i];
            XCTAssertTrue(sub0i is MTMathListDisplay);
            
            let row = sub0i as! MTMathListDisplay
            XCTAssertEqual(row.type, .regular)
            XCTAssertTrue(CGPointMake(0, rowPos[i]).isEqual(to: row.position, accuracy: 0.01))
            XCTAssertTrue(NSEqualRanges(row.range, NSMakeRange(0, 3)));
            XCTAssertFalse(row.hasScript);
            XCTAssertEqual(row.index, NSNotFound);
            XCTAssertEqual(row.subDisplays.count, 3);
            
            for j in 0..<3 {
                let sub0ij = row.subDisplays[j];
                XCTAssertTrue(sub0ij is MTMathListDisplay);
                
                let col = sub0ij as! MTMathListDisplay
                XCTAssertEqual(col.type, .regular);
                XCTAssertTrue(CGPointMake(cellPos[i][j], 0).isEqual(to: col.position, accuracy: 0.01))
                XCTAssertFalse(col.hasScript)
                XCTAssertEqual(col.index, NSNotFound);
            }
        }
    }

    func testLatexSymbols() throws {
        // Test all latex symbols
        let allSymbols = MTMathAtomFactory.supportedLatexSymbolNames
        for symName in allSymbols {
            let list = MTMathList()
            let atom = MTMathAtomFactory.atom(forLatexSymbol:symName)
            XCTAssertNotNil(atom)
            if atom!.type >= .boundary {
                // Skip these types as they aren't symbols.
                continue;
            }
            
            list.add(atom)
            
            let display = MTTypesetter.createLineForMathList(list, font:self.font, style:.display)!
            XCTAssertNotNil(display, "Symbol \(symName)")
            
            XCTAssertEqual(display.type, .regular);
            XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
            XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)))
            XCTAssertFalse(display.hasScript);
            XCTAssertEqual(display.index, NSNotFound);
            XCTAssertEqual(display.subDisplays.count, 1, "Symbol \(symName)");
            
            let sub0 = display.subDisplays[0];
            if atom!.type == .largeOperator && atom!.nucleus.count == 1 {
                // These large operators are rendered differently;
                XCTAssertTrue(sub0 is MTGlyphDisplay);
                let glyph = sub0 as! MTGlyphDisplay
                XCTAssertTrue(CGPointEqualToPoint(glyph.position, CGPointZero))
                XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(0, 1)))
                XCTAssertFalse(glyph.hasScript);
            } else {
                XCTAssertTrue(sub0 is MTCTLineDisplay, "Symbol \(symName)");
                let line = sub0 as! MTCTLineDisplay
                XCTAssertEqual(line.atoms.count, 1);
                if atom!.type != .variable {
                    XCTAssertEqual(line.attributedString?.string, atom!.nucleus);
                }
                XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
                XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 1)))
                XCTAssertFalse(line.hasScript);
            }
            
            // dimensions
            XCTAssertEqual(display.ascent, sub0.ascent);
            XCTAssertEqual(display.descent, sub0.descent);
            XCTAssertEqual(display.width, sub0.width);
            
            // All chars will occupy some space.
            if atom!.nucleus != " " {
                // all chars except space have height
                XCTAssertGreaterThan(display.ascent + display.descent, 0, "Symbol \(symName)")
            }
            // all chars have a width.
            XCTAssertGreaterThan(display.width, 0);
        }
    }

    func testAtomWithAllFontStyles(_ atom:MTMathAtom?) throws {
        guard let atom = atom else { return }
        let fontStyles = [
            MTFontStyle.defaultStyle,
            .roman,
            .bold,
            .caligraphic,
            .typewriter,
            .italic,
            .sansSerif,
            .fraktur,
            .blackboard,
            .boldItalic,
        ]
        for fontStyle in fontStyles {
            let style = fontStyle
            let copy : MTMathAtom = atom.copy()
            copy.fontStyle = style
            let list = MTMathList(atom: copy)

            let display = MTTypesetter.createLineForMathList(list, font:self.font, style:.display)!
            XCTAssertNotNil(display, "Symbol \(atom.nucleus)")

            XCTAssertEqual(display.type, .regular);
            XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
            XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)))
            XCTAssertFalse(display.hasScript);
            XCTAssertEqual(display.index, NSNotFound);
            XCTAssertEqual(display.subDisplays.count, 1, "Symbol \(atom.nucleus)")

            let sub0 = display.subDisplays[0];
            XCTAssertTrue(sub0 is MTCTLineDisplay, "Symbol \(atom.nucleus)")
            let line = sub0 as! MTCTLineDisplay
            XCTAssertEqual(line.atoms.count, 1);
            XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
            XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 1)))
            XCTAssertFalse(line.hasScript);

            // dimensions
            XCTAssertEqual(display.ascent, sub0.ascent);
            XCTAssertEqual(display.descent, sub0.descent);
            XCTAssertEqual(display.width, sub0.width);

            // All chars will occupy some space.
            XCTAssertGreaterThan(display.ascent + display.descent, 0, "Symbol \(atom.nucleus)")
            // all chars have a width.
            XCTAssertGreaterThan(display.width, 0);
        }
    }

    func testVariables() throws {
        // Test all variables
        let allSymbols = MTMathAtomFactory.supportedLatexSymbolNames
        for symName in allSymbols {
            let atom = MTMathAtomFactory.atom(forLatexSymbol:symName)!
            XCTAssertNotNil(atom)
            if atom.type != .variable {
                // Skip these types as we are only interested in variables.
                continue;
            }
            try self.testAtomWithAllFontStyles(atom)
        }
        let alphaNum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
        let mathList = MTMathAtomFactory.mathListForCharacters(alphaNum)
        for atom in mathList!.atoms {
            try self.testAtomWithAllFontStyles(atom)
        }
    }

    func testStyleChanges() throws {
        let frac = MTMathAtomFactory.fraction(withNumeratorString: "1", denominatorString: "2")
        let list = MTMathList(atoms: [frac])
        let style = MTMathStyle(style: .text)
        let textList = MTMathList(atoms: [style, frac])
        
        // This should make the display same as text.
        let display = MTTypesetter.createLineForMathList(textList, font:self.font, style:.display)!
        let textDisplay = MTTypesetter.createLineForMathList(list, font:self.font, style:.text)!
        let originalDisplay = MTTypesetter.createLineForMathList(list, font:self.font, style:.display)!
        
        // Display should be the same as rendering the fraction in text style.
        XCTAssertEqual(display.ascent, textDisplay.ascent);
        XCTAssertEqual(display.descent, textDisplay.descent);
        XCTAssertEqual(display.width, textDisplay.width);
        
        // Original display should be larger than display since it is greater.
        XCTAssertGreaterThan(originalDisplay.ascent, display.ascent);
        XCTAssertGreaterThan(originalDisplay.descent, display.descent);
        XCTAssertGreaterThan(originalDisplay.width, display.width);
    }

    func testStyleMiddle() throws {
        let atom1 = MTMathAtomFactory.atom(forCharacter: "x")!
        let style1 = MTMathStyle(style: .script) as MTMathAtom
        let atom2 = MTMathAtomFactory.atom(forCharacter: "y")!
        let style2 = MTMathStyle(style: .scriptOfScript) as MTMathAtom
        let atom3 = MTMathAtomFactory.atom(forCharacter: "z")!
        let list = MTMathList(atoms: [atom1, style1, atom2, style2, atom3])
        
        let display = MTTypesetter.createLineForMathList(list, font:self.font, style:.display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 5)))
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 3);
        
        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTCTLineDisplay);
        let line = sub0 as! MTCTLineDisplay
        XCTAssertEqual(line.atoms.count, 1);
        XCTAssertEqual(line.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line.range, NSMakeRange(0, 1)))
        XCTAssertFalse(line.hasScript);
        
        let sub1 = display.subDisplays[1];
        XCTAssertTrue(sub1 is MTCTLineDisplay);
        let line1 = sub1 as! MTCTLineDisplay
        XCTAssertEqual(line1.atoms.count, 1);
        XCTAssertEqual(line1.attributedString?.string, "𝑦");
        XCTAssertTrue(NSEqualRanges(line1.range, NSMakeRange(2, 1)))
        XCTAssertFalse(line1.hasScript);
        
        let sub2 = display.subDisplays[2];
        XCTAssertTrue(sub2 is MTCTLineDisplay);
        let line2 = sub2 as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑧");
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(4, 1)))
        XCTAssertFalse(line2.hasScript);
    }

    func testAccent() throws {
        let mathList = MTMathList()
        let accent = MTMathAtomFactory.accent(withName: "hat")
        let inner = MTMathList()
        inner.add(MTMathAtomFactory.atom(forCharacter: "x"))
        accent?.innerList = inner;
        mathList.add(accent)

        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);

        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTAccentDisplay)
        let accentDisp = sub0 as! MTAccentDisplay
        XCTAssertTrue(NSEqualRanges(accentDisp.range, NSMakeRange(0, 1)));
        XCTAssertFalse(accentDisp.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(accentDisp.position, CGPointZero));
        XCTAssertNotNil(accentDisp.accentee);
        XCTAssertNotNil(accentDisp.accent);

        let display2 = accentDisp.accentee!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointZero))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);

        let subaccentee = display2.subDisplays[0];
        XCTAssertTrue(subaccentee is MTCTLineDisplay);
        let line2 = subaccentee as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 1);
        XCTAssertEqual(line2.attributedString?.string, "𝑥");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 1)));
        XCTAssertFalse(line2.hasScript);

        let glyph = accentDisp.accent!
        XCTAssertTrue(CGPointEqualToPoint(glyph.position, CGPointMake(11.86, 0)))
        XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(0, 1)))
        XCTAssertFalse(glyph.hasScript);

        // dimensions
        XCTAssertEqual(display.ascent, 14.68, accuracy: 0.01)
        XCTAssertEqual(display.descent, 0.22, accuracy: 0.01)
        XCTAssertEqual(display.width, 11.44, accuracy: 0.01)
    }

    func testWideAccent() throws {
        let mathList = MTMathList()
        let accent = MTMathAtomFactory.accent(withName: "hat")
        accent?.innerList = MTMathAtomFactory.mathListForCharacters("xyzw")
        mathList.add(accent)

        let display = MTTypesetter.createLineForMathList(mathList, font: self.font, style: .display)!
        XCTAssertNotNil(display);
        XCTAssertEqual(display.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(display.range, NSMakeRange(0, 1)));
        XCTAssertFalse(display.hasScript);
        XCTAssertEqual(display.index, NSNotFound);
        XCTAssertEqual(display.subDisplays.count, 1);

        let sub0 = display.subDisplays[0];
        XCTAssertTrue(sub0 is MTAccentDisplay)
        let accentDisp = sub0 as! MTAccentDisplay
        XCTAssertTrue(NSEqualRanges(accentDisp.range, NSMakeRange(0, 1)));
        XCTAssertFalse(accentDisp.hasScript);
        XCTAssertTrue(CGPointEqualToPoint(accentDisp.position, CGPointZero));
        XCTAssertNotNil(accentDisp.accentee);
        XCTAssertNotNil(accentDisp.accent);

        let display2 = accentDisp.accentee!
        XCTAssertEqual(display2.type, .regular);
        XCTAssertTrue(CGPointEqualToPoint(display2.position, CGPointZero))
        XCTAssertTrue(NSEqualRanges(display2.range, NSMakeRange(0, 4)));
        XCTAssertFalse(display2.hasScript);
        XCTAssertEqual(display2.index, NSNotFound);
        XCTAssertEqual(display2.subDisplays.count, 1);

        let subaccentee = display2.subDisplays[0];
        XCTAssertTrue(subaccentee is MTCTLineDisplay);
        let line2 = subaccentee as! MTCTLineDisplay
        XCTAssertEqual(line2.atoms.count, 4);
        XCTAssertEqual(line2.attributedString?.string, "𝑥𝑦𝑧𝑤");
        XCTAssertTrue(CGPointEqualToPoint(line2.position, CGPointZero));
        XCTAssertTrue(NSEqualRanges(line2.range, NSMakeRange(0, 4)));
        XCTAssertFalse(line2.hasScript);

        let glyph = accentDisp.accent!
        XCTAssertTrue(CGPointMake(3.47, 0).isEqual(to: glyph.position, accuracy: 0.01))
        XCTAssertTrue(NSEqualRanges(glyph.range, NSMakeRange(0, 1)))
        XCTAssertFalse(glyph.hasScript);

        // dimensions
        XCTAssertEqual(display.ascent, 14.98, accuracy: 0.01)
        XCTAssertEqual(display.descent, 4.10, accuracy: 0.01)
        XCTAssertEqual(display.width, 44.86, accuracy: 0.01)
    }

}

