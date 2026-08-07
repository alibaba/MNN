import XCTest
@testable import SwiftMath

//
//  MathRenderSwiftTests.swift
//  MathRenderSwiftTests
//
//  Created by Mike Griebling on 2023-01-02.
//

final class MTMathListBuilderTests: XCTestCase {

    func checkAtomTypes(_ list:MTMathList?, types:[MTMathAtomType], desc:String) {
        if let list = list {
            XCTAssertEqual(list.atoms.count, types.count, desc)
            for i in 0..<list.atoms.count {
                let atom = list.atoms[i]
                XCTAssertNotNil(atom, desc)
                XCTAssertEqual(atom.type, types[i], desc)
            }
        } else {
            XCTAssert(types.count == 0, "MathList should have no atoms!")
        }
    }
    
//    override func setUpWithError() throws {
//        // Put setup code here. This method is called before the invocation of each test method in the class.
//    }
//
//    override func tearDownWithError() throws {
//        // Put teardown code here. This method is called after the invocation of each test method in the class.
//    }
    
    struct TestRecord {
        let build : String
        let atomType : [MTMathAtomType]
        let types : [MTMathAtomType]
        let extra : [MTMathAtomType]
        let result : String
        
        init(build: String, atomType: [MTMathAtomType], types: [MTMathAtomType], extra: [MTMathAtomType] = [MTMathAtomType](), result: String) {
            self.build = build
            self.atomType = atomType
            self.types = types
            self.extra = extra
            self.result = result
        }
    }
    
    func getTestData() -> [TestRecord] {
        [
            TestRecord(build: "x", atomType: [.variable ], types: [], result: "x"),
            TestRecord(build: "1", atomType: [.number ] , types: [], result: "1"),
            TestRecord(build: "*", atomType: [.binaryOperator ] ,types: [], result:"*"),
            TestRecord(build: "+", atomType: [.binaryOperator ], types: [], result:"+"),
            TestRecord(build: ".", atomType: [.number ], types: [], result:"."),
            TestRecord(build: "(", atomType: [.open ], types: [], result:"(" ),
            TestRecord(build: ")", atomType: [.close ], types: [], result:")"),
            TestRecord(build: ",", atomType: [.punctuation], types: [], result:","),
            TestRecord(build: "!", atomType: [.close], types: [], result:"!"),
            TestRecord(build: "=", atomType: [.relation], types: [], result:"="),
            TestRecord(build: "x+2", atomType: [.variable, .binaryOperator, .number ], types: [], result:"x+2"),
            // spaces are ignored
            TestRecord(build: "(2.3 * 8)", atomType: [.open, .number, .number, .number, .binaryOperator, .number , .close ], types: [], result:"(2.3*8)"),
            // braces are just for grouping
            TestRecord(build: "5{3+4}", atomType: [.number, .number, .binaryOperator, .number], types: [], result:"53+4"),
            // commands
            TestRecord(build: "\\pi+\\theta\\geq 3",atomType: [.variable, .binaryOperator, .variable, .relation, .number], types: [], result:"\\pi +\\theta \\geq 3"),
            // aliases
            TestRecord(build: "\\pi\\ne 5 \\land 3", atomType: [.variable, .relation, .number, .binaryOperator, .number], types: [], result:"\\pi \\neq 5\\wedge 3"),
            // control space
            TestRecord(build: "x \\ y", atomType: [ .variable, .ordinary, .variable], types: [], result:"x\\  y"),
            // spacing
            TestRecord(build: "x \\quad y \\; z \\! q", atomType: [ .variable, .space, .variable,.space, .variable, .space, .variable], types: [], result:"x\\quad y\\; z\\! q")
        ]
    }
    
    func getTestDataSuperScript() -> [TestRecord] {
        [
            TestRecord(build: "x^2", atomType: [.variable], types: [.number], result: "x^{2}"),
            TestRecord(build: "x^23", atomType: [ .variable, .number ],  types: [ .number ], result: "x^{2}3"),
            TestRecord(build: "x^{23}", atomType: [ .variable ],  types: [ .number, .number ], result: "x^{23}"),
            TestRecord(build: "x^2^3", atomType: [ .variable, .ordinary ],  types: [ .number ], result: "x^{2}{}^{3}" ),
            TestRecord(build: "x^{2^3}", atomType: [ .variable ], types: [ .number], extra: [ .number ], result: "x^{2^{3}}"),
            TestRecord(build: "x^{^2*}", atomType: [ .variable ], types: [ .ordinary, .binaryOperator], extra:[ .number ], result:"x^{{}^{2}*}"),
            TestRecord(build: "^2",  atomType: [ .ordinary], types: [ .number ], result: "{}^{2}"),
            TestRecord(build: "{}^2",  atomType: [ .ordinary], types: [ .number ], result: "{}^{2}"),
            TestRecord(build: "x^^2", atomType: [ .variable, .ordinary ],  types: [ ], result: "x^{}{}^{2}"),
            TestRecord(build: "5{x}^2",  atomType: [ .number, .variable], types: [ ], result: "5x^{2}"),
        ]
    }
    
    func getTestDataSubScript() -> [TestRecord] {
        [
            TestRecord(build: "x_2", atomType: [.variable], types: [.number], result: "x_{2}"),
            TestRecord(build: "x_23", atomType: [ .variable, .number ],  types: [ .number ], result: "x_{2}3"),
            TestRecord(build: "x_{23}", atomType: [ .variable ],  types: [ .number, .number ], result: "x_{23}"),
            TestRecord(build: "x_2_3", atomType: [ .variable, .ordinary ],  types: [ .number ], result: "x_{2}{}_{3}" ),
            TestRecord(build: "x_{2_3}", atomType: [ .variable ], types: [ .number], extra: [ .number ], result: "x_{2_{3}}"),
            TestRecord(build: "x_{_2*}", atomType: [ .variable ], types: [ .ordinary, .binaryOperator], extra:[ .number ], result:"x_{{}_{2}*}"),
            TestRecord(build: "_2",  atomType: [ .ordinary], types: [ .number ], result: "{}_{2}"),
            TestRecord(build: "{}_2",  atomType: [ .ordinary], types: [ .number ], result: "{}_{2}"),
            TestRecord(build: "x__2", atomType: [ .variable, .ordinary ],  types: [ ], result: "x_{}{}_{2}"),
            TestRecord(build: "5{x}_2",  atomType: [ .number, .variable], types: [ ], result: "5x_{2}"),
        ]
    }
    
    func getTestDataSuperSubScript() -> [TestRecord] {
        [
            TestRecord(build: "x_2^*", atomType: [.variable], types: [.number], extra: [.binaryOperator], result: "x^{*}_{2}"),
            TestRecord(build: "x^*_2", atomType: [.variable], types: [.number], extra: [.binaryOperator], result: "x^{*}_{2}"),
            TestRecord(build: "x_^*", atomType: [.variable], types: [ ], extra: [.binaryOperator], result: "x^{*}_{}"),
            TestRecord(build: "x^_2", atomType: [.variable], types: [.number], result: "x^{}_{2}"),
            TestRecord(build: "x_{2^*}", atomType: [.variable], types: [.number], result: "x_{2^{*}}"),
            TestRecord(build: "x^{*_2}", atomType: [.variable], types: [ ], extra: [.binaryOperator], result: "x^{*_{2}}"),
            TestRecord(build: "_2^*", atomType: [.ordinary], types: [.number], extra: [.binaryOperator], result: "{}^{*}_{2}")
        ]
    }
    
    struct TestRecord2 {
        let build : String
        let type1 : [MTMathAtomType]
        let number : Int
        let type2 : [MTMathAtomType]
        let left : String
        let right : String
        let result : String
    }
    
    func getTestDataLeftRight() -> [TestRecord2] {
        [
            TestRecord2(build: "\\left( 2 \\right)", type1: [ .inner ], number: 0, type2: [ .number], left: "(", right: ")", result: "\\left( 2\\right) "),
            // spacing
            TestRecord2(build: "\\left ( 2 \\right )", type1: [ .inner ], number: 0, type2: [ .number], left: "(", right: ")", result: "\\left( 2\\right) "),
            // commands
            TestRecord2(build: "\\left\\{ 2 \\right\\}", type1: [ .inner ], number: 0, type2: [ .number], left: "{", right: "}", result: "\\left\\{ 2\\right\\} "),
            // complex commands
            TestRecord2(build: "\\left\\langle x \\right\\rangle", type1: [ .inner ], number: 0, type2: [ .variable], left: "\u{2329}", right: "\u{232A}", result: "\\left< x\\right> "),
            // bars
            TestRecord2(build: "\\left| x \\right\\|", type1: [ .inner ], number: 0, type2: [ .variable], left: "|", right: "\u{2016}", result: "\\left| x\\right\\| "),
            // inner in between
            TestRecord2(build: "5 + \\left( 2 \\right) - 2", type1: [ .number, .binaryOperator, .inner, .binaryOperator, .number ], number: 2, type2: [ .number], left: "(", right: ")", result: "5+\\left( 2\\right) -2"),
            // long inner
            TestRecord2(build: "\\left( 2 + \\frac12\\right)", type1: [ .inner ], number: 0, type2: [ .number, .binaryOperator, .fraction], left: "(", right: ")", result: "\\left( 2+\\frac{1}{2}\\right) "),
            // nested
            TestRecord2(build: "\\left[ 2 + \\left|\\frac{-x}{2}\\right| \\right]", type1: [ .inner ], number: 0, type2: [ .number, .binaryOperator, .inner], left: "[", right: "]", result: "\\left[ 2+\\left| \\frac{-x}{2}\\right| \\right] "),
            // With scripts
            TestRecord2(build: "\\left( 2 \\right)^2", type1: [ .inner ], number: 0, type2: [ .number], left: "(", right: ")", result: "\\left( 2\\right) ^{2}"),
            // Scripts on left
            TestRecord2(build: "\\left(^2 \\right )", type1: [ .inner], number: 0, type2: [ .ordinary], left: "(", right: ")", result: "\\left( {}^{2}\\right) "),
            // Dot
            TestRecord2(build: "\\left( 2 \\right.", type1: [ .inner], number: 0, type2: [ .number], left: "(", right: "", result: "\\left( 2\\right. ")
        ]
    }
    
    func getTestDataParseErrors() -> [(String, MTParseErrors)] {
        return [
                  ("}a", .mismatchBraces),
                  ("\\notacommand", .invalidCommand),
                  ("\\sqrt[5+3", .characterNotFound),
                  ("{5+3", .mismatchBraces),
                  ("5+3}", .mismatchBraces),
                  ("{1+\\frac{3+2", .mismatchBraces),
                  ("1+\\left", .missingDelimiter),
                  ("\\left(\\frac12\\right", .missingDelimiter),
                  ("\\left 5 + 3 \\right)", .invalidDelimiter),
                  ("\\left(\\frac12\\right + 3", .invalidDelimiter),
                  ("\\left\\lmoustache 5 + 3 \\right)", .invalidDelimiter),
                  ("\\left(\\frac12\\right\\rmoustache + 3", .invalidDelimiter),
                  ("5 + 3 \\right)", .missingLeft),
                  ("\\left(\\frac12", .missingRight),
                  ("\\left(5 + \\left| \\frac12 \\right)", .missingRight),
                  ("5+ \\left|\\frac12\\right| \\right)", .missingLeft),
                  ("\\begin matrix \\end matrix", .characterNotFound), // missing {
                  ("\\begin", .characterNotFound), // missing {
                  ("\\begin{", .characterNotFound), // missing }
                  ("\\begin{matrix parens}", .characterNotFound), // missing } (no spaces in env)
                  ("\\begin{matrix} x", .missingEnd),
                  ("\\begin{matrix} x \\end", .characterNotFound), // missing {
                  ("\\begin{matrix} x \\end + 3", .characterNotFound), // missing {
                  ("\\begin{matrix} x \\end{", .characterNotFound), // missing }
                  ("\\begin{matrix} x \\end{matrix + 3", .characterNotFound), // missing }
                  ("\\begin{matrix} x \\end{pmatrix}", .invalidEnv),
                  ("x \\end{matrix}", .missingBegin),
                  ("\\begin{notanenv} x \\end{notanenv}", .invalidEnv),
                  ("\\begin{matrix} \\notacommand \\end{matrix}", .invalidCommand),
                  ("\\begin{displaylines} x & y \\end{displaylines}", .invalidNumColumns),
                  ("\\begin{eqalign} x \\end{eqalign}", .invalidNumColumns),
                  ("\\nolimits", .invalidLimits),
                  ("\\frac\\limits{1}{2}", .invalidLimits),
                  ("&\\begin", .characterNotFound),
                  ("x & y \\\\ z & w \\end{matrix}", .invalidEnv)
            ]
    }
    
    func testBuilder() throws {
        let data = getTestData()
        for testCase in data {
            let str = testCase.build
            var error : NSError? = nil
            let list = MTMathListBuilder.build(fromString: str, error: &error)
            XCTAssertNil(error)
            let desc = "Error for string:\(str)"
            let atomTypes = testCase.atomType
            self.checkAtomTypes(list, types:atomTypes, desc:desc)
            
            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, testCase.result, desc)
        }
    }

    func testSuperScript() throws {
        let data = getTestDataSuperScript()
        for testCase in data {
            let str = testCase.build
            var error : NSError? = nil
            let list = MTMathListBuilder.build(fromString: str, error:&error)
            XCTAssertNil(error)
            let desc = "Error for string:\(str)"
            let atomTypes = testCase.atomType
            checkAtomTypes(list, types:atomTypes, desc:desc)
            
            // get the first atom
            let first = list!.atoms[0]
            // check it's superscript
            let types = testCase.types
            if types.count > 0 {
                XCTAssertNotNil(first.superScript, desc)
            }
            let superlist = first.superScript
            checkAtomTypes(superlist, types:types, desc:desc)
            
            if !testCase.extra.isEmpty {
                // one more level
                let superFirst = superlist!.atoms[0]
                let supersuperList = superFirst.superScript
                checkAtomTypes(supersuperList, types:testCase.extra, desc:desc)
            }
            
            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, testCase.result, desc)
        }
    }
    
    func testSubScript() throws {
        let data = getTestDataSubScript()
        for testCase in data {
            let str = testCase.build
            var error : NSError? = nil
            let list = MTMathListBuilder.build(fromString: str, error:&error)
            XCTAssertNil(error)
            let desc = "Error for string:\(str)"
            let atomTypes = testCase.atomType
            checkAtomTypes(list, types:atomTypes, desc:desc)
            
            // get the first atom
            let first = list!.atoms[0]
            // check it's superscript
            let types = testCase.types
            if (types.count > 0) {
                XCTAssertNotNil(first.subScript, desc);
            }
            let sublist = first.subScript
            checkAtomTypes(sublist, types:types, desc:desc)
            
            if !testCase.extra.isEmpty {
                // one more level
                let subFirst = sublist!.atoms[0]
                let subsubList = subFirst.subScript
                checkAtomTypes(subsubList, types:testCase.extra, desc:desc)
            }
            
            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, testCase.result, desc)
        }
    }
    
    func testSuperSubScript() throws {
        let data = getTestDataSuperSubScript()
        for testCase in data {
            let str = testCase.build
            var error : NSError? = nil
            let list = MTMathListBuilder.build(fromString: str, error:&error)
            XCTAssertNil(error)
            let desc = "Error for string:\(str)"
            let atomTypes = testCase.atomType
            checkAtomTypes(list, types:atomTypes, desc:desc)
            
            // get the first atom
            let first = list!.atoms[0]
            // check its subscript
            let sub = testCase.types
            if sub.count > 0 {
                XCTAssertNotNil(first.subScript, desc)
                let sublist = first.subScript
                checkAtomTypes(sublist, types: sub, desc: desc)
            }
            let sup = testCase.extra
            if sup.count > 0 {
                XCTAssertNotNil(first.superScript, desc)
                let sublist = first.superScript
                checkAtomTypes(sublist, types: sup, desc: desc)
            }

            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, testCase.result, desc)
        }
    }
    
    func testSymbols() throws {
        let str = "5\\times3^{2\\div2}";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 3, desc)
        var atom = list.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "5", desc)
        atom = list.atoms[1];
        XCTAssertEqual(atom.type, .binaryOperator, desc)
        XCTAssertEqual(atom.nucleus, "\u{00D7}", desc)
        atom = list.atoms[2];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "3", desc)
        
        // super script
        let superList = atom.superScript!
        XCTAssertNotNil(superList, desc)
        XCTAssertEqual((superList.atoms.count), 3, desc)
        atom = superList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "2", desc)
        atom = superList.atoms[1];
        XCTAssertEqual(atom.type, .binaryOperator, desc)
        XCTAssertEqual(atom.nucleus, "\u{00F7}", desc)
        atom = superList.atoms[2];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "2", desc)
    }

    func testFrac() throws {
        let str = "\\frac1c";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc)
        XCTAssertEqual(frac.nucleus, "", desc)
        XCTAssertTrue(frac.hasRule);
        XCTAssertTrue(frac.rightDelimiter.isEmpty)
        XCTAssertTrue(frac.leftDelimiter.isEmpty)
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "1", desc)
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "c", desc)
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\frac{1}{c}", desc)
    }

    func testFracInFrac() throws {
        let str = "\\frac1\\frac23";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        var frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type,  .fraction, desc)
        XCTAssertEqual(frac.nucleus, "", desc)
        XCTAssertTrue(frac.hasRule);
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "1", desc)
        
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        frac = subList.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type,  .fraction, desc)
        XCTAssertEqual(frac.nucleus, "", desc)
        
        subList = frac.numerator!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "2", desc)
        
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "3", desc)
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\frac{1}{\\frac{2}{3}}", desc)
    }

    func testSqrt() throws {
        let str = "\\sqrt2";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        let rad = list.atoms[0] as! MTRadical
        XCTAssertEqual(rad.type, .radical, desc)
        XCTAssertEqual(rad.nucleus, "", desc)

        let subList = rad.radicand!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        let atom = subList.atoms[0]
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "2", desc)

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sqrt{2}", desc)
    }

    func testSqrtInSqrt() throws {
        let str = "\\sqrt\\sqrt2";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        var rad = list.atoms[0] as! MTRadical
        XCTAssertEqual(rad.type, .radical, desc)
        XCTAssertEqual(rad.nucleus, "", desc)

        var subList = rad.radicand!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        rad = subList.atoms[0] as! MTRadical
        XCTAssertEqual(rad.type, .radical, desc)
        XCTAssertEqual(rad.nucleus, "", desc)


        subList = rad.radicand!
        XCTAssertNotNil(subList, desc)
        XCTAssertEqual((subList.atoms.count), 1, desc)
        let atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc)
        XCTAssertEqual(atom.nucleus, "2", desc)

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sqrt{\\sqrt{2}}", desc)
    }

    func testRad() throws {
        let str = "\\sqrt[3]2";
        let list = MTMathListBuilder.build(fromString: str)!

        XCTAssertNotNil(list);
        XCTAssertEqual((list.atoms.count), 1);
        let rad = list.atoms[0] as! MTRadical
        XCTAssertEqual(rad.type, .radical);
        XCTAssertEqual(rad.nucleus, "");

        var subList = rad.radicand!
        XCTAssertNotNil(subList);
        XCTAssertEqual((subList.atoms.count), 1);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number);
        XCTAssertEqual(atom.nucleus, "2");

        subList = rad.degree!
        XCTAssertNotNil(subList);
        XCTAssertEqual((subList.atoms.count), 1);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number);
        XCTAssertEqual(atom.nucleus, "3");

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sqrt[3]{2}");
    }
    
    func testSqrtWithoutRadicand() throws {
        let str = "\\sqrt"
        let list = try XCTUnwrap(MTMathListBuilder.build(fromString: str))
        
        XCTAssertEqual(list.atoms.count, 1)
        let rad = try XCTUnwrap(list.atoms.first as? MTRadical)
        XCTAssertEqual(rad.type, .radical)
        XCTAssertEqual(rad.nucleus, "")
        
        XCTAssertEqual(rad.radicand?.atoms.isEmpty, true)
        XCTAssertNil(rad.degree)
        
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sqrt{}")
    }
    
    func testSqrtWithDegreeWithoutRadicand() throws {
        let str = "\\sqrt[3]"
        let list = try XCTUnwrap(MTMathListBuilder.build(fromString: str))
        
        XCTAssertEqual(list.atoms.count, 1)
        let rad = try XCTUnwrap(list.atoms.first as? MTRadical)
        XCTAssertEqual(rad.type, .radical)
        XCTAssertEqual(rad.nucleus, "")
        
        XCTAssertEqual(rad.radicand?.atoms.isEmpty, true)
        
        let subList = try XCTUnwrap(rad.degree)
        XCTAssertEqual(subList.atoms.count, 1)
        let atom = try XCTUnwrap(subList.atoms.first)
        XCTAssertEqual(atom.type, .number)
        XCTAssertEqual(atom.nucleus, "3")
        
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sqrt[3]{}");
    }
    
    func testLeftRight() throws {
        let data = getTestDataLeftRight()
        for testCase in data {
            let str = testCase.build
            
            var error : NSError? = nil
            let list = MTMathListBuilder.build(fromString: str, error: &error)!
            
            XCTAssertNotNil(list, str);
            XCTAssertNil(error, str);
            
            checkAtomTypes(list, types:testCase.type1, desc:"\(str) outer")

            let innerLoc = testCase.number
            let inner = list.atoms[innerLoc] as! MTInner
            XCTAssertEqual(inner.type, .inner, str);
            XCTAssertEqual(inner.nucleus, "", str);
        
            let innerList = inner.innerList!
            XCTAssertNotNil(innerList, str);
            checkAtomTypes(innerList, types:testCase.type2, desc:"\(str) inner")
            
            XCTAssertNotNil(inner.leftBoundary, str);
            XCTAssertEqual(inner.leftBoundary!.type, .boundary, str);
            XCTAssertEqual(inner.leftBoundary!.nucleus, testCase.left, str);
            
            XCTAssertNotNil(inner.rightBoundary, str);
            XCTAssertEqual(inner.rightBoundary!.type, .boundary, str);
            XCTAssertEqual(inner.rightBoundary!.nucleus, testCase.right, str);
            
            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, testCase.result, str);
        }
    }
    
    func testOver() throws {
        let str = "1 \\over c";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertTrue(frac.hasRule);
        XCTAssertTrue(frac.rightDelimiter.isEmpty)
        XCTAssertTrue(frac.leftDelimiter.isEmpty)
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc);
        XCTAssertEqual(atom.nucleus, "1", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "c", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\frac{1}{c}", desc);
    }

    func testOverInParens() throws {
        let str = "5 + {1 \\over c} + 8";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 5, desc);
        let types = [MTMathAtomType.number, .binaryOperator, .fraction, .binaryOperator, .number]
        self.checkAtomTypes(list, types:types, desc:desc)
        
        let frac = list.atoms[2] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertTrue(frac.hasRule);
        XCTAssertTrue(frac.rightDelimiter.isEmpty)
        XCTAssertTrue(frac.leftDelimiter.isEmpty)
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc);
        XCTAssertEqual(atom.nucleus, "1", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "c", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "5+\\frac{1}{c}+8", desc);
    }

    func testAtop() throws {
        let str = "1 \\atop c";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertFalse(frac.hasRule);
        XCTAssertTrue(frac.rightDelimiter.isEmpty)
        XCTAssertTrue(frac.leftDelimiter.isEmpty)
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc);
        XCTAssertEqual(atom.nucleus, "1", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "c", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "{1 \\atop c}", desc);
    }

    func testAtopInParens() throws {
        let str = "5 + {1 \\atop c} + 8";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 5, desc);
        let types = [MTMathAtomType.number, .binaryOperator, .fraction, .binaryOperator, .number]
        self.checkAtomTypes(list, types:types, desc:desc)
        
        let frac = list.atoms[2] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertFalse(frac.hasRule);
        XCTAssertTrue(frac.rightDelimiter.isEmpty)
        XCTAssertTrue(frac.leftDelimiter.isEmpty)
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc);
        XCTAssertEqual(atom.nucleus, "1", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "c", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "5+{1 \\atop c}+8", desc);
    }

    func testChoose() throws {
        let str = "n \\choose k";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertFalse(frac.hasRule);
        XCTAssertEqual(frac.rightDelimiter, ")");
        XCTAssertEqual(frac.leftDelimiter, "(");
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "n", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "k", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "{n \\choose k}", desc);
    }

    func testBrack() throws {
        let str = "n \\brack k";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertFalse(frac.hasRule);
        XCTAssertEqual(frac.rightDelimiter, "]");
        XCTAssertEqual(frac.leftDelimiter, "[");
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "n", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "k", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "{n \\brack k}", desc);
    }

    func testBrace() throws {
        let str = "n \\brace k";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertFalse(frac.hasRule);
        XCTAssertEqual(frac.rightDelimiter, "}");
        XCTAssertEqual(frac.leftDelimiter, "{");
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "n", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "k", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "{n \\brace k}", desc);
    }

    func testBinom() throws {
        let str = "\\binom{n}{k}";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let frac = list.atoms[0] as! MTFraction
        XCTAssertEqual(frac.type, .fraction, desc);
        XCTAssertEqual(frac.nucleus, "", desc);
        XCTAssertFalse(frac.hasRule);
        XCTAssertEqual(frac.rightDelimiter, ")");
        XCTAssertEqual(frac.leftDelimiter, "(");
        
        var subList = frac.numerator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "n", desc);
        
        atom = list.atoms[0];
        subList = frac.denominator!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "k", desc);
        
        // convert it back to latex (binom converts to choose)
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "{n \\choose k}", desc);
    }

    func testOverLine() throws {
        let str = "\\overline 2";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let over = list.atoms[0] as! MTOverLine
        XCTAssertEqual(over.type, .overline, desc);
        XCTAssertEqual(over.nucleus, "", desc);
        
        let subList = over.innerList!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        let atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc);
        XCTAssertEqual(atom.nucleus, "2", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\overline{2}", desc);
    }

    func testUnderline() throws {
        let str = "\\underline 2";
        let  list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let under = list.atoms[0] as! MTUnderLine
        XCTAssertEqual(under.type, .underline, desc);
        XCTAssertEqual(under.nucleus, "", desc);
        
        let subList = under.innerList!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        let atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .number, desc);
        XCTAssertEqual(atom.nucleus, "2", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\underline{2}", desc);
    }

    func testAccent() throws {
        let str = "\\bar x";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let accent = list.atoms[0] as! MTAccent
        XCTAssertEqual(accent.type, .accent, desc);
        XCTAssertEqual(accent.nucleus, "\u{0304}", desc);
        
        let subList = accent.innerList!
        XCTAssertNotNil(subList, desc);
        XCTAssertEqual((subList.atoms.count), 1, desc);
        let atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc);
        XCTAssertEqual(atom.nucleus, "x", desc);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\bar{x}", desc);
    }
	
	func testAccentedCharacter() throws {
		let str = "á"
		let list = MTMathListBuilder.build(fromString: str)!
		let desc = "Error for string:\(str)"
		
		XCTAssertNotNil(list, desc)
		XCTAssertEqual((list.atoms.count), 1, desc)
		let accent = list.atoms[0] as! MTAccent
		XCTAssertEqual(accent.type, .accent, desc)
		XCTAssertEqual(accent.nucleus, "\u{0301}", desc)
		
		let subList = accent.innerList!
		XCTAssertNotNil(subList, desc)
		XCTAssertEqual((subList.atoms.count), 1, desc)
		let atom = subList.atoms[0]
		XCTAssertEqual(atom.type, .variable, desc)
		XCTAssertEqual(atom.nucleus, "a", desc)
		
		// convert it back to latex
		let latex = MTMathListBuilder.mathListToString(list)
		XCTAssertEqual(latex, "\\acute{a}", desc)
	}

    func testMathSpace() throws {
        let str = "\\!";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 1, desc);
        let space = list.atoms[0] as! MTMathSpace
        XCTAssertEqual(space.type, .space, desc);
        XCTAssertEqual(space.nucleus, "", desc);
        XCTAssertEqual(space.space, -3);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\! ", desc);
    }

    func testMathStyle() throws {
        let str = "\\textstyle y \\scriptstyle x";
        let list = MTMathListBuilder.build(fromString:str)!
        let desc = "Error for string:\(str)"
        
        XCTAssertNotNil(list, desc);
        XCTAssertEqual((list.atoms.count), 4, desc);
        let style = list.atoms[0] as! MTMathStyle
        XCTAssertEqual(style.type, .style, desc);
        XCTAssertEqual(style.nucleus, "", desc);
        XCTAssertEqual(style.style, .text);
        
        let style2 = list.atoms[2] as! MTMathStyle
        XCTAssertEqual(style2.type, .style, desc);
        XCTAssertEqual(style2.nucleus, "", desc);
        XCTAssertEqual(style2.style, .script);
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\textstyle y\\scriptstyle x", desc);
    }

    func testMatrix() throws {
        let str = "\\begin{matrix} x & y \\\\ z & w \\end{matrix}";
        let list = MTMathListBuilder.build(fromString:str)!
        
        XCTAssertNotNil(list);
        XCTAssertEqual((list.atoms.count), 1);
        let table = list.atoms[0] as! MTMathTable
        XCTAssertEqual(table.type, .table);
        XCTAssertEqual(table.nucleus, "");
        XCTAssertEqual(table.environment, "matrix");
        XCTAssertEqual(table.interRowAdditionalSpacing, 0);
        XCTAssertEqual(table.interColumnSpacing, 18);
        XCTAssertEqual(table.numRows, 2);
        XCTAssertEqual(table.numColumns, 2);
        
        for i in 0..<2 {
            let alignment = table.get(alignmentForColumn:i)
            XCTAssertEqual(alignment, .center);
            for j in 0..<2 {
                let cell = table.cells[j][i];
                XCTAssertEqual(cell.atoms.count, 2);
                let style = cell.atoms[0] as! MTMathStyle
                XCTAssertEqual(style.type, .style);
                XCTAssertEqual(style.style, .text);
                
                let atom = cell.atoms[1];
                XCTAssertEqual(atom.type, .variable);
            }
        }
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\begin{matrix}x&y\\\\ z&w\\end{matrix}");
    }

    func testPMatrix() throws {
        let str = "\\begin{pmatrix} x & y \\\\ z & w \\end{pmatrix}";
        let  list = MTMathListBuilder.build(fromString:str)!
        
        XCTAssertNotNil(list);
        XCTAssertEqual((list.atoms.count), 1);
        let inner = list.atoms[0] as! MTInner
        XCTAssertEqual(inner.type, .inner, str);
        XCTAssertEqual(inner.nucleus, "", str);
        
        let innerList = inner.innerList!
        XCTAssertNotNil(innerList, str);
        
        XCTAssertNotNil(inner.leftBoundary, str);
        XCTAssertEqual(inner.leftBoundary!.type, .boundary, str);
        XCTAssertEqual(inner.leftBoundary!.nucleus, "(", str);
        
        XCTAssertNotNil(inner.rightBoundary, str);
        XCTAssertEqual(inner.rightBoundary!.type, .boundary, str);
        XCTAssertEqual(inner.rightBoundary!.nucleus, ")", str);
        
        XCTAssertEqual((innerList.atoms.count), 1);
        let table = innerList.atoms[0] as! MTMathTable
        XCTAssertEqual(table.type, .table);
        XCTAssertEqual(table.nucleus, "");
        XCTAssertEqual(table.environment, "matrix");
        XCTAssertEqual(table.interRowAdditionalSpacing, 0);
        XCTAssertEqual(table.interColumnSpacing, 18);
        XCTAssertEqual(table.numRows, 2);
        XCTAssertEqual(table.numColumns, 2);
        
        for i in 0..<2 {
            let alignment = table.get(alignmentForColumn:i)
            XCTAssertEqual(alignment, .center);
            for j in 0..<2 {
                let  cell = table.cells[j][i];
                XCTAssertEqual(cell.atoms.count, 2);
                let style = cell.atoms[0] as! MTMathStyle
                XCTAssertEqual(style.type, .style);
                XCTAssertEqual(style.style, .text);
                
                let atom = cell.atoms[1];
                XCTAssertEqual(atom.type, .variable);
            }
        }
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\left( \\begin{matrix}x&y\\\\ z&w\\end{matrix}\\right) ");
    }

    func testDefaultTable() throws {
        let str = "x \\\\ y";
        let list = MTMathListBuilder.build(fromString:str)!
        
        XCTAssertNotNil(list);
        XCTAssertEqual(list.atoms.count, 1);
        let table = list.atoms[0] as! MTMathTable
        XCTAssertEqual(table.type, .table);
        XCTAssertEqual(table.nucleus, "");
        XCTAssertTrue(table.environment.isEmpty);
        XCTAssertEqual(table.interRowAdditionalSpacing, 1);
        XCTAssertEqual(table.interColumnSpacing, 0);
        XCTAssertEqual(table.numRows, 2);
        XCTAssertEqual(table.numColumns, 1);
        
        for i in 0..<1 {
            let alignment = table.get(alignmentForColumn: i)
            XCTAssertEqual(alignment, .left);
            for j in 0..<2 {
                let  cell = table.cells[j][i];
                XCTAssertEqual(cell.atoms.count, 1);
                let atom = cell.atoms[0];
                XCTAssertEqual(atom.type, .variable);
            }
        }
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "x\\\\ y");
    }

    func testDefaultTableWithCols() throws {
        let str = "x & y \\\\ z & w";
        let list = MTMathListBuilder.build(fromString:str)!
        
        XCTAssertNotNil(list);
        XCTAssertEqual((list.atoms.count), 1);
        let table = list.atoms[0] as! MTMathTable
        XCTAssertEqual(table.type, .table);
        XCTAssertEqual(table.nucleus, "");
        XCTAssertTrue(table.environment.isEmpty);
        XCTAssertEqual(table.interRowAdditionalSpacing, 1);
        XCTAssertEqual(table.interColumnSpacing, 0);
        XCTAssertEqual(table.numRows, 2);
        XCTAssertEqual(table.numColumns, 2);
        
        for i in 0..<2 {
            let alignment = table.get(alignmentForColumn:i)
            XCTAssertEqual(alignment, .left);
            for j in 0..<2 {
                let  cell = table.cells[j][i];
                XCTAssertEqual(cell.atoms.count, 1);
                let atom = cell.atoms[0];
                XCTAssertEqual(atom.type, .variable);
            }
        }
        
        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "x&y\\\\ z&w");
    }

    func testEqalign() throws {
        let str1 = "\\begin{eqalign}x&y\\\\ z&w\\end{eqalign}";
        let str2 = "\\begin{split}x&y\\\\ z&w\\end{split}";
        let str3 = "\\begin{aligned}x&y\\\\ z&w\\end{aligned}";
        for str in [str1, str2, str3] {
            let list = MTMathListBuilder.build(fromString:str)!
            
            XCTAssertNotNil(list);
            XCTAssertEqual((list.atoms.count), 1);
            let table = list.atoms[0] as! MTMathTable
            XCTAssertEqual(table.type, .table);
            XCTAssertEqual(table.nucleus, "");
            XCTAssertEqual(table.interRowAdditionalSpacing, 1);
            XCTAssertEqual(table.interColumnSpacing, 0);
            XCTAssertEqual(table.numRows, 2);
            XCTAssertEqual(table.numColumns, 2);
            
            for i in 0..<2 {
                let alignment = table.get(alignmentForColumn:i)
                XCTAssertEqual(alignment, (i == 0) ? .right: .left);
                for j in 0..<2 {
                    let  cell = table.cells[j][i];
                    if (i == 0) {
                        XCTAssertEqual(cell.atoms.count, 1);
                        let atom = cell.atoms[0];
                        XCTAssertEqual(atom.type, .variable);
                    } else {
                        XCTAssertEqual(cell.atoms.count, 2);
                        self.checkAtomTypes(cell, types:[.ordinary, .variable], desc:str)
                    }
                }
            }
            
            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, str);
        }
    }

    func testDisplayLines() throws {
        let str1 = "\\begin{displaylines}x\\\\ y\\end{displaylines}";
        let str2 = "\\begin{gather}x\\\\ y\\end{gather}";
        for  str in [str1, str2] {
            let list = MTMathListBuilder.build(fromString:str)
            
            XCTAssertNotNil(list)
            XCTAssertEqual(list?.atoms.count, 1);
            let table = list?.atoms[0] as! MTMathTable
            XCTAssertEqual(table.type, .table);
            XCTAssertEqual(table.nucleus, "");
            XCTAssertEqual(table.interRowAdditionalSpacing, 1);
            XCTAssertEqual(table.interColumnSpacing, 0);
            XCTAssertEqual(table.numRows, 2);
            XCTAssertEqual(table.numColumns, 1);
            
            for i in 0..<1 {
                let alignment = table.get(alignmentForColumn:i)
                XCTAssertEqual(alignment, .center);
                for j in 0..<2 {
                    let cell = table.cells[j][i];
                    XCTAssertEqual(cell.atoms.count, 1);
                    let atom = cell.atoms[0];
                    XCTAssertEqual(atom.type, .variable);
                }
            }
            
            // convert it back to latex
            let latex = MTMathListBuilder.mathListToString(list)
            XCTAssertEqual(latex, str);
        }
    }

    func testErrors() throws {
        let data = getTestDataParseErrors()
        for testCase in data {
            let str = testCase.0
            var error : NSError? = nil
            let list = MTMathListBuilder.build(fromString: str, error:&error)
            let desc = "Error for string:\(str)"
            XCTAssertNil(list, desc)
            XCTAssertNotNil(error, desc)
            XCTAssertEqual(error!.domain, MTParseError, desc)
            let num = testCase.1
            XCTAssertEqual(error!.code, num.rawValue, desc)
        }
    }

    func testCustom() throws {
        let str = "\\lcm(a,b)";
        var error : NSError? = nil
        var list = MTMathListBuilder.build(fromString: str, error:&error)
        XCTAssertNil(list)
        XCTAssertNotNil(error)

        MTMathAtomFactory.add(latexSymbol: "lcm", value: MTMathAtomFactory.operatorWithName("lcm", limits:false))
        error = nil
        list = MTMathListBuilder.build(fromString: str, error:&error)
        let atomTypes = [MTMathAtomType.largeOperator, .open, .variable, .punctuation, .variable, .close]
        self.checkAtomTypes(list, types:atomTypes, desc:"Error for lcm")

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\lcm (a,b)");
    }

    func testFontSingle() throws {
        let str = "\\mathbf x";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual(list.atoms.count, 1, desc)
        let atom = list.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "x", desc)
        XCTAssertEqual(atom.fontStyle, .bold)

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\mathbf{x}", desc)
    }

    func testFontOneChar() throws {
        let str = "\\cal xy";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 2, desc)
        var atom = list.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "x", desc)
        XCTAssertEqual(atom.fontStyle, .caligraphic);

        atom = list.atoms[1];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "y", desc)
        XCTAssertEqual(atom.fontStyle, .defaultStyle);

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\mathcal{x}y", desc)
    }

    func testFontMultipleChars() throws {
        let str = "\\frak{xy}";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 2, desc)
        var atom = list.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "x", desc)
        XCTAssertEqual(atom.fontStyle, .fraktur);

        atom = list.atoms[1];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "y", desc)
        XCTAssertEqual(atom.fontStyle, .fraktur);

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\mathfrak{xy}", desc)
    }

    func testFontOneCharInside() throws {
        let str = "\\sqrt \\mathrm x y";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 2, desc)

        let rad = list.atoms[0] as! MTRadical
        XCTAssertEqual(rad.type, .radical, desc)
        XCTAssertEqual(rad.nucleus, "", desc)

        let subList = rad.radicand!
        var atom = subList.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "x", desc)
        XCTAssertEqual(atom.fontStyle, .roman);

        atom = list.atoms[1];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "y", desc)
        XCTAssertEqual(atom.fontStyle, .defaultStyle)

        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sqrt{\\mathrm{x}}y", desc)
    }

    func testText() throws {
        let str = "\\text{x y}";
        let list = MTMathListBuilder.build(fromString: str)!
        let desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 3, desc)
        var atom = list.atoms[0];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "x", desc)
        XCTAssertEqual(atom.fontStyle, .roman);

        atom = list.atoms[1];
        XCTAssertEqual(atom.type, .ordinary, desc)
        XCTAssertEqual(atom.nucleus, " ", desc)

        atom = list.atoms[2];
        XCTAssertEqual(atom.type, .variable, desc)
        XCTAssertEqual(atom.nucleus, "y", desc)
        XCTAssertEqual(atom.fontStyle, .roman);


        // convert it back to latex
        let latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\mathrm{x\\  y}", desc)
    }

    func testLimits() throws {
        // Int with no limits (default)
        var str = "\\int";
        var list = MTMathListBuilder.build(fromString: str)!
        var desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        var op = list.atoms[0] as! MTLargeOperator
        XCTAssertEqual(op.type, .largeOperator, desc)
        XCTAssertFalse(op.limits);

        // convert it back to latex
        var latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\int ", desc)

        // Int with limits
        str = "\\int\\limits"
        list = MTMathListBuilder.build(fromString: str)!
        desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        op = list.atoms[0] as! MTLargeOperator
        XCTAssertEqual(op.type, .largeOperator, desc)
        XCTAssertTrue(op.limits)

        // convert it back to latex
        latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\int \\limits ", desc)
    }

    func testNoLimits() throws {
        // Sum with limits (default)
        var str = "\\sum";
        var list = MTMathListBuilder.build(fromString: str)!
        var desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual((list.atoms.count), 1, desc)
        var op = list.atoms[0] as! MTLargeOperator
        XCTAssertEqual(op.type, .largeOperator, desc)
        XCTAssertTrue(op.limits);

        // convert it back to latex
        var latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sum ", desc)

        // Int with limits
        str = "\\sum\\nolimits";
        list = MTMathListBuilder.build(fromString: str)!
        desc = "Error for string:\(str)"

        XCTAssertNotNil(list, desc)
        XCTAssertEqual(list.atoms.count, 1, desc)
        op = list.atoms[0] as! MTLargeOperator
        XCTAssertEqual(op.type, .largeOperator, desc)
        XCTAssertFalse(op.limits);

        // convert it back to latex
        latex = MTMathListBuilder.mathListToString(list)
        XCTAssertEqual(latex, "\\sum \\nolimits ", desc)
    }

//    func testPerformanceExample() throws {
//        // This is an example of a performance test case.
//        measure {
//            // Put the code you want to measure the time of here.
//        }
//    }

}

