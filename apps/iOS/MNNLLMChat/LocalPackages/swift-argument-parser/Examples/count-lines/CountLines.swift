//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import ArgumentParser
import Foundation

@main
@available(macOS 12, *)
struct CountLines: AsyncParsableCommand {
    @Argument(
        help: "A file to count lines in. If omitted, counts the lines of stdin.",
        completion: .file(), transform: URL.init(fileURLWithPath:))
    var inputFile: URL? = nil
    
    @Option(help: "Only count lines with this prefix.")
    var prefix: String? = nil
    
    @Flag(help: "Include extra information in the output.")
    var verbose = false
}

@available(macOS 12, *)
extension CountLines {
    var fileHandle: FileHandle {
        get throws {
            guard let inputFile else {
                return .standardInput
            }
            return try FileHandle(forReadingFrom: inputFile)
        }
    }
    
    func printCount(_ count: Int) {
        guard verbose else {
            print(count)
            return
        }
        
        if let filename = inputFile?.lastPathComponent {
            print("Lines in '\(filename)'", terminator: "")
        } else {
            print("Lines from stdin", terminator: "")
        }
        
        if let prefix {
            print(", prefixed by '\(prefix)'", terminator: "")
        }
        
        print(": \(count)")
    }
    
    mutating func run() async throws {
        let countAllLines = prefix == nil
        let lineCount = try await fileHandle.bytes.lines.reduce(0) { count, line in
            if countAllLines || line.starts(with: prefix!) {
                return count + 1
            } else {
                return count
            }
        }
        
        printCount(lineCount)
    }
}
