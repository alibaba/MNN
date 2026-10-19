//
// SQLite.swift
// https://github.com/stephencelis/SQLite.swift
// Copyright © 2014-2015 Stephen Celis.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

public struct Blob {

    public let bytes: [UInt8]

    public init(bytes: [UInt8]) {
        self.bytes = bytes
    }

    public init(bytes: UnsafeRawPointer, length: Int) {
        let i8bufptr = UnsafeBufferPointer(start: bytes.assumingMemoryBound(to: UInt8.self), count: length)
        self.init(bytes: [UInt8](i8bufptr))
    }

    public func toHex() -> String {
        bytes.map {
           ($0 < 16 ? "0" : "") + String($0, radix: 16, uppercase: false)
        }.joined(separator: "")
    }
}

extension Blob: CustomStringConvertible {
    public var description: String {
        "x'\(toHex())'"
    }
}

extension Blob: Equatable {
}

public func ==(lhs: Blob, rhs: Blob) -> Bool {
    lhs.bytes == rhs.bytes
}
