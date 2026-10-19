//
//  MLMultiArray+Utils.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import CoreML
import Foundation

public extension MLMultiArray {
    /// All values will be stored in the last dimension of the MLMultiArray (default is dims=1)
    static func from(_ arr: [Int], dims: Int = 1) -> MLMultiArray {
        var shape = Array(repeating: 1, count: dims)
        shape[shape.count - 1] = arr.count
        // Examples:
        // dims=1 : [arr.count]
        // dims=2 : [1, arr.count]
        //
        let o = try! MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(o.dataPointer))
        for (i, item) in arr.enumerated() {
            ptr[i] = Int32(item)
        }
        return o
    }

    /// All values will be stored in the last dimension of the MLMultiArray (default is dims=1)
    static func from(_ arr: [Double], dims: Int = 1) -> MLMultiArray {
        var shape = Array(repeating: 1, count: dims)
        shape[shape.count - 1] = arr.count
        // Examples:
        // dims=1 : [arr.count]
        // dims=2 : [1, arr.count]
        //
        let o = try! MLMultiArray(shape: shape as [NSNumber], dataType: .float64)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(o.dataPointer))
        for (i, item) in arr.enumerated() {
            ptr[i] = Double(item)
        }
        return o
    }

    /// This will concatenate all dimensions into one one-dim array.
    static func toIntArray(_ o: MLMultiArray) -> [Int] {
        var arr = Array(repeating: 0, count: o.count)
        let ptr = UnsafeMutablePointer<Int32>(OpaquePointer(o.dataPointer))
        for i in 0..<o.count {
            arr[i] = Int(ptr[i])
        }
        return arr
    }

    func toIntArray() -> [Int] { Self.toIntArray(self) }

    /// This will concatenate all dimensions into one one-dim array.
    static func toDoubleArray(_ o: MLMultiArray) -> [Double] {
        var arr: [Double] = Array(repeating: 0, count: o.count)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(o.dataPointer))
        for i in 0..<o.count {
            arr[i] = Double(ptr[i])
        }
        return arr
    }

    func toDoubleArray() -> [Double] { Self.toDoubleArray(self) }

    /// Helper to construct a sequentially-indexed multi array,
    /// useful for debugging and unit tests
    /// Example in 3 dimensions:
    /// ```
    /// [[[ 0, 1, 2, 3 ],
    ///   [ 4, 5, 6, 7 ],
    ///   [ 8, 9, 10, 11 ]],
    ///  [[ 12, 13, 14, 15 ],
    ///   [ 16, 17, 18, 19 ],
    ///   [ 20, 21, 22, 23 ]]]
    /// ```
    static func testTensor(shape: [Int]) -> MLMultiArray {
        let arr = try! MLMultiArray(shape: shape as [NSNumber], dataType: .double)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
        for i in 0..<arr.count {
            ptr.advanced(by: i).pointee = Double(i)
        }
        return arr
    }
}

public extension MLMultiArray {
    /// Provides a way to index n-dimensionals arrays a la numpy.
    enum Indexing: Equatable {
        case select(Int)
        case slice
    }

    /// Slice an array according to a list of `Indexing` enums.
    ///
    /// You must specify all dimensions.
    /// Note: only one slice is supported at the moment.
    static func slice(_ o: MLMultiArray, indexing: [Indexing]) -> MLMultiArray {
        assert(
            indexing.count == o.shape.count
        )
        assert(
            indexing.filter { $0 == Indexing.slice }.count == 1
        )
        var selectDims: [Int: Int] = [:]
        for (i, idx) in indexing.enumerated() {
            if case let .select(select) = idx {
                selectDims[i] = select
            }
        }
        return slice(
            o,
            sliceDim: indexing.firstIndex { $0 == Indexing.slice }!,
            selectDims: selectDims
        )
    }

    /// Slice an array according to a list, according to `sliceDim` (which dimension to slice on)
    /// and a dictionary of `dim` to `index`.
    ///
    /// You must select all other dimensions than the slice dimension (cf. the assert).
    static func slice(_ o: MLMultiArray, sliceDim: Int, selectDims: [Int: Int]) -> MLMultiArray {
        assert(
            selectDims.count + 1 == o.shape.count
        )
        var shape: [NSNumber] = Array(repeating: 1, count: o.shape.count)
        shape[sliceDim] = o.shape[sliceDim]
        // print("About to slice ndarray of shape \(o.shape) into ndarray of shape \(shape)")
        let arr = try! MLMultiArray(shape: shape, dataType: .double)

        // let srcPtr = UnsafeMutablePointer<Double>(OpaquePointer(o.dataPointer))
        // TODO: use srcPtr instead of array subscripting.
        let dstPtr = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
        for i in 0..<arr.count {
            var index: [Int] = []
            for j in 0..<shape.count {
                if j == sliceDim {
                    index.append(i)
                } else {
                    index.append(selectDims[j]!)
                }
            }
            // print("Accessing element \(index)")
            dstPtr[i] = o[index as [NSNumber]] as! Double
        }
        return arr
    }
}

extension MLMultiArray {
    var debug: String {
        debug([])
    }

    /// From https://twitter.com/mhollemans
    ///
    /// Slightly tweaked
    ///
    func debug(_ indices: [Int]) -> String {
        func indent(_ x: Int) -> String {
            String(repeating: " ", count: x)
        }

        // This function is called recursively for every dimension.
        // Add an entry for this dimension to the end of the array.
        var indices = indices + [0]

        let d = indices.count - 1 // the current dimension
        let N = shape[d].intValue // how many elements in this dimension
        var s = "["
        if indices.count < shape.count { // not last dimension yet?
            for i in 0..<N {
                indices[d] = i
                s += debug(indices) // then call recursively again
                if i != N - 1 {
                    s += ",\n" + indent(d + 1)
                }
            }
        } else { // the last dimension has actual data
            s += " "
            for i in 0..<N {
                indices[d] = i
                s += "\(self[indices as [NSNumber]])"
                if i != N - 1 { // not last element?
                    s += ", "
                    if i % 11 == 10 { // wrap long lines
                        s += "\n " + indent(d + 1)
                    }
                }
            }
            s += " "
        }
        return s + "]"
    }
}
