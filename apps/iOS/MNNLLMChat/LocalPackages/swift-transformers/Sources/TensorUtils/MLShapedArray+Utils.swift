//
//  MLShapedArray+Utils.swift
//
//
//  Created by Pedro Cuenca on 13/5/23.
//

import CoreML

public extension MLShapedArray<Float> {
    var floats: [Float] {
        guard strides.first == 1, strides.count == 1 else {
            // For some reason this path is slow.
            // If strides is not 1, we can write a Metal kernel to copy the values properly.
            return scalars
        }

        // Fast path: memcpy
        let mlArray = MLMultiArray(self)
        return mlArray.floats ?? scalars
    }
}

public extension MLShapedArraySlice<Float> {
    var floats: [Float] {
        guard strides.first == 1, strides.count == 1 else {
            // For some reason this path is slow.
            // If strides is not 1, we can write a Metal kernel to copy the values properly.
            return scalars
        }

        // Fast path: memcpy
        let mlArray = MLMultiArray(self)
        return mlArray.floats ?? scalars
    }
}

public extension MLMultiArray {
    var floats: [Float]? {
        guard dataType == .float32 else { return nil }

        var result: [Float] = Array(repeating: 0, count: count)
        return withUnsafeBytes { ptr in
            guard let source = ptr.baseAddress else { return nil }
            result.withUnsafeMutableBytes { resultPtr in
                let dest = resultPtr.baseAddress!
                memcpy(dest, source, self.count * MemoryLayout<Float>.stride)
            }
            return result
        }
    }
}
