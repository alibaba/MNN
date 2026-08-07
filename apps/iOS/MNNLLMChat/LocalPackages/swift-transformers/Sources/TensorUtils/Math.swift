//
//  Math.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Accelerate
import CoreML
import Foundation

///
/// From M.I. Hollemans
///
/// https://github.com/hollance/CoreMLHelpers
///
public struct Math {
    /**
     Returns the index and value of the largest element in the array.

     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    public static func argmax(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }

    /**
     Returns the index and value of the largest element in the array.
     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    public static func argmax(_ ptr: UnsafePointer<Double>, count: Int, stride: Int = 1) -> (Int, Double) {
        var maxValue: Double = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxviD(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }

    public static func argmax32(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }

    /// MLMultiArray helper.
    /// Works in our specific use case.
    public static func argmax(_ multiArray: MLMultiArray) -> (Int, Double) {
        assert(multiArray.dataType == .double)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(multiArray.dataPointer))
        return Math.argmax(ptr, count: multiArray.count)
    }

    /// MLMultiArray helper.
    /// Works in our specific use case.
    public static func argmax32(_ multiArray: MLMultiArray) -> (Int, Float) {
        assert(multiArray.dataType == .float32)
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        return Math.argmax32(ptr, count: multiArray.count)
    }

    /// Returns the cumulative sum of the array.
    public static func cumsum(_ arr: [Float]) -> [Float] {
        guard !arr.isEmpty else {
            return []
        }
        let arrCount = vDSP_Length(arr.count)
        var weight: Float = 1.0
        var result: [Float] = Array(repeating: 0.0, count: arr.count)
        var firstItem = arr[0]
        vDSP_vrsum(arr, 1, &weight, &result, 1, arrCount)
        vDSP_vsadd(result, 1, &firstItem, &result, 1, arrCount)
        return result
    }

    /// Multinomial sampling from an array of probs. Works well with topK
    public static func sample(indexes: [Int], probs: [Float]) -> Int {
        let i = randomNumber(probabilities: probs)
        return indexes[i]
    }

    /**
     Computes the "softmax" function over an array.
     Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/
     This is what softmax looks like in "pseudocode" (actually using Python
     and numpy):
     x -= np.max(x)
     exp_scores = np.exp(x)
     softmax = exp_scores / np.sum(exp_scores)
     First we shift the values of x so that the highest value in the array is 0.
     This ensures numerical stability with the exponents, so they don't blow up.
     */
    public static func softmax(_ x: [Float]) -> [Float] {
        var x = x
        let len = vDSP_Length(x.count)

        // Find the maximum value in the input array.
        var max: Float = 0
        vDSP_maxv(x, 1, &max, len)

        // Subtract the maximum from all the elements in the array.
        // Now the highest value in the array is 0.
        max = -max
        vDSP_vsadd(x, 1, &max, &x, 1, len)

        // Exponentiate all the elements in the array.
        var count = Int32(x.count)
        vvexpf(&x, x, &count)

        // Compute the sum of all exponentiated values.
        var sum: Float = 0
        vDSP_sve(x, 1, &sum, len)

        // Divide each element by the sum. This normalizes the array contents
        // so that they all add up to 1.
        vDSP_vsdiv(x, 1, &sum, &x, 1, len)

        return x
    }

    /// Multinomial sampling
    ///
    /// From https://stackoverflow.com/questions/30309556/generate-random-numbers-with-a-given-distribution
    ///
    public static func randomNumber(probabilities: [Float]) -> Int {
        // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
        let sum = probabilities.reduce(0, +)
        // Random number in the range 0.0 <= rnd < sum :
        let rnd = sum * Float(arc4random_uniform(UInt32.max)) / Float(UInt32.max)
        // Find the first interval of accumulated probabilities into which `rnd` falls:
        var accum: Float = 0.0
        for (i, p) in probabilities.enumerated() {
            accum += p
            if rnd < accum {
                return i
            }
        }
        // This point might be reached due to floating point inaccuracies:
        return probabilities.count - 1
    }
}

// MLShapedArray versions

public extension Math {
    static func argmax(_ shapedArray: MLShapedArray<Float>) -> (Int, Float) {
        shapedArray.withUnsafeShapedBufferPointer { ptr, shape, strides in
            assert(shape.count == 1, "Only supported for 1-dimensional arrays or slices")
            return Math.argmax32(ptr.baseAddress!, count: shapedArray.count, stride: strides.first!)
        }
    }

    // TODO: handle Double, etc.
    static func argmax(_ shapedArray: some MLShapedArrayProtocol) -> (Int, Float) {
        shapedArray.withUnsafeShapedBufferPointer { ptr, shape, strides in
            assert(shape.count == 1, "Only supported for 1-dimensional arrays or slices")
            let floatsPtr = ptr.baseAddress as! UnsafePointer<Float>
            return Math.argmax32(floatsPtr, count: shapedArray.count, stride: strides.first!)
        }
    }
}
