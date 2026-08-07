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

public extension Connection {

    /// Creates or redefines a custom SQL function.
    ///
    /// - Parameters:
    ///
    ///   - function: The name of the function to create or redefine.
    ///
    ///   - deterministic: Whether or not the function is deterministic (_i.e._
    ///     the function always returns the same result for a given input).
    ///
    ///     Default: `false`
    ///
    ///   - block: A block of code to run when the function is called.
    ///     The assigned types must be explicit.
    ///
    /// - Returns: A closure returning an SQL expression to call the function.
    func createFunction<Z: Value>(_ function: String, deterministic: Bool = false, _ block: @escaping () -> Z) throws
                    -> () -> Expression<Z> {
        let fn = try createFunction(function, 0, deterministic) { _ in block() }
        return { fn([]) }
    }

    func createFunction<Z: Value>(_ function: String, deterministic: Bool = false, _ block: @escaping () -> Z?) throws
                    -> () -> Expression<Z?> {
        let fn = try createFunction(function, 0, deterministic) { _ in block() }
        return { fn([]) }
    }

    // MARK: -

    func createFunction<Z: Value, A: Value>(_ function: String, deterministic: Bool = false, _ block: @escaping (A) -> Z) throws
                    -> (Expression<A>) -> Expression<Z> {
        let fn = try createFunction(function, 1, deterministic) { args in block(value(args[0])) }
        return { arg in fn([arg]) }
    }

    func createFunction<Z: Value, A: Value>(_ function: String, deterministic: Bool = false, _ block: @escaping (A?) -> Z) throws
                    -> (Expression<A?>) -> Expression<Z> {
        let fn = try createFunction(function, 1, deterministic) { args in block(args[0].map(value)) }
        return { arg in fn([arg]) }
    }

    func createFunction<Z: Value, A: Value>(_ function: String, deterministic: Bool = false, _ block: @escaping (A) -> Z?) throws
                    -> (Expression<A>) -> Expression<Z?> {
        let fn = try createFunction(function, 1, deterministic) { args in block(value(args[0])) }
        return { arg in fn([arg]) }
    }

    func createFunction<Z: Value, A: Value>(_ function: String, deterministic: Bool = false, _ block: @escaping (A?) -> Z?) throws
                    -> (Expression<A?>) -> Expression<Z?> {
        let fn = try createFunction(function, 1, deterministic) { args in block(args[0].map(value)) }
        return { arg in fn([arg]) }
    }

    // MARK: -

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A, B) -> Z) throws -> (Expression<A>, Expression<B>)
    -> Expression<Z> {
        let fn = try createFunction(function, 2, deterministic) { args in block(value(args[0]), value(args[1])) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A?, B) -> Z) throws
                    -> (Expression<A?>, Expression<B>) -> Expression<Z> {
        let fn = try createFunction(function, 2, deterministic) { args in block(args[0].map(value), value(args[1])) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A, B?) -> Z) throws ->
            (Expression<A>, Expression<B?>) -> Expression<Z> {
        let fn = try createFunction(function, 2, deterministic) { args in block(value(args[0]), args[1].map(value)) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A, B) -> Z?) throws
                    -> (Expression<A>, Expression<B>) -> Expression<Z?> {
        let fn = try createFunction(function, 2, deterministic) { args in block(value(args[0]), value(args[1])) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A?, B?) -> Z) throws
                    -> (Expression<A?>, Expression<B?>) -> Expression<Z> {
        let fn = try createFunction(function, 2, deterministic) { args in block(args[0].map(value), args[1].map(value)) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A?, B) -> Z?) throws
                    -> (Expression<A?>, Expression<B>) -> Expression<Z?> {
        let fn = try createFunction(function, 2, deterministic) { args in block(args[0].map(value), value(args[1])) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A, B?) -> Z?) throws
                    -> (Expression<A>, Expression<B?>) -> Expression<Z?> {
        let fn = try createFunction(function, 2, deterministic) { args in block(value(args[0]), args[1].map(value)) }
        return { a, b in fn([a, b]) }
    }

    func createFunction<Z: Value, A: Value, B: Value>(_ function: String, deterministic: Bool = false,
                                                      _ block: @escaping (A?, B?) -> Z?) throws
                    -> (Expression<A?>, Expression<B?>) -> Expression<Z?> {
        let fn = try createFunction(function, 2, deterministic) { args in block(args[0].map(value), args[1].map(value)) }
        return { a, b in fn([a, b]) }
    }

    // MARK: -

    fileprivate func createFunction<Z: Value>(_ function: String, _ argumentCount: UInt, _ deterministic: Bool,
                                              _ block: @escaping ([Binding?]) -> Z) throws
                    -> ([Expressible]) -> Expression<Z> {
        createFunction(function, argumentCount: argumentCount, deterministic: deterministic) { arguments in
            block(arguments).datatypeValue
        }
        return { arguments in
            function.quote().wrap(", ".join(arguments))
        }
    }

    fileprivate func createFunction<Z: Value>(_ function: String, _ argumentCount: UInt, _ deterministic: Bool,
                                              _ block: @escaping ([Binding?]) -> Z?) throws
                    -> ([Expressible]) -> Expression<Z?> {
        createFunction(function, argumentCount: argumentCount, deterministic: deterministic) { arguments in
            block(arguments)?.datatypeValue
        }
        return { arguments in
            function.quote().wrap(", ".join(arguments))
        }
    }

}
