import Foundation
#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

extension Connection {
    private typealias Aggregate = @convention(block) (Int, Context, Int32, Argv) -> Void

    /// Creates or redefines a custom SQL aggregate.
    ///
    /// - Parameters:
    ///
    ///   - aggregate: The name of the aggregate to create or redefine.
    ///
    ///   - argumentCount: The number of arguments that the aggregate takes. If
    ///     `nil`, the aggregate may take any number of arguments.
    ///
    ///     Default: `nil`
    ///
    ///   - deterministic: Whether or not the aggregate is deterministic (_i.e._
    ///     the aggregate always returns the same result for a given input).
    ///
    ///     Default: `false`
    ///
    ///   - step: A block of code to run for each row of an aggregation group.
    ///     The block is called with an array of raw SQL values mapped to the
    ///     aggregate’s parameters, and an UnsafeMutablePointer to a state
    ///     variable.
    ///
    ///   - final: A block of code to run after each row of an aggregation group
    ///     is processed. The block is called with an UnsafeMutablePointer to a
    ///     state variable, and should return a raw SQL value (or nil).
    ///
    ///   - state: A block of code to run to produce a fresh state variable for
    ///     each aggregation group. The block should return an
    ///     UnsafeMutablePointer to the fresh state variable.
    public func createAggregation<T>(
            _ functionName: String,
            argumentCount: UInt? = nil,
            deterministic: Bool = false,
            step: @escaping ([Binding?], UnsafeMutablePointer<T>) -> Void,
            final: @escaping (UnsafeMutablePointer<T>) -> Binding?,
            state: @escaping () -> UnsafeMutablePointer<T>) {

        let argc = argumentCount.map { Int($0) } ?? -1
        let box: Aggregate = { (stepFlag: Int, context: Context, argc: Int32, argv: Argv) in
            let nBytes = Int32(MemoryLayout<UnsafeMutablePointer<Int64>>.size)
            guard let aggregateContext = sqlite3_aggregate_context(context, nBytes) else {
                fatalError("Could not get aggregate context")
            }
            let mutablePointer = aggregateContext.assumingMemoryBound(to: UnsafeMutableRawPointer.self)
            if stepFlag > 0 {
                let arguments = argv.getBindings(argc: argc)
                if aggregateContext.assumingMemoryBound(to: Int64.self).pointee == 0 {
                    mutablePointer.pointee = UnsafeMutableRawPointer(mutating: state())
                }
                step(arguments, mutablePointer.pointee.assumingMemoryBound(to: T.self))
            } else {
                let result = final(mutablePointer.pointee.assumingMemoryBound(to: T.self))
                context.set(result: result)
            }
        }

        func xStep(context: Context, argc: Int32, value: Argv) {
            unsafeBitCast(sqlite3_user_data(context), to: Aggregate.self)(1, context, argc, value)
        }

        func xFinal(context: Context) {
            unsafeBitCast(sqlite3_user_data(context), to: Aggregate.self)(0, context, 0, nil)
        }

        let flags = SQLITE_UTF8 | (deterministic ? SQLITE_DETERMINISTIC : 0)
        let resultCode = sqlite3_create_function_v2(
            handle,
            functionName,
            Int32(argc),
            flags,
            /* pApp */ unsafeBitCast(box, to: UnsafeMutableRawPointer.self),
            /* xFunc */ nil, xStep, xFinal, /* xDestroy */ nil
        )
        if let result = Result(errorCode: resultCode, connection: self) {
            fatalError("Error creating function: \(result)")
        }
        register(functionName, argc: argc, value: box)
    }

    public func createAggregation<T: AnyObject>(
            _ aggregate: String,
            argumentCount: UInt? = nil,
            deterministic: Bool = false,
            initialValue: T,
            reduce: @escaping (T, [Binding?]) -> T,
            result: @escaping (T) -> Binding?
    ) {
        let step: ([Binding?], UnsafeMutablePointer<UnsafeMutableRawPointer>) -> Void = { (bindings, ptr) in
            let pointer = ptr.pointee.assumingMemoryBound(to: T.self)
            let current = Unmanaged<T>.fromOpaque(pointer).takeRetainedValue()
            let next = reduce(current, bindings)
            ptr.pointee = Unmanaged.passRetained(next).toOpaque()
        }

        let final: (UnsafeMutablePointer<UnsafeMutableRawPointer>) -> Binding? = { ptr in
            let pointer = ptr.pointee.assumingMemoryBound(to: T.self)
            let obj = Unmanaged<T>.fromOpaque(pointer).takeRetainedValue()
            let value = result(obj)
            ptr.deallocate()
            return value
        }

        let state: () -> UnsafeMutablePointer<UnsafeMutableRawPointer> = {
            let pointer = UnsafeMutablePointer<UnsafeMutableRawPointer>.allocate(capacity: 1)
            pointer.pointee = Unmanaged.passRetained(initialValue).toOpaque()
            return pointer
        }

        createAggregation(aggregate, step: step, final: final, state: state)
    }

    public func createAggregation<T>(
            _ aggregate: String,
            argumentCount: UInt? = nil,
            deterministic: Bool = false,
            initialValue: T,
            reduce: @escaping (T, [Binding?]) -> T,
            result: @escaping (T) -> Binding?
    ) {

        let step: ([Binding?], UnsafeMutablePointer<T>) -> Void = { (bindings, pointer) in
            let current = pointer.pointee
            let next = reduce(current, bindings)
            pointer.pointee = next
        }

        let final: (UnsafeMutablePointer<T>) -> Binding? = { pointer in
            let value = result(pointer.pointee)
            pointer.deallocate()
            return value
        }

        let state: () -> UnsafeMutablePointer<T> = {
            let pointer = UnsafeMutablePointer<T>.allocate(capacity: 1)
            pointer.initialize(to: initialValue)
            return pointer
        }

        createAggregation(aggregate, step: step, final: final, state: state)
    }

}
