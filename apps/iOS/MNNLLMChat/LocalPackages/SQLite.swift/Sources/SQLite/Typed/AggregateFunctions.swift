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

private enum Function: String {
    case count
    case max
    case min
    case avg
    case sum
    case total

    func wrap<T>(_ expression: Expressible) -> Expression<T> {
        self.rawValue.wrap(expression)
    }
}

extension ExpressionType where UnderlyingType: Value {

    /// Builds a copy of the expression prefixed with the `DISTINCT` keyword.
    ///
    ///     let name = Expression<String>("name")
    ///     name.distinct
    ///     // DISTINCT "name"
    ///
    /// - Returns: A copy of the expression prefixed with the `DISTINCT`
    ///   keyword.
    public var distinct: Expression<UnderlyingType> {
        Expression("DISTINCT \(template)", bindings)
    }

    /// Builds a copy of the expression wrapped with the `count` aggregate
    /// function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.count
    ///     // count("name")
    ///     name.distinct.count
    ///     // count(DISTINCT "name")
    ///
    /// - Returns: A copy of the expression wrapped with the `count` aggregate
    ///   function.
    public var count: Expression<Int> {
        Function.count.wrap(self)
    }

}

extension ExpressionType where UnderlyingType: _OptionalType, UnderlyingType.WrappedType: Value {

    /// Builds a copy of the expression prefixed with the `DISTINCT` keyword.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.distinct
    ///     // DISTINCT "name"
    ///
    /// - Returns: A copy of the expression prefixed with the `DISTINCT`
    ///   keyword.
    public var distinct: Expression<UnderlyingType> {
        Expression("DISTINCT \(template)", bindings)
    }

    /// Builds a copy of the expression wrapped with the `count` aggregate
    /// function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.count
    ///     // count("name")
    ///     name.distinct.count
    ///     // count(DISTINCT "name")
    ///
    /// - Returns: A copy of the expression wrapped with the `count` aggregate
    ///   function.
    public var count: Expression<Int> {
        Function.count.wrap(self)
    }

}

extension ExpressionType where UnderlyingType: Value, UnderlyingType.Datatype: Comparable {

    /// Builds a copy of the expression wrapped with the `max` aggregate
    /// function.
    ///
    ///     let age = Expression<Int>("age")
    ///     age.max
    ///     // max("age")
    ///
    /// - Returns: A copy of the expression wrapped with the `max` aggregate
    ///   function.
    public var max: Expression<UnderlyingType?> {
        Function.max.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `min` aggregate
    /// function.
    ///
    ///     let age = Expression<Int>("age")
    ///     age.min
    ///     // min("age")
    ///
    /// - Returns: A copy of the expression wrapped with the `min` aggregate
    ///   function.
    public var min: Expression<UnderlyingType?> {
        Function.min.wrap(self)
    }

}

extension ExpressionType where UnderlyingType: _OptionalType, UnderlyingType.WrappedType: Value, UnderlyingType.WrappedType.Datatype: Comparable {

    /// Builds a copy of the expression wrapped with the `max` aggregate
    /// function.
    ///
    ///     let age = Expression<Int?>("age")
    ///     age.max
    ///     // max("age")
    ///
    /// - Returns: A copy of the expression wrapped with the `max` aggregate
    ///   function.
    public var max: Expression<UnderlyingType> {
        Function.max.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `min` aggregate
    /// function.
    ///
    ///     let age = Expression<Int?>("age")
    ///     age.min
    ///     // min("age")
    ///
    /// - Returns: A copy of the expression wrapped with the `min` aggregate
    ///   function.
    public var min: Expression<UnderlyingType> {
        Function.min.wrap(self)
    }

}

extension ExpressionType where UnderlyingType: Value, UnderlyingType.Datatype: Number {

    /// Builds a copy of the expression wrapped with the `avg` aggregate
    /// function.
    ///
    ///     let salary = Expression<Double>("salary")
    ///     salary.average
    ///     // avg("salary")
    ///
    /// - Returns: A copy of the expression wrapped with the `avg` aggregate
    ///   function.
    public var average: Expression<Double?> {
        Function.avg.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `sum` aggregate
    /// function.
    ///
    ///     let salary = Expression<Double>("salary")
    ///     salary.sum
    ///     // sum("salary")
    ///
    /// - Returns: A copy of the expression wrapped with the `sum` aggregate
    ///   function.
    public var sum: Expression<UnderlyingType?> {
        Function.sum.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `total` aggregate
    /// function.
    ///
    ///     let salary = Expression<Double>("salary")
    ///     salary.total
    ///     // total("salary")
    ///
    /// - Returns: A copy of the expression wrapped with the `total` aggregate
    ///   function.
    public var total: Expression<Double> {
        Function.total.wrap(self)
    }

}

extension ExpressionType where UnderlyingType: _OptionalType, UnderlyingType.WrappedType: Value, UnderlyingType.WrappedType.Datatype: Number {

    /// Builds a copy of the expression wrapped with the `avg` aggregate
    /// function.
    ///
    ///     let salary = Expression<Double?>("salary")
    ///     salary.average
    ///     // avg("salary")
    ///
    /// - Returns: A copy of the expression wrapped with the `min` aggregate
    ///   function.
    public var average: Expression<Double?> {
        Function.avg.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `sum` aggregate
    /// function.
    ///
    ///     let salary = Expression<Double?>("salary")
    ///     salary.sum
    ///     // sum("salary")
    ///
    /// - Returns: A copy of the expression wrapped with the `min` aggregate
    ///   function.
    public var sum: Expression<UnderlyingType> {
        Function.sum.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `total` aggregate
    /// function.
    ///
    ///     let salary = Expression<Double?>("salary")
    ///     salary.total
    ///     // total("salary")
    ///
    /// - Returns: A copy of the expression wrapped with the `min` aggregate
    ///   function.
    public var total: Expression<Double> {
        Function.total.wrap(self)
    }

}

extension ExpressionType where UnderlyingType == Int {

    static func count(_ star: Star) -> Expression<UnderlyingType> {
        Function.count.wrap(star(nil, nil))
    }

}

/// Builds an expression representing `count(*)` (when called with the `*`
/// function literal).
///
///     count(*)
///     // count(*)
///
/// - Returns: An expression returning `count(*)` (when called with the `*`
///   function literal).
public func count(_ star: Star) -> Expression<Int> {
    Expression.count(star)
}
