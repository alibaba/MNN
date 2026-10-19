import Foundation

// see https://www.sqlite.org/windowfunctions.html#builtins
private enum WindowFunction: String {
    // swiftlint:disable identifier_name
    case ntile
    case row_number
    case rank
    case dense_rank
    case percent_rank
    case cume_dist
    case lag
    case lead
    case first_value
    case last_value
    case nth_value
    // swiftlint:enable identifier_name

    func wrap<T>(_ value: Int? = nil) -> Expression<T> {
        if let value {
            return self.rawValue.wrap(Expression(value: value))
        }
        return Expression(literal: "\(rawValue)()")
    }

    func over<T>(value: Int? = nil, _ orderBy: Expressible) -> Expression<T> {
        return Expression<T>(" ".join([
            self.wrap(value),
            Expression<T>("OVER (ORDER BY \(orderBy.expression.template))", orderBy.expression.bindings)
        ]).expression)
    }

    func over<T>(valueExpr: Expressible, _ orderBy: Expressible) -> Expression<T> {
        return Expression<T>(" ".join([
            self.rawValue.wrap(valueExpr),
            Expression<T>("OVER (ORDER BY \(orderBy.expression.template))", orderBy.expression.bindings)
        ]).expression)
    }
}

extension ExpressionType where UnderlyingType: Value {
    /// Builds a copy of the expression with `lag(self, offset, default) OVER (ORDER BY {orderBy})` window function
    ///
    /// - Parameter orderBy: Expression to evaluate window order
    /// - Returns: An expression returning `lag(self, offset, default) OVER (ORDER BY {orderBy})` window function
    public func lag(offset: Int = 0, default: Expressible? = nil, _ orderBy: Expressible) -> Expression<UnderlyingType> {
        if let defaultExpression = `default` {
            return Expression(
                "lag(\(template), \(offset), \(defaultExpression.asSQL())) OVER (ORDER BY \(orderBy.expression.template))",
                bindings + orderBy.expression.bindings
            )

        }
        return Expression("lag(\(template), \(offset)) OVER (ORDER BY \(orderBy.expression.template))", bindings + orderBy.expression.bindings)
    }

    /// Builds a copy of the expression with `lead(self, offset, default) OVER (ORDER BY {orderBy})` window function
    ///
    /// - Parameter orderBy: Expression to evaluate window order
    /// - Returns: An expression returning `lead(self, offset, default) OVER (ORDER BY {orderBy})` window function
    public func lead(offset: Int = 0, default: Expressible? = nil, _ orderBy: Expressible) -> Expression<UnderlyingType> {
        if let defaultExpression = `default` {
            return Expression(
                "lead(\(template), \(offset), \(defaultExpression.asSQL())) OVER (ORDER BY \(orderBy.expression.template))",
                bindings + orderBy.expression.bindings)

        }
        return Expression("lead(\(template), \(offset)) OVER (ORDER BY \(orderBy.expression.template))", bindings + orderBy.expression.bindings)
    }

    /// Builds a copy of the expression with `first_value(self) OVER (ORDER BY {orderBy})` window function
    ///
    /// - Parameter orderBy: Expression to evaluate window order
    /// - Returns: An expression returning `first_value(self) OVER (ORDER BY {orderBy})` window function
    public func firstValue(_ orderBy: Expressible) -> Expression<UnderlyingType> {
        WindowFunction.first_value.over(valueExpr: self, orderBy)
    }

    /// Builds a copy of the expression with `last_value(self) OVER (ORDER BY {orderBy})` window function
    ///
    /// - Parameter orderBy: Expression to evaluate window order
    /// - Returns: An expression returning `last_value(self) OVER (ORDER BY {orderBy})` window function
    public func lastValue(_ orderBy: Expressible) -> Expression<UnderlyingType> {
        WindowFunction.last_value.over(valueExpr: self, orderBy)
    }

    /// Builds a copy of the expression with `nth_value(self) OVER (ORDER BY {orderBy})` window function
    ///
    /// - Parameter index: Row N of the window frame to return
    /// - Parameter orderBy: Expression to evaluate window order
    /// - Returns: An expression returning `nth_value(self) OVER (ORDER BY {orderBy})` window function
    public func value(_ index: Int, _ orderBy: Expressible) -> Expression<UnderlyingType> {
        Expression("nth_value(\(template), \(index)) OVER (ORDER BY \(orderBy.expression.template))", bindings + orderBy.expression.bindings)
    }
}

/// Builds an expression representing `ntile(size) OVER (ORDER BY {orderBy})`
///
/// - Parameter orderBy: Expression to evaluate window order
/// - Returns: An expression returning  `ntile(size) OVER (ORDER BY {orderBy})`
public func ntile(_ size: Int, _ orderBy: Expressible) -> Expression<Int> {
//    Expression.ntile(size, orderBy)

        WindowFunction.ntile.over(value: size, orderBy)
}

/// Builds an expression representing `row_count() OVER (ORDER BY {orderBy})`
///
/// - Parameter orderBy: Expression to evaluate window order
/// - Returns: An expression returning `row_count() OVER (ORDER BY {orderBy})`
public func rowNumber(_ orderBy: Expressible) -> Expression<Int> {
    WindowFunction.row_number.over(orderBy)
}

/// Builds an expression representing `rank() OVER (ORDER BY {orderBy})`
///
/// - Parameter orderBy: Expression to evaluate window order
/// - Returns: An expression returning `rank() OVER (ORDER BY {orderBy})`
public func rank(_ orderBy: Expressible) -> Expression<Int> {
    WindowFunction.rank.over(orderBy)
}

/// Builds an expression representing `dense_rank() OVER (ORDER BY {orderBy})`
///
/// - Parameter orderBy: Expression to evaluate window order
/// - Returns: An expression returning `dense_rank() OVER ('over')`
public func denseRank(_ orderBy: Expressible) -> Expression<Int> {
    WindowFunction.dense_rank.over(orderBy)
}

/// Builds an expression representing `percent_rank() OVER (ORDER BY {orderBy})`
///
/// - Parameter orderBy: Expression to evaluate window order
/// - Returns: An expression returning `percent_rank() OVER (ORDER BY {orderBy})`
public func percentRank(_ orderBy: Expressible) -> Expression<Double> {
    WindowFunction.percent_rank.over(orderBy)
}

/// Builds an expression representing `cume_dist() OVER (ORDER BY {orderBy})`
///
/// - Parameter orderBy: Expression to evaluate window order
/// - Returns: An expression returning `cume_dist() OVER (ORDER BY {orderBy})`
public func cumeDist(_ orderBy: Expressible) -> Expression<Double> {
    WindowFunction.cume_dist.over(orderBy)
}
