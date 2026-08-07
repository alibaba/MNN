import XCTest
import SQLite

class AggregateFunctionsTests: XCTestCase {

    func test_distinct_prependsExpressionsWithDistinctKeyword() {
        assertSQL("DISTINCT \"int\"", int.distinct)
        assertSQL("DISTINCT \"intOptional\"", intOptional.distinct)
        assertSQL("DISTINCT \"double\"", double.distinct)
        assertSQL("DISTINCT \"doubleOptional\"", doubleOptional.distinct)
        assertSQL("DISTINCT \"string\"", string.distinct)
        assertSQL("DISTINCT \"stringOptional\"", stringOptional.distinct)
    }

    func test_count_wrapsOptionalExpressionsWithCountFunction() {
        assertSQL("count(\"intOptional\")", intOptional.count)
        assertSQL("count(\"doubleOptional\")", doubleOptional.count)
        assertSQL("count(\"stringOptional\")", stringOptional.count)
    }

    func test_max_wrapsComparableExpressionsWithMaxFunction() {
        assertSQL("max(\"int\")", int.max)
        assertSQL("max(\"intOptional\")", intOptional.max)
        assertSQL("max(\"double\")", double.max)
        assertSQL("max(\"doubleOptional\")", doubleOptional.max)
        assertSQL("max(\"string\")", string.max)
        assertSQL("max(\"stringOptional\")", stringOptional.max)
        assertSQL("max(\"date\")", date.max)
        assertSQL("max(\"dateOptional\")", dateOptional.max)
    }

    func test_min_wrapsComparableExpressionsWithMinFunction() {
        assertSQL("min(\"int\")", int.min)
        assertSQL("min(\"intOptional\")", intOptional.min)
        assertSQL("min(\"double\")", double.min)
        assertSQL("min(\"doubleOptional\")", doubleOptional.min)
        assertSQL("min(\"string\")", string.min)
        assertSQL("min(\"stringOptional\")", stringOptional.min)
        assertSQL("min(\"date\")", date.min)
        assertSQL("min(\"dateOptional\")", dateOptional.min)
    }

    func test_average_wrapsNumericExpressionsWithAvgFunction() {
        assertSQL("avg(\"int\")", int.average)
        assertSQL("avg(\"intOptional\")", intOptional.average)
        assertSQL("avg(\"double\")", double.average)
        assertSQL("avg(\"doubleOptional\")", doubleOptional.average)
    }

    func test_sum_wrapsNumericExpressionsWithSumFunction() {
        assertSQL("sum(\"int\")", int.sum)
        assertSQL("sum(\"intOptional\")", intOptional.sum)
        assertSQL("sum(\"double\")", double.sum)
        assertSQL("sum(\"doubleOptional\")", doubleOptional.sum)
    }

    func test_total_wrapsNumericExpressionsWithTotalFunction() {
        assertSQL("total(\"int\")", int.total)
        assertSQL("total(\"intOptional\")", intOptional.total)
        assertSQL("total(\"double\")", double.total)
        assertSQL("total(\"doubleOptional\")", doubleOptional.total)
    }

    func test_count_withStar_wrapsStarWithCountFunction() {
        assertSQL("count(*)", count(*))
    }

}
