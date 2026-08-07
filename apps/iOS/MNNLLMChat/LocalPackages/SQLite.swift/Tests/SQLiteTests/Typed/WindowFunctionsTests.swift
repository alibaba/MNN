import XCTest
import SQLite

class WindowFunctionsTests: XCTestCase {

    func test_ntile_wrapsExpressionWithOverClause() {
        assertSQL("ntile(1) OVER (ORDER BY \"int\" DESC)", ntile(1, int.desc))
        assertSQL("ntile(20) OVER (ORDER BY \"intOptional\" ASC)", ntile(20, intOptional.asc))
        assertSQL("ntile(20) OVER (ORDER BY \"double\" ASC)", ntile(20, double.asc))
        assertSQL("ntile(1) OVER (ORDER BY \"doubleOptional\" ASC)", ntile(1, doubleOptional.asc))
        assertSQL("ntile(1) OVER (ORDER BY \"int\" DESC)", ntile(1, int.desc))
    }

    func test_row_number_wrapsExpressionWithOverClause() {
        assertSQL("row_number() OVER (ORDER BY \"int\" DESC)", rowNumber(int.desc))
    }

    func test_rank_wrapsExpressionWithOverClause() {
        assertSQL("rank() OVER (ORDER BY \"int\" DESC)", rank(int.desc))
    }

    func test_dense_rank_wrapsExpressionWithOverClause() {
        assertSQL("dense_rank() OVER (ORDER BY \"int\" DESC)", denseRank(int.desc))
    }

    func test_percent_rank_wrapsExpressionWithOverClause() {
        assertSQL("percent_rank() OVER (ORDER BY \"int\" DESC)", percentRank(int.desc))
    }

    func test_cume_dist_wrapsExpressionWithOverClause() {
        assertSQL("cume_dist() OVER (ORDER BY \"int\" DESC)", cumeDist(int.desc))
    }

    func test_lag_wrapsExpressionWithOverClause() {
        assertSQL("lag(\"int\", 0) OVER (ORDER BY \"int\" DESC)", int.lag(int.desc))
        assertSQL("lag(\"int\", 7) OVER (ORDER BY \"int\" DESC)", int.lag(offset: 7, int.desc))
        assertSQL("lag(\"int\", 1, 3) OVER (ORDER BY \"int\" DESC)", int.lag(offset: 1, default: Expression<Int>(value: 3), int.desc))
    }

    func test_lead_wrapsExpressionWithOverClause() {
        assertSQL("lead(\"int\", 0) OVER (ORDER BY \"int\" DESC)", int.lead(int.desc))
        assertSQL("lead(\"int\", 7) OVER (ORDER BY \"int\" DESC)", int.lead(offset: 7, int.desc))
        assertSQL("lead(\"int\", 1, 3) OVER (ORDER BY \"int\" DESC)", int.lead(offset: 1, default: Expression<Int>(value: 3), int.desc))
    }

    func test_firstValue_wrapsExpressionWithOverClause() {
        assertSQL("first_value(\"int\") OVER (ORDER BY \"int\" DESC)", int.firstValue(int.desc))
        assertSQL("first_value(\"double\") OVER (ORDER BY \"int\" DESC)", double.firstValue(int.desc))
    }

    func test_lastValue_wrapsExpressionWithOverClause() {
        assertSQL("last_value(\"int\") OVER (ORDER BY \"int\" DESC)", int.lastValue(int.desc))
    }

    func test_nth_value_wrapsExpressionWithOverClause() {
        assertSQL("nth_value(\"int\", 3) OVER (ORDER BY \"int\" DESC)", int.value(3, int.desc))
    }
}
