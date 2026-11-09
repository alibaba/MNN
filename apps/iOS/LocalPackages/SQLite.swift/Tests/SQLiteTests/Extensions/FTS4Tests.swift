import XCTest
import SQLite

class FTS4Tests: XCTestCase {

    func test_create_onVirtualTable_withFTS4_compilesCreateVirtualTableExpression() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4()",
            virtualTable.create(.FTS4())
        )
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(\"string\")",
            virtualTable.create(.FTS4(string))
        )
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=simple)",
            virtualTable.create(.FTS4(tokenize: .Simple))
        )
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(\"string\", tokenize=porter)",
            virtualTable.create(.FTS4([string], tokenize: .Porter))
        )
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=unicode61 \"remove_diacritics=0\")",
            virtualTable.create(.FTS4(tokenize: .Unicode61(removeDiacritics: false)))
        )
        XCTAssertEqual(
            """
            CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=unicode61 \"remove_diacritics=1\"
             \"tokenchars=.\" \"separators=X\")
            """.replacingOccurrences(of: "\n", with: ""),
            virtualTable.create(.FTS4(tokenize: .Unicode61(removeDiacritics: true,
                                                           tokenchars: ["."], separators: ["X"])))
        )
    }

    func test_match_onVirtualTableAsExpression_compilesMatchExpression() {
        assertSQL("(\"virtual_table\" MATCH 'string')", virtualTable.match("string") as Expression<Bool>)
        assertSQL("(\"virtual_table\" MATCH \"string\")", virtualTable.match(string) as Expression<Bool>)
        assertSQL("(\"virtual_table\" MATCH \"stringOptional\")", virtualTable.match(stringOptional) as Expression<Bool?>)
    }

    func test_match_onVirtualTableAsQueryType_compilesMatchExpression() {
        assertSQL("SELECT * FROM \"virtual_table\" WHERE (\"virtual_table\" MATCH 'string')",
                  virtualTable.match("string") as QueryType)
        assertSQL("SELECT * FROM \"virtual_table\" WHERE (\"virtual_table\" MATCH \"string\")",
                  virtualTable.match(string) as QueryType)
        assertSQL("SELECT * FROM \"virtual_table\" WHERE (\"virtual_table\" MATCH \"stringOptional\")",
                  virtualTable.match(stringOptional) as QueryType)
    }

}

class FTS4ConfigTests: XCTestCase {
    var config: FTS4Config!

    override func setUp() {
        super.setUp()
        config = FTS4Config()
    }

    func test_empty_config() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4()",
            sql(config))
    }

    func test_config_column() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(\"string\")",
            sql(config.column(string)))
    }

    func test_config_columns() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(\"string\", \"int\")",
            sql(config.columns([string, int])))
    }

    func test_config_unindexed_column() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(\"string\", notindexed=\"string\")",
            sql(config.column(string, [.unindexed])))
    }

    func test_external_content_view() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(content=\"view\")",
            sql(config.externalContent(_view )))
    }

    func test_external_content_virtual_table() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(content=\"virtual_table\")",
            sql(config.externalContent(virtualTable)))
    }

    func test_tokenizer_simple() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=simple)",
            sql(config.tokenizer(.Simple)))
    }

    func test_tokenizer_porter() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=porter)",
            sql(config.tokenizer(.Porter)))
    }

    func test_tokenizer_unicode61() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=unicode61)",
            sql(config.tokenizer(.Unicode61())))
    }

    func test_tokenizer_unicode61_with_options() {
        XCTAssertEqual(
            """
            CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(tokenize=unicode61 \"remove_diacritics=1\"
             \"tokenchars=.\" \"separators=X\")
            """.replacingOccurrences(of: "\n", with: ""),
            sql(config.tokenizer(.Unicode61(removeDiacritics: true, tokenchars: ["."], separators: ["X"]))))
    }

    func test_content_less() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(content=\"\")",
            sql(config.contentless()))
    }

    func test_config_matchinfo() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(matchinfo=\"fts3\")",
            sql(config.matchInfo(.fts3)))
    }

    func test_config_order_asc() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(order=\"asc\")",
            sql(config.order(.asc)))
    }

    func test_config_order_desc() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(order=\"desc\")",
            sql(config.order(.desc)))
    }

    func test_config_compress() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(compress=\"compress_foo\")",
            sql(config.compress("compress_foo")))
    }

    func test_config_uncompress() {
        XCTAssertEqual(
           "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(uncompress=\"uncompress_foo\")",
            sql(config.uncompress("uncompress_foo")))
    }

    func test_config_languageId() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(languageid=\"lid\")",
            sql(config.languageId("lid")))
    }

    func test_config_all() {
        XCTAssertEqual(
            """
            CREATE VIRTUAL TABLE \"virtual_table\" USING fts4(\"int\", \"string\", \"date\",
             tokenize=porter, prefix=\"2,4\", content=\"table\", notindexed=\"string\", notindexed=\"date\",
             languageid=\"lid\", matchinfo=\"fts3\", order=\"desc\")
            """.replacingOccurrences(of: "\n", with: ""),
            sql(config
                .tokenizer(.Porter)
                .column(int)
                .column(string, [.unindexed])
                .column(date, [.unindexed])
                .externalContent(table)
                .matchInfo(.fts3)
                .languageId("lid")
                .order(.desc)
                .prefix([2, 4]))
        )
    }

    func sql(_ config: FTS4Config) -> String {
        virtualTable.create(.FTS4(config))
    }
}
