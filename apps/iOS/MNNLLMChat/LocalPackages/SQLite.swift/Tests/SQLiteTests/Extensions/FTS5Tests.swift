import XCTest
import SQLite

class FTS5Tests: XCTestCase {
    var config: FTS5Config!

    override func setUp() {
        super.setUp()
        config = FTS5Config()
    }

    func test_empty_config() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5()",
            sql(config))
    }

    func test_config_column() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(\"string\")",
            sql(config.column(string)))
    }

    func test_config_columns() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(\"string\", \"int\")",
            sql(config.columns([string, int])))
    }

    func test_config_unindexed_column() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(\"string\" UNINDEXED)",
            sql(config.column(string, [.unindexed])))
    }

    func test_external_content_table() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(content=\"table\")",
            sql(config.externalContent(table)))
    }

    func test_external_content_view() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(content=\"view\")",
            sql(config.externalContent(_view)))
    }

    func test_external_content_virtual_table() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(content=\"virtual_table\")",
            sql(config.externalContent(virtualTable)))
    }

    func test_content_less() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(content=\"\")",
            sql(config.contentless()))
    }

    func test_content_rowid() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(content_rowid=\"string\")",
            sql(config.contentRowId(string)))
    }

    func test_tokenizer_porter() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(tokenize=porter)",
            sql(config.tokenizer(.Porter)))
    }

    func test_tokenizer_unicode61() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(tokenize=unicode61)",
            sql(config.tokenizer(.Unicode61())))
    }

    func test_tokenizer_unicode61_with_options() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(tokenize=unicode61 \"remove_diacritics=1\" \"tokenchars=.\" \"separators=X\")",
            sql(config.tokenizer(.Unicode61(removeDiacritics: true, tokenchars: ["."], separators: ["X"]))))
    }

    func test_tokenizer_trigram() {
        XCTAssertEqual(
                "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(tokenize=trigram case_sensitive 0)",
                sql(config.tokenizer(.Trigram())))

        XCTAssertEqual(
                "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(tokenize=trigram case_sensitive 1)",
                sql(config.tokenizer(.Trigram(caseSensitive: true))))
    }

    func test_column_size() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(columnsize=1)",
            sql(config.columnSize(1)))
    }

    func test_detail_full() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(detail=\"full\")",
            sql(config.detail(.full)))
    }

    func test_detail_column() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(detail=\"column\")",
            sql(config.detail(.column)))
    }

    func test_detail_none() {
        XCTAssertEqual(
            "CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(detail=\"none\")",
            sql(config.detail(.none)))
    }

    func test_fts5_config_all() {
        XCTAssertEqual(
            """
            CREATE VIRTUAL TABLE \"virtual_table\" USING fts5(\"int\", \"string\" UNINDEXED, \"date\" UNINDEXED,
             tokenize=porter, prefix=\"2,4\", content=\"table\")
            """.replacingOccurrences(of: "\n", with: ""),
            sql(config
                .tokenizer(.Porter)
                .column(int)
                .column(string, [.unindexed])
                .column(date, [.unindexed])
                .externalContent(table)
                .prefix([2, 4]))
        )
    }

    func sql(_ config: FTS5Config) -> String {
        virtualTable.create(.FTS5(config))
    }
}
