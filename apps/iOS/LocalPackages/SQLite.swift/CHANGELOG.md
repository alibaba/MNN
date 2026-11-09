0.15.3 (19-04-2024), [diff][diff-0.15.3]
========================================
* Update `podspec` to include privacy manifest ([#1265][])

0.15.2 (16-04-2024), [diff][diff-0.15.2]
========================================
* fix: visionos to cocoapods ([#1260][])

0.15.1 (14-04-2024), [diff][diff-0.15.1]
========================================

* Update CoreFunctions.swift fix typo ([#1249][])
* Fix #1247 support nil case when decoding optionals ([#1248][])
* Change deployment targets for Xcode 15 and add dependency on custom Cocoapods fork ([#1255][])
* Add VisionOS support ([#1237][])

0.15.0 (24-02-2024), [diff][diff-0.15.0]
========================================

* Fix incorrect behavior when preparing `SELECT *` preceded by a `WITH` ([#1179][])
* Adds support for returning extended error codes ([#1178][])
* Fix typos ([#1182][])
* fix Xcode build error ([#1192][])
* Make the IndexDefinition properties public ([#1196][])
* Fix GitHub Actions build badge ([#1200][])
* Run CI on macOS 13 ([#1206][])
* SchemaReader: return the correct column definition for a composite primary key ([#1217][])
* Add optional support for decoding ([#1224][])
* make fromDatatypeValue throw ([#1242][])
* Implements built-in window functions ([#1228][])
* Fix column affinity parsing to match how SQLite determines affinity ([#1218][])
* Handle FK definitions w/o key references ([#1210][])
* Add privacy manifest ([#1245][])
* New minimum deployment targets: iOS/tvOS 11.0, watchOS 4.0

0.14.1 (01-11-2022), [diff][diff-0.14.1]
========================================

* Reverted `Blob` changes (See [#1167][] for rationale).

0.14.0 (27-10-2022), [diff][diff-0.14.0]
========================================
For breaking changes, see [Upgrading.md](Documentation/Upgrading.md).

* Support more complex schema changes and queries ([#1073][], [#1146][] [#1148][])
* Support `ATTACH`/`DETACH` ([#30][], [#1142][])
* Expose connection flags (via `URIQueryParameter`) to open db ([#1074][])
* Support `WITH` clause ([#1139][])
* Add `Value` conformance for `NSURL` ([#1110][], [#1141][])
* Add decoding for `UUID` ([#1137][])
* SQLCipher: improve documentation ([#1098][]), add `sqlcipher_export` ([#1101][])
* Fix `insertMany([Encodable])` ([#1130][], [#1138][])
* Fix incorrect spelling of `remove_diacritics` ([#1128][])
* Fix project build order ([#1131][])
* Blob performance improvements ([#416][], [#1167][])
* Various performance improvements ([#1109][], [#1115][], [#1132][])
* Removed FTS3/4 tokenizer integration (`registerTokenizer`, [#1104][], [#1144][])

0.13.3 (27-03-2022), [diff][diff-0.13.3]
========================================

* UUID Fix ([#1112][])
* Add prepareRowIterator method to an extension of Statement. ([#1119][])
* Adding primary key support to column with references ([#1121][])

0.13.2 (25-01-2022), [diff][diff-0.13.2]
========================================

* Closing bracket position ([#1100][])
* Native user_version support in Connection ([#1105][])

0.13.1 (17-11-2021), [diff][diff-0.13.1]
========================================

* Support for database backup ([#919][])
* Support for custom SQL aggregates ([#881][])
* Restore previous behavior in `FailableIterator` ([#1075][])
* Fix compilation on Linux ([#1077][])
* Align platform versions in SPM manifest and Xcode ([#1094][])
* Revert OSX deployment target back to 10.10 ([#1095][])

0.13.0 (22-08-2021), [diff][diff-0.13.0]
========================================

* Swift 5.3 support
* Xcode 12.5 support
* Bumps minimum deployment versions
* Fixes up Package.swift to build SQLiteObjc module

0.12.1, 0.12.2 (21-06-2019) [diff][diff-0.12.2]
========================================

* CocoaPods modular headers support

0.12.0 (24-04-2019) [diff][diff-0.12.0]
========================================

* Version with Swift 5 Support

0.11.6 (19-04-2019), [diff][diff-0.11.6]
========================================

* Swift 4.2, SQLCipher 4.x ([#866][])

0.11.5 (04-14-2018), [diff][diff-0.11.5]
========================================

* Swift 4.1 ([#797][])

0.11.4 (30-09-2017), [diff][diff-0.11.4]
========================================

* Collate `.nocase` strictly enforces `NOT NULL` even when using Optional ([#697][])
* Fix transactions not being rolled back when committing fails ([#426][])
* Add possibility to have expression on right hand side of like ([#591][])
* Added Date and Time functions ([#142][])
* Add Swift4 Coding support ([#733][])
* Preliminary Linux support ([#315][], [#681][])
* Add `RowIterator` for more safety ([#647][], [#726][])
* Make `Row.get` throw instead of crash ([#649][])
* Fix create/drop index functions ([#666][])
* Revert deployment target to 8.0 ([#625][], [#671][], [#717][])
* Added support for the union query clause ([#723][])
* Add support for `ORDER` and `LIMIT` on `UPDATE` and `DELETE` ([#657][], [#722][])
* Swift 4 support ([#668][])

0.11.3 (30-03-2017), [diff][diff-0.11.3]
========================================

* Fix compilation problems when using Carthage ([#615][])
* Add `WITHOUT ROWID` table option ([#541][])
* Argument count fixed for binary custom functions ([#481][])
* Documentation updates
* Tested with Xcode 8.3 / iOS 10.3

0.11.2 (25-12-2016), [diff][diff-0.11.2]
========================================

* Fixed SQLCipher integration with read-only databases ([#559][])
* Preliminary Swift Package Manager support ([#548][], [#560][])
* Fixed null pointer when fetching an empty BLOB ([#561][])
* Allow `where` as alias for `filter` ([#571][])

0.11.1 (06-12-2016), [diff][diff-0.11.1]
========================================

* Integrate SQLCipher via CocoaPods ([#546][], [#553][])
* Made lastInsertRowid consistent with other SQLite wrappers ([#532][])
* Fix for `~=` operator used with Double ranges
* Various documentation updates

0.11.0 (19-10-2016)
===================

* Swift3 migration ([diff][diff-0.11.0])


[diff-0.11.0]: https://github.com/stephencelis/SQLite.swift/compare/0.10.1...0.11.0
[diff-0.11.1]: https://github.com/stephencelis/SQLite.swift/compare/0.11.0...0.11.1
[diff-0.11.2]: https://github.com/stephencelis/SQLite.swift/compare/0.11.1...0.11.2
[diff-0.11.3]: https://github.com/stephencelis/SQLite.swift/compare/0.11.2...0.11.3
[diff-0.11.4]: https://github.com/stephencelis/SQLite.swift/compare/0.11.3...0.11.4
[diff-0.11.5]: https://github.com/stephencelis/SQLite.swift/compare/0.11.4...0.11.5
[diff-0.11.6]: https://github.com/stephencelis/SQLite.swift/compare/0.11.5...0.11.6
[diff-0.12.0]: https://github.com/stephencelis/SQLite.swift/compare/0.11.6...0.12.0
[diff-0.12.2]: https://github.com/stephencelis/SQLite.swift/compare/0.12.0...0.12.2
[diff-0.13.0]: https://github.com/stephencelis/SQLite.swift/compare/0.12.2...0.13.0
[diff-0.13.1]: https://github.com/stephencelis/SQLite.swift/compare/0.13.0...0.13.1
[diff-0.13.2]: https://github.com/stephencelis/SQLite.swift/compare/0.13.1...0.13.2
[diff-0.13.3]: https://github.com/stephencelis/SQLite.swift/compare/0.13.2...0.13.3
[diff-0.14.0]: https://github.com/stephencelis/SQLite.swift/compare/0.13.3...0.14.0
[diff-0.14.1]: https://github.com/stephencelis/SQLite.swift/compare/0.14.0...0.14.1
[diff-0.15.0]: https://github.com/stephencelis/SQLite.swift/compare/0.14.0...0.15.0
[diff-0.15.1]: https://github.com/stephencelis/SQLite.swift/compare/0.15.0...0.15.1
[diff-0.15.2]: https://github.com/stephencelis/SQLite.swift/compare/0.15.1...0.15.2

[#30]: https://github.com/stephencelis/SQLite.swift/issues/30
[#142]: https://github.com/stephencelis/SQLite.swift/issues/142
[#315]: https://github.com/stephencelis/SQLite.swift/issues/315
[#416]: https://github.com/stephencelis/SQLite.swift/pull/416
[#426]: https://github.com/stephencelis/SQLite.swift/pull/426
[#481]: https://github.com/stephencelis/SQLite.swift/pull/481
[#532]: https://github.com/stephencelis/SQLite.swift/issues/532
[#541]: https://github.com/stephencelis/SQLite.swift/issues/541
[#546]: https://github.com/stephencelis/SQLite.swift/issues/546
[#548]: https://github.com/stephencelis/SQLite.swift/pull/548
[#553]: https://github.com/stephencelis/SQLite.swift/pull/553
[#559]: https://github.com/stephencelis/SQLite.swift/pull/559
[#560]: https://github.com/stephencelis/SQLite.swift/pull/560
[#561]: https://github.com/stephencelis/SQLite.swift/issues/561
[#571]: https://github.com/stephencelis/SQLite.swift/issues/571
[#591]: https://github.com/stephencelis/SQLite.swift/pull/591
[#615]: https://github.com/stephencelis/SQLite.swift/pull/615
[#625]: https://github.com/stephencelis/SQLite.swift/issues/625
[#647]: https://github.com/stephencelis/SQLite.swift/pull/647
[#649]: https://github.com/stephencelis/SQLite.swift/pull/649
[#657]: https://github.com/stephencelis/SQLite.swift/issues/657
[#666]: https://github.com/stephencelis/SQLite.swift/pull/666
[#668]: https://github.com/stephencelis/SQLite.swift/pull/668
[#671]: https://github.com/stephencelis/SQLite.swift/issues/671
[#681]: https://github.com/stephencelis/SQLite.swift/issues/681
[#697]: https://github.com/stephencelis/SQLite.swift/issues/697
[#717]: https://github.com/stephencelis/SQLite.swift/issues/717
[#722]: https://github.com/stephencelis/SQLite.swift/pull/722
[#723]: https://github.com/stephencelis/SQLite.swift/pull/723
[#733]: https://github.com/stephencelis/SQLite.swift/pull/733
[#726]: https://github.com/stephencelis/SQLite.swift/pull/726
[#797]: https://github.com/stephencelis/SQLite.swift/pull/797
[#866]: https://github.com/stephencelis/SQLite.swift/pull/866
[#881]: https://github.com/stephencelis/SQLite.swift/pull/881
[#919]: https://github.com/stephencelis/SQLite.swift/pull/919
[#1073]: https://github.com/stephencelis/SQLite.swift/issues/1073
[#1074]: https://github.com/stephencelis/SQLite.swift/issues/1074
[#1075]: https://github.com/stephencelis/SQLite.swift/pull/1075
[#1077]: https://github.com/stephencelis/SQLite.swift/issues/1077
[#1094]: https://github.com/stephencelis/SQLite.swift/pull/1094
[#1095]: https://github.com/stephencelis/SQLite.swift/pull/1095
[#1098]: https://github.com/stephencelis/SQLite.swift/issues/1098
[#1100]: https://github.com/stephencelis/SQLite.swift/pull/1100
[#1101]: https://github.com/stephencelis/SQLite.swift/issues/1101
[#1104]: https://github.com/stephencelis/SQLite.swift/issues/1104
[#1105]: https://github.com/stephencelis/SQLite.swift/pull/1105
[#1109]: https://github.com/stephencelis/SQLite.swift/issues/1109
[#1110]: https://github.com/stephencelis/SQLite.swift/pull/1110
[#1112]: https://github.com/stephencelis/SQLite.swift/pull/1112
[#1115]: https://github.com/stephencelis/SQLite.swift/pull/1115
[#1119]: https://github.com/stephencelis/SQLite.swift/pull/1119
[#1121]: https://github.com/stephencelis/SQLite.swift/pull/1121
[#1128]: https://github.com/stephencelis/SQLite.swift/issues/1128
[#1130]: https://github.com/stephencelis/SQLite.swift/issues/1130
[#1131]: https://github.com/stephencelis/SQLite.swift/pull/1131
[#1132]: https://github.com/stephencelis/SQLite.swift/pull/1132
[#1137]: https://github.com/stephencelis/SQLite.swift/pull/1137
[#1138]: https://github.com/stephencelis/SQLite.swift/pull/1138
[#1139]: https://github.com/stephencelis/SQLite.swift/pull/1139
[#1141]: https://github.com/stephencelis/SQLite.swift/pull/1141
[#1142]: https://github.com/stephencelis/SQLite.swift/pull/1142
[#1144]: https://github.com/stephencelis/SQLite.swift/pull/1144
[#1146]: https://github.com/stephencelis/SQLite.swift/pull/1146
[#1148]: https://github.com/stephencelis/SQLite.swift/pull/1148
[#1167]: https://github.com/stephencelis/SQLite.swift/pull/1167
[#1179]: https://github.com/stephencelis/SQLite.swift/pull/1179
[#1178]: https://github.com/stephencelis/SQLite.swift/pull/1178
[#1182]: https://github.com/stephencelis/SQLite.swift/pull/1182
[#1192]: https://github.com/stephencelis/SQLite.swift/pull/1192
[#1196]: https://github.com/stephencelis/SQLite.swift/pull/1196
[#1200]: https://github.com/stephencelis/SQLite.swift/pull/1200
[#1206]: https://github.com/stephencelis/SQLite.swift/pull/1206
[#1217]: https://github.com/stephencelis/SQLite.swift/pull/1217
[#1224]: https://github.com/stephencelis/SQLite.swift/pull/1224
[#1242]: https://github.com/stephencelis/SQLite.swift/pull/1242
[#1228]: https://github.com/stephencelis/SQLite.swift/pull/1228
[#1218]: https://github.com/stephencelis/SQLite.swift/pull/1218
[#1210]: https://github.com/stephencelis/SQLite.swift/pull/1210
[#1245]: https://github.com/stephencelis/SQLite.swift/pull/1245
[#1249]: https://github.com/stephencelis/SQLite.swift/pull/1249
[#1248]: https://github.com/stephencelis/SQLite.swift/pull/1248
[#1255]: https://github.com/stephencelis/SQLite.swift/pull/1255
[#1237]: https://github.com/stephencelis/SQLite.swift/pull/1237
[#1260]: https://github.com/stephencelis/SQLite.swift/pull/1260
[#1265]: https://github.com/stephencelis/SQLite.swift/pull/1265
