import SQLite

let table = Table("test")
let name = Expression<String>("name")

let db = try Connection("db.sqlite", readonly: true)

for row in try db.prepare(table) {
    print(row[name])
}
