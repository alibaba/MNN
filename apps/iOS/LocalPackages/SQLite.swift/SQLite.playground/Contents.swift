import SQLite

/// Create an in-memory database
let db = try Connection(.inMemory)

/// enable statement logging
db.trace { print($0) }

/// define a "users" table with some fields
let users = Table("users")

let id = Expression<Int64>("id")
let email = Expression<String>("email") // non-null
let name = Expression<String?>("name")  // nullable

/// prepare the query
let statement = users.create { t in
    t.column(id, primaryKey: true)
    t.column(email, unique: true, check: email.like("%@%"))
    t.column(name)
}

/// …and run it
try db.run(statement)

/// insert "alice"
let rowid = try db.run(users.insert(email <- "alice@mac.com"))

/// insert multiple rows using `insertMany`
let lastRowid = try db.run(users.insertMany([
  [email <- "bob@mac.com"],
  [email <- "mallory@evil.com"]
]))


let query = try db.prepare(users)
for user in query {
    print("id: \(user[id]), email: \(user[email])")
}

// re-requery just rowid of Alice
let alice = try db.prepare(users.filter(id == rowid))
for user in alice {
    print("id: \(user[id]), email: \(user[email])")
}

/// using the `RowIterator` API
let rowIterator = try db.prepareRowIterator(users)
for user in try Array(rowIterator) {
    print("id: \(user[id]), email: \(user[email])")
}

/// also with `map()`
let mapRowIterator = try db.prepareRowIterator(users)

let userIds = try mapRowIterator.map { $0[id] }

/// using `failableNext()` on `RowIterator`
let iterator = try db.prepareRowIterator(users)
do {
    while let row = try rowIterator.failableNext() {
        print(row)
    }
} catch {
    // Handle error
}

/// define a virtual table for the FTS index
let emails = VirtualTable("emails")

let subject = Expression<String>("subject")
let body = Expression<String?>("body")

/// create the index
try db.run(emails.create(.FTS5(
    FTS5Config()
      .column(subject)
      .column(body)
)))

/// populate with data
try db.run(emails.insert(
    subject <- "Hello, world!",
    body <- "This is a hello world message."
))

/// run a query
let ftsQuery = try db.prepare(emails.match("hello"))

for row in ftsQuery {
    print(row[subject])
}

/// custom aggregations
let reduce: (String, [Binding?]) -> String = { (last, bindings) in
    last + " " + (bindings.first as? String ?? "")
}

db.createAggregation("customConcat",
                     initialValue: "users:",
                     reduce: reduce,
                     result: { $0 })
let result = try db.prepare("SELECT customConcat(email) FROM users").scalar() as! String
print(result)

/// schema queries
let schema = db.schema
let objects = try schema.objectDefinitions()
print(objects)

let columns = try schema.columnDefinitions(table: "users")
print(columns)

/// schema alteration

let schemaChanger = SchemaChanger(connection: db)
try schemaChanger.alter(table: "users") { table in
    table.add(column: ColumnDefinition(name: "age", type: .INTEGER))
    table.rename(column: "email", to: "electronic_mail")
    table.drop(column: "name")
}

let changedColumns = try schema.columnDefinitions(table: "users")
print(changedColumns)

let age = Expression<Int?>("age")
let electronicMail = Expression<String>("electronic_mail")

let newRowid = try db.run(users.insert(
    electronicMail <- "carol@mac.com",
    age <- 33
))
