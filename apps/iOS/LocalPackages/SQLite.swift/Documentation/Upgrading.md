# Upgrading

## 0.13 → 0.14

- `Expression.asSQL()` is no longer available. Expressions now implement `CustomStringConvertible`,
  where `description` returns the SQL.
- `Statement.prepareRowIterator()` is now longer available. Instead, use the methods
  of the same name on `Connection`.
- `Connection.registerTokenizer` is no longer available to register custom FTS4 tokenizers.
