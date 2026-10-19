# Providing Custom Validation

Provide helpful feedback to users when things go wrong.

## Overview

While `ArgumentParser` validates that the inputs given by your user match the requirements and types that you define in each command, there are some requirements that can't easily be described in Swift's type system, such as the number of elements in an array, or an expected integer value.

### Validating Command-Line Input

To validate your commands properties after parsing, implement the ``ParsableArguments/validate()-5r0ge`` method on any ``ParsableCommand``, ``ParsableArguments``, or ``AsyncParsableCommand`` type. Throwing an error from the `validate()` method causes the program to print a message to standard error and exit with an error code, preventing the `run()` method from being called with invalid inputs.

Here's a command that prints out one or more random elements from the list you provide. Its `validate()` method catches three different errors that a user can make and throws a relevant error for each one.

```swift
struct Select: ParsableCommand {
    @Option var count: Int = 1
    @Argument var elements: [String] = []

    mutating func validate() throws {
        guard count >= 1 else {
            throw ValidationError("Please specify a 'count' of at least 1.")
        }

        guard !elements.isEmpty else {
            throw ValidationError("Please provide at least one element to choose from.")
        }

        guard count <= elements.count else {
            throw ValidationError("Please specify a 'count' less than the number of elements.")
        }
    }

    mutating func run() {
        print(elements.shuffled().prefix(count).joined(separator: "\n"))
    }
}
```

When you provide useful error messages, they can guide new users to success with your command-line tool!

```
% select
Error: Please provide at least one element to choose from.
Usage: select [--count <count>] [<elements> ...]
  See 'select --help' for more information.
% select --count 2 hello
Error: Please specify a 'count' less than the number of elements.
Usage: select [--count <count>] [<elements> ...]
  See 'select --help' for more information.
% select --count 0 hello hey hi howdy
Error: Please specify a 'count' of at least 1.
Usage: select [--count <count>] [<elements> ...]
  See 'select --help' for more information.
% select --count 2 hello hey hi howdy
howdy
hey
```

## Handling Post-Validation Errors

The ``ValidationError`` type is a special `ArgumentParser` error — a validation error's message is always accompanied by an appropriate usage string. You can throw other errors, from either the `validate()` or `run()` method to indicate that something has gone wrong that isn't validation-specific. Errors that conform to `CustomStringConvertible` or `LocalizedError` provide the best experience for users.

```swift
struct LineCount: ParsableCommand {
    @Argument var file: String

    mutating func run() throws {
        let contents = try String(contentsOfFile: file, encoding: .utf8)
        let lines = contents.split(separator: "\n")
        print(lines.count)
    }
}
```

The throwing `String(contentsOfFile:encoding:)` initializer fails when the user specifies an invalid file. `ArgumentParser` prints its error message to standard error and exits with an error code.

```
% line-count file1.swift
37
% line-count non-existing-file.swift
Error: The file “non-existing-file.swift” couldn’t be opened because
there is no such file.
```

If you print your error output yourself, you still need to throw an error from `validate()` or `run()`, so that your command exits with the appropriate exit code. To avoid printing an extra error message, use the `ExitCode` error, which has static properties for success, failure, and validation errors, or lets you specify a specific exit code.

```swift
struct RuntimeError: Error, CustomStringConvertible {
    var description: String
}

struct Example: ParsableCommand {
    @Argument var inputFile: String

    mutating func run() throws {
        if !ExampleCore.processFile(inputFile) {
            // ExampleCore.processFile(_:) prints its own errors
            // and returns `false` on failure.
            throw ExitCode.failure
        }
    }
}
```

## Handling Transform Errors

During argument and option parsing, you can use a closure to transform the command line strings to custom types. If this transformation fails, you can throw a `ValidationError`; its `message` property will be displayed to the user.

In addition, you can throw your own errors. Errors that conform to `CustomStringConvertible` or `LocalizedError` provide the best experience for users.

```swift
struct ExampleTransformError: Error, CustomStringConvertible {
  var description: String
}

struct ExampleDataModel: Codable {
  let identifier: UUID
  let tokens: [String]
  let tokenCount: Int

  static func dataModel(_ jsonString: String) throws -> ExampleDataModel  {
    guard let data = jsonString.data(using: .utf8) else { throw ValidationError("Badly encoded string, should be UTF-8") }
    return try JSONDecoder().decode(ExampleDataModel.self, from: data)
  }
}

struct Example: ParsableCommand {
  // Reads in the argument string and attempts to transform it to
  // an `ExampleDataModel` object using the `JSONDecoder`. If the
  // string is not valid JSON, `decode` will throw an error and
  // parsing will halt.
  @Argument(transform: ExampleDataModel.dataModel)
  var inputJSON: ExampleDataModel

  // Specifiying this option will always cause the parser to exit
  // and print the custom error.
  @Option(transform: { throw ExampleTransformError(description: "Trying to write to failOption always produces an error. Input: \($0)") })
  var failOption: String?
}
```

Throwing from a transform closure benefits users by providing context and can reduce development time by pinpointing issues quickly and more precisely.

```
% example '{"Bad JSON"}'
Error: The value '{"Bad JSON"}' is invalid for '<input-json>': dataCorrupted(Swift.DecodingError.Context(codingPath: [], debugDescription: "The given data was not valid JSON.", underlyingError: Optional(Error Domain=NSCocoaErrorDomain Code=3840 "No value for key in object around character 11." UserInfo={NSDebugDescription=No value for key in object around character 11.})))
Usage: example <input-json> --fail-option <fail-option>
  See 'select --help' for more information.
```

While throwing standard library or Foundation errors adds context, custom errors provide the best experience for users and developers.

```
% example '{"tokenCount":0,"tokens":[],"identifier":"F77D661C-C5B7-448E-9344-267B284F66AD"}' --fail-option="Some Text Here!"
Error: The value 'Some Text Here!' is invalid for '--fail-option <fail-option>': Trying to write to failOption always produces an error. Input: Some Text Here!
Usage: example <input-json> --fail-option <fail-option>
  See 'select --help' for more information.
```
