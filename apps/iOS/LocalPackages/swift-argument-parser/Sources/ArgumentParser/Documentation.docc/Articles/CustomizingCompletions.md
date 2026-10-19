# Customizing Completions

Provide custom shell completions for your command-line tool's arguments and options.

## Overview

`ArgumentParser` provides default completions for any types that it can. For example, an `@Option` property that is a `CaseIterable` type will automatically have the correct values as completion suggestions.

When declaring an option or argument, you can customize the completions that are offered by specifying a ``CompletionKind``. With this completion kind you can specify that the value should be a file, a directory, or one of a list of strings:

```swift
struct Example: ParsableCommand {
    @Option(help: "The file to read from.", completion: .file())
    var input: String

    @Option(help: "The output directory.", completion: .directory)
    var outputDir: String

    @Option(help: "The preferred file format.", completion: .list(["markdown", "rst"]))
    var format: String

    enum CompressionType: String, CaseIterable, ExpressibleByArgument {
        case zip, gzip
    }

    @Option(help: "The compression type to use.")
    var compression: CompressionType
}
```

The generated completion script will suggest only file names for the `--input` option, only directory names for `--output-dir`, and only the strings `markdown` and `rst` for `--format`. The `--compression` option uses the default completions for a `CaseIterable` type, so the completion script will suggest `zip` and `gzip`.

You can define the default completion kind for custom ``ExpressibleByArgument`` types by implementing ``ExpressibleByArgument/defaultCompletionKind-866se``. For example, any arguments or options with this `File` type will automatically use files for completions:

```swift
struct File: Hashable, ExpressibleByArgument {
    var path: String
    
    init?(argument: String) {
        self.path = argument
    }
    
    static var defaultCompletionKind: CompletionKind {
        .file()
    }
}
```

For even more control over the suggested completions, you can specify a function that will be called during completion by using the `.custom` completion kind.

```swift
func listExecutables(_ arguments: [String]) -> [String] {
    // Generate the list of executables in the current directory
}

struct SwiftRun {
    @Option(help: "The target to execute.", completion: .custom(listExecutables))
    var target: String?
}
```

In this example, when a user requests completions for the `--target` option, the completion script runs the `SwiftRun` command-line tool with a special syntax, calling the `listExecutables` function with an array of the arguments given so far.
