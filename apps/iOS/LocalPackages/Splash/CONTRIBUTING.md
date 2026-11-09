# Splash Contribution Guide

Welcome to the *Splash Contribution Guide* - a document that aims to give you all the information you need to contribute to the Splash project.

Before you continue, make sure that you've read & agree to [the Code of Conduct](https://github.com/JohnSundell/Splash/blob/master/CODE_OF_CONDUCT.md).

## Bugs, feature requests and support

Splash doesn't use GitHub issues, so all form of support - whether that's asking a question, reporting a bug, or discussing a feature request - takes place in Pull Requests.

The idea behind this workflow is to encourage more people using Splash to dive into the source code and familiarize themselves with how it works, in order to better be able to *self-service* on bugs and issues - hopefully leading to a better and more fluid experience for everyone involved 😊.

*This workflow is still very much an experiment, so please be patient and try to keep an open mind as we work things out together* 😉

**🐞 I found a bug, how do I report it?**

If you find a bug, for example a piece of code that doesn't highlight correctly, here's the recommended workflow:

1. Come up with the simplest code possible that reproduces the issue (`SplashTokenizer` can be a great tool to use in order to quickly test how Splash tokenizes a string of code).
2. Write a test case using the code that reproduces the issue. [See Splash's existing tests for inspiration on how to get started](https://github.com/JohnSundell/Splash/tree/master/Tests/SplashTests/Tests). When writing a test, you essentially give `SyntaxHighlighter` a string to highlight, and then perform an `XCTAssertEqual` against an array of expected components.
3. Either fix the bug yourself, or simply submit your failing test as a Pull Request, and we can work on a solution together.

While doing the above does require a bit of extra work for the person who found the bug, it gives us as a community a very nice starting point for fixing issues - hopefully leading to quicker fixes and a more constructive workflow.

**💡 I have an idea for a feature request!**

First of all, that's awesome! 👍 Your ideas on how to make Splash better and more powerful are super welcome. Here's the recommended workflow for feature requests:

1. Do some prototyping and come up with a sample implementation of your idea or feature request. Note that this doesn't have to be a fully working, complete implementation, just something that illustrates the feature and your idea on how it could be added to Splash.
2. Submit your sample implementation as a Pull Request. Use the description field to write down why you think the feature should be added and some initial discussion points.
3. Together we'll discuss the feature and your sample implementation, and either accept it as-is, use it as a starting point for a new implementation, or decide that the idea is not worth implementing as this time.

**🤔 I have a question that the documentation doesn't yet answer**

With Splash, the goal is to end up with state of the art documentation that answers most of the questions that both users and developers of the tool might have - and the only way to get there is through continued improvement, with your help.

Here's the recommended workflow for getting your question answered:

1. Start by looking through the code. Splash is a normal Swift package that uses standard Swift conventions with a (hopefully 😅) well-defined structure. Chances are high that you'll be able to answer your own question by reading through the implementation, the tests, and the inline code documentation.
2. If you found out the answer to your question (congrats! 🎉) - then don't stop there. Other people will probably ask themselves the same question at some point - so let's improve the documentation! Find an appropriate place where your question could've been answered by clearer documentation or a better structure (for example this document, or inline in the code) - and add the documentation you wish would've been there. If you didn't manage to find an answer (no worries, we're all always learning 👍), write down your question as a comment - either in the code or in one of the Markdown documents.
3. Submit your new documentation or your comment as a Pull Request, and we'll work on improving the documentation together.

## Design and technical decisions

Like most programs & frameworks, Splash could've been written in many different ways. Specifically for the task of Swift syntax highlighting, there were three main options to consider:

1. Apply regular expressions to the code in order to tokenize it. This is how most JavaScript-based syntax highlighters work. It's a common and proven approach, but it usually doesn't yield the most accurate results (writing really granular regular expressions is really hard), and can be a bit alienating for people who haven't used advanced regular expressions before.
2. Hook into Apple's SourceKit service. SourceKit is what powers Xcode's syntax highlighting, and works in tandem with the Swift compiler to tokenize and highlight code. SourceKit is awesome, but using it is quite complicated and requires cross-process communication.
3. Simply parse the code manually. Like all programming languages, Swift has a well-defined syntax and clear grammar that we can model in code, in order to parse and tokenize code by iterating through it.

When I first started exploring the idea of a custom Swift syntax highlighter, I built quick prototypes using all of the above three techniques - and the one that I liked the most (by far) was option number 3. Writing Splash as a normal Swift package, using normal Swift code, with standard Swift conventions turned out (at least for me) to be the most easy to understand and easy to work with solution.

The next challenge then became to decide exactly *how* to write such a Swift package. Swift's syntax changes over time, so Splash required a flexible setup in order to avoid becoming hard to maintain due to complicated logic and lots of different conditions scattered all over the code.

## Architectural overview

Splash's architecture was designed to enable easy tweaking of how it parses and tokenizes code, and to make bugs and edge cases easier to debug - but also to hide all those implementation details from the API user.

**SyntaxHighlighter**

The most top level API that most API users will interact with is `SyntaxHighlighter`. It doesn't do much itself, but instead works as the *"middleman"* between the internal `Tokenizer` type and user-configurable implementations of `Grammar` and `OutputFormat`.

So as an API user, all you have to know in order to use Splash is this:

```swift
let highlighter = SyntaxHighlighter(format: HTMLOutputFormat())
let code = highlighter.highlight("func hello() -> Int")
```

**Tokenizer**

The `Tokenizer` type enables `SyntaxHighlighter` to ask for a sequence of code segments for a given string, using a set of delimiters to use to split the code up. It then iterates through each character of the given string and uses the set of delimiters to check when each segment should begin and end.

The reason `Tokenizer` doesn't simply *split* the string is to enable Splash to have as close to `O(N)` performance characteristics as possible. If we first had to split the string, *then* iterate through it, we would always make at least two passes through the code. With the current approach, only one full pass has to be made.

**Grammar**

What delimiters that `Tokenizer` should use to split the code up into segments is determined by the given language `Grammar`. The default implementation is called `SwiftGrammar`, which aims to mimic the behavior of the Swift compiler as close as possible without actually having to compile the code (which is what enables Splash to be so fast).

The decision to not simply hardcode `SwiftGrammar` across the code base was to [decouple the code using a protocol](https://www.swiftbysundell.com/posts/separation-of-concerns-using-protocols-in-swift) (in this case `Grammar`) to achieve a much more flexible solution. If the Swift grammar changes a lot in the future, we can always add a second implementation while still maintaining backward compatibility, and it also opens up the possibility of using Splash with languages other than Swift - since it doesn't make many (if any) hard assumptions about Swift itself (Objective-C support, anyone? 😉).

Apart from supplying `Tokenizer` with delimiters, the most important role of a `Grammar` implementation is to provide an array of `SyntaxRule` implementations. When `SyntaxHighlighter` iterates through the segments that its `Tokenizer` gave it, it applies the syntax rules from its `Grammar` to each one of them to figure out each token's type. Each rule is asked if it matches a given segment, and as soon as a match is found that rule's `TokenType` is used to determine the type of that token.

Have a look at `SwiftGrammar` to see all of its `SyntaxRule` implementations and how they decide how to classify each token using code segments.

**OutputFormat**

The final piece of the puzzle is `OutputFormat`, which determines how the result of tokenizing a string of code should be transformed into its final form. Splash ships with two implementations of this protocol `HTMLOutputFormat` and `AttributedStringOutputFormat`, but the framework makes no assumptions about what output format that the API user may want, since the output format can be fully customized.

An `OutputFormat` has two responsibilities. The first is to define what type that the output will actually be (through its `Output` associated type). The second is to construct an `OutputBuilder` to build up a value of that output format type. Splash uses the *[builder pattern](https://www.swiftbysundell.com/posts/using-the-builder-pattern-in-swift)* to be able to continuously build up the output as it iterates through each token, and at the end call `build()` on the builder to output the final result.

**Conclusion**

Hopefully this document has given you an introduction to how Splash works, both in terms of its recommended project workflow and its technical implementation. Feel free to submit Pull Requests to improve this document, and I look forward to working with you on Splash and seeing how you use it 😀
