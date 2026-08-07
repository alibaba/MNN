# Contributing

The where and when to open an [issue](#issues) or [pull
request](#pull-requests).


## Issues

Issues are used to track **bugs** and **feature requests**. Need **help** or
have a **general question**? [Ask on Stack Overflow][] (tag `sqlite.swift`).

Before reporting a bug or requesting a feature, [run a few searches][Search] to
see if a similar issue has already been opened and ensure you’re not submitting
a duplicate.

If you find a similar issue, read the existing conversation and see if it
addresses everything. If it doesn’t, continue the conversation there.

If your searches return empty, see the [bug](#bugs) or [feature
request](#feature-requests) guidelines below.

[Ask on Stack Overflow]: https://stackoverflow.com/questions/tagged/sqlite.swift
[Search]: https://github.com/stephencelis/SQLite.swift/search?type=Issues


### Bugs

Think you’ve discovered a new **bug**? Let’s try troubleshooting a few things
first.

  - **Is it an installation issue?** <a name='bugs-1'/>

    If this is your first time building SQLite.swift in your project, you may
    encounter a build error, _e.g._:

        No such module 'SQLite'

    Please carefully re-read the [installation instructions][] to make sure
    everything is in order.

  - **Have you read the documentation?** <a name='bugs-2'/>

    If you can’t seem to get something working, check
    [the documentation][See Documentation] to see if the solution is there.

  - **Are you up-to-date?** <a name='bugs-3'/>

    If you’re perusing [the documentation][See Documentation] online and find
    that an example is just not working, please upgrade to the latest version
    of SQLite.swift and try again before continuing.

  - **Is it an unhelpful build error?** <a name='bugs-4'/>

    While Swift error messaging is improving with each release, complex
    expressions still lend themselves to misleading errors. If you encounter an
    error on a complex line, breaking it down into smaller pieces generally
    yields a more understandable error.

  - **Is it an _even more_ unhelpful build error?** <a name='bugs-5'/>

    Have you updated Xcode recently? Did your project stop building out of the
    blue?

    Hold down the **option** key and select **Clean Build Folder…** from the
    **Product** menu (⌥⇧⌘K).

Made it through everything above and still having trouble? Sorry!
[Open an issue][]! And _please_:

  - Be as descriptive as possible.
  - Provide as much information needed to _reliably reproduce_ the issue.
  - Attach screenshots if possible.
  - Better yet: attach GIFs or link to video.
  - Even better: link to a sample project exhibiting the issue.
  - Include the SQLite.swift commit or branch experiencing the issue.
  - Include devices and operating systems affected.
  - Include build information: the Xcode and macOS versions affected.

[installation instructions]: Documentation/Index.md#installation
[See Documentation]: Documentation/Index.md#sqliteswift-documentation
[Open an issue]: https://github.com/stephencelis/SQLite.swift/issues/new


### Feature Requests

Have an innovative **feature request**? [Open an issue][]! Be thorough! Provide
context and examples. Be open to discussion.


## Pull Requests

Interested in contributing but don’t know where to start? Try the [`help
wanted`][help wanted] label.

Ready to submit a fix or a feature? [Submit a pull request][]! And _please_:

  - If code changes, run the tests and make sure everything still works.
  - Write new tests for new functionality.
  - Update documentation comments where applicable.
  - Maintain the existing style.
  - Don’t forget to have fun.

While we cannot guarantee a merge to every pull request, we do read each one
and love your input.


[help wanted]: https://github.com/stephencelis/SQLite.swift/labels/help%20wanted
[Submit a pull request]: https://github.com/stephencelis/SQLite.swift/fork
