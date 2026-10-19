# Experimental Features

Learn about ArgumentParser's experimental features.

## Overview

Command-line programs built using `ArgumentParser` may include some built-in experimental features, available with the prefix `--experimental`. These features should not be considered stable while still prefixed, as future releases may change their behavior or remove them.

If you have any feedback on experimental features, please [open a GitHub issue][issue].

## List of Experimental Features

| Name | Description | related PRs | Version |
| ------------- | ------------- | ------------- | ------------- |
| `--experimental-dump-help`  | Dumps command/argument/help information as JSON | [#310][] [#335][] | 0.5.0 or newer |

[#310]: https://github.com/apple/swift-argument-parser/pull/310
[#335]: https://github.com/apple/swift-argument-parser/pull/335
[issue]: https://github.com/apple/swift-argument-parser/issues/new/choose 
