# Unicode Character Database Tools

- [Build Dependencies](#build-dependencies)
  - [Debian](#debian)
- [Building](#building)
- [Updating the UCD Data](#updating-the-ucd-data)
- [Bugs](#bugs)
- [License Information](#license-information)

----------

The Unicode Character Database (UCD) Tools is a set of Python tools and a
[C library](src/include/ucd/ucd.h) with a C++ API binding. The Python tools
are designed to support extracting and processing data from the text-based
UCD source files, while the C library is designed to provide easy access to
this information within a C or C++ program.

The project uses and supports the following sources of Unicode codepoint data:

*  [Unicode Character Database](http://www.unicode.org/Public/11.0.0/ucd/) 11.0.0
*  [Unicode Emoji](http://www.unicode.org/Public/emoji/11.0/) 11.0 (UTR #51)
*  [ConScript Unicode Registry](http://www.evertype.com/standards/csur/)

## Build Dependencies

In order to build ucd-tools, you need:

1.  a functional autotools system (`make`, `autoconf`, `automake` and `libtool`);
2.  a functional C and C++ compiler.

__NOTE__: The C++ compiler is used to build the test for the C++ API.

To build the documentation, you need:

1.  the doxygen program to build the api documentation;
2.  the dot program from the graphviz library to generate graphs in the api documentation.

### Debian

Core Dependencies:

| Dependency       | Install                                               |
|------------------|-------------------------------------------------------|
| autotools        | `sudo apt-get install make autoconf automake libtool` |
| C++ compiler     | `sudo apt-get install gcc g++`                        |

Documentation Dependencies:

| Dependency | Install                         |
|------------|---------------------------------|
| doxygen    | `sudo apt-get install doxygen`  |
| graphviz   | `sudo apt-get install graphviz` |

## Building

UCD Tools supports the standard GNU autotools build system. The source code
does not contain the generated `configure` files, so to build it you need to
run:

	./autogen.sh
	./configure --prefix=/usr
	make

The tests can be run by using:

	make check

The program can be installed using:

	sudo make install

The documentation can be built using:

	make html

## Updating the UCD Data

To re-generate the source files from the UCD data when a new version of
unicode is released, you need to run:

	./configure --prefix=/usr --with-unicode-version=VERSION
	make ucd-update

where `VERSION` is the Unicode version (e.g. `6.3.0`).

Additionally, you can use the `UCD_FLAGS` option to control how the data is
generated. The following flags are supported:

| Flag        | Description |
|-------------|-------------|
| --with-csur | Add ConScript Unicode Registry data. |

## Bugs

Report bugs to the [ucd-tools issues](https://github.com/rhdunn/ucd-tools/issues)
page on GitHub.

## License Information

UCD Tools is released under the GPL version 3 or later license.

The UCD data files in `data/ucd` are downloaded from the UCD website and are
licensed under the [Unicode Terms of Use](COPYING.UCD). These data files are
used in their unmodified form. They have the following Copyright notice:

    Copyright © 1991-2014 Unicode, Inc. All rights reserved.

The files in `data/csur` are based on the information from the ConScript
Unicode Registry maintained by John Cowan and Michael Everson.
