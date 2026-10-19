cmark-gfm
=========

`cmark-gfm` is an extended version of the C reference implementation of
[CommonMark], a rationalized version of Markdown syntax with a spec.  This
repository adds GitHub Flavored Markdown extensions to
[the upstream implementation], as defined in [the spec].

Changes upstream in `cmark` will be pulled into this `cmark-gfm` project repository
as the upstream project evolves. The original `cmark` repository can be found here:
<https://github.com/commonmark/cmark>.

The rest of the README is preserved as-is from the upstream source.  Note that
the library and binaries produced by this fork are suffixed with `-gfm` in
order to distinguish them from the upstream.

## License

The original `cmark` code is released under a BSD2 license. This same license
applies to the Swift code included in this `cmark-gfm` repository.

---

It provides a shared library (`libcmark`) with functions for parsing
CommonMark documents to an abstract syntax tree (AST), manipulating
the AST, and rendering the document to HTML, groff man, LaTeX,
CommonMark, or an XML representation of the AST.  It also provides a
command-line program (`cmark`) for parsing and rendering CommonMark
documents.

Advantages of this library:

- **Portable.**  The library and program are written in standard
  C99 and have no external dependencies.  They have been tested with
  MSVC, gcc, tcc, and clang.

- **Fast.** cmark can render a Markdown version of *War and Peace* in
  the blink of an eye (127 milliseconds on a ten year old laptop,
  vs. 100-400 milliseconds for an eye blink).  In our [benchmarks],
  cmark is 10,000 times faster than the original `Markdown.pl`, and
  on par with the very fastest available Markdown processors.

- **Accurate.** The library passes all CommonMark conformance tests.

- **Standardized.** The library can be expected to parse CommonMark
  the same way as any other conforming parser.  So, for example,
  you can use `commonmark.js` on the client to preview content that
  will be rendered on the server using `cmark`.

- **Robust.** The library has been extensively fuzz-tested using
  [american fuzzy lop].  The test suite includes pathological cases
  that bring many other Markdown parsers to a crawl (for example,
  thousands-deep nested bracketed text or block quotes).

- **Flexible.** CommonMark input is parsed to an AST which can be
  manipulated programmatically prior to rendering.

- **Multiple renderers.**  Output in HTML, groff man, LaTeX, CommonMark,
  and a custom XML format is supported. And it is easy to write new
  renderers to support other formats.

- **Free.** BSD2-licensed.

It is easy to use `libcmark` in python, lua, ruby, and other dynamic
languages: see the `wrappers/` subdirectory for some simple examples.

There are also libraries that wrap `libcmark` for
[Go](https://github.com/rhinoman/go-commonmark),
[Haskell](https://hackage.haskell.org/package/cmark),
[Ruby](https://github.com/gjtorikian/commonmarker),
[Lua](https://github.com/jgm/cmark-lua),
[Perl](https://metacpan.org/release/CommonMark),
[Python](https://pypi.python.org/pypi/paka.cmark),
[R](https://cran.r-project.org/package=commonmark),
[Tcl](https://github.com/apnadkarni/tcl-cmark),
[Scala](https://github.com/sparsetech/cmark-scala) and
[Node.js](https://github.com/killa123/node-cmark).

Installing
----------

Building the C program (`cmark`) and shared library (`libcmark`)
requires [cmake].  If you modify `scanners.re`, then you will also
need [re2c] \(>= 0.14.2\), which is used to generate `scanners.c` from
`scanners.re`.  We have included a pre-generated `scanners.c` in
the repository to reduce build dependencies.

If you have GNU make, you can simply `make`, `make test`, and `make
install`.  This calls [cmake] to create a `Makefile` in the `build`
directory, then uses that `Makefile` to create the executable and
library.  The binaries can be found in `build/src`.  The default
installation prefix is `/usr/local`.  To change the installation
prefix, pass the `INSTALL_PREFIX` variable if you run `make` for the
first time: `make INSTALL_PREFIX=path`.

For a more portable method, you can use [cmake] manually. [cmake] knows
how to create build environments for many build systems.  For example,
on FreeBSD:

    mkdir build
    cd build
    cmake ..  # optionally: -DCMAKE_INSTALL_PREFIX=path
    make      # executable will be created as build/src/cmark
    make test
    make install

Or, to create Xcode project files on OSX:

    mkdir build
    cd build
    cmake -G Xcode ..
    open cmark.xcodeproj

The GNU Makefile also provides a few other targets for developers.
To run a benchmark:

    make bench

For more detailed benchmarks:

    make newbench

To run a test for memory leaks using `valgrind`:

    make leakcheck

To reformat source code using `clang-format`:

    make format

To run a "fuzz test" against ten long randomly generated inputs:

    make fuzztest

To do a more systematic fuzz test with [american fuzzy lop]:

    AFL_PATH=/path/to/afl_directory make afl

Fuzzing with [libFuzzer] is also supported but, because libFuzzer is still
under active development, may not work with your system-installed version of
clang. Assuming LLVM has been built in `$HOME/src/llvm/build` the fuzzer can be
run with:

    CC="$HOME/src/llvm/build/bin/clang" LIB_FUZZER_PATH="$HOME/src/llvm/lib/Fuzzer/libFuzzer.a" make libFuzzer

To make a release tarball and zip archive:

    make archive

Installing (Windows)
--------------------

To compile with MSVC and NMAKE:

    nmake

You can cross-compile a Windows binary and dll on linux if you have the
`mingw32` compiler:

    make mingw

The binaries will be in `build-mingw/windows/bin`.

Usage
-----

Instructions for the use of the command line program and library can
be found in the man pages in the `man` subdirectory.

Security
--------

By default, the library will scrub raw HTML and potentially
dangerous links (`javascript:`, `vbscript:`, `data:`, `file:`).

To allow these, use the option `CMARK_OPT_UNSAFE` (or
`--unsafe`) with the command line program. If doing so, we
recommend you use a HTML sanitizer specific to your needs to
protect against [XSS
attacks](http://en.wikipedia.org/wiki/Cross-site_scripting).

Contributing
------------

There is a [forum for discussing
CommonMark](http://talk.commonmark.org); you should use it instead of
github issues for questions and possibly open-ended discussions.
Use the [github issue tracker](http://github.com/commonmark/CommonMark/issues)
only for simple, clear, actionable issues.

Authors
-------

John MacFarlane wrote the original library and program.
The block parsing algorithm was worked out together with David
Greenspan. Vicent Marti optimized the C implementation for
performance, increasing its speed tenfold.  Kārlis Gaņģis helped
work out a better parsing algorithm for links and emphasis,
eliminating several worst-case performance issues.
Nick Wellnhofer contributed many improvements, including
most of the C library's API and its test harness.

[benchmarks]: benchmarks.md
[the spec]: https://github.github.com/gfm/
[the upstream implementation]: https://github.com/jgm/cmark
[CommonMark]: http://commonmark.org
[cmake]: http://www.cmake.org/download/
[re2c]: http://re2c.org
[commonmark.js]: https://github.com/commonmark/commonmark.js
[Build Status]: https://img.shields.io/travis/github/cmark-gfm/master.svg?style=flat
[Windows Build Status]: https://ci.appveyor.com/api/projects/status/wv7ifhqhv5itm3d5?svg=true
[american fuzzy lop]: http://lcamtuf.coredump.cx/afl/
[libFuzzer]: http://llvm.org/docs/LibFuzzer.html
