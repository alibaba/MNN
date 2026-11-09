# Benchmarks

Here are some benchmarks, run on an ancient Thinkpad running Intel
Core 2 Duo at 2GHz.  The input text is a 11MB Markdown file built by
concatenating the Markdown sources of all the localizations of the
first edition of
[*Pro Git*](https://github.com/progit/progit/tree/master/en) by Scott
Chacon.

|Implementation     |  Time (sec)|
|-------------------|-----------:|
| Markdown.pl       | 2921.24    |
| Python markdown   |  291.25    |
| PHP markdown      |   20.82    |
| kramdown          |   17.32    |
| cheapskate        |    8.24    |
| peg-markdown      |    5.45    |
| parsedown         |    5.06    |
| **commonmark.js** |    2.09    |
| marked            |    1.99    |
| discount          |    1.85    |
| **cmark**         |    0.29    |
| hoedown           |    0.21    |

To run these benchmarks, use `make bench PROG=/path/to/program`.

`time` is used to measure execution speed.  The reported
time is the *difference* between the time to run the program
with the benchmark input and the time to run it with no input.
(This procedure ensures that implementations in dynamic languages are
not penalized by startup time.) A median of ten runs is taken.  The
process is reniced to a high priority so that the system doesn't
interrupt runs.
