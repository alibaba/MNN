import re
import sys
from spec_tests import get_tests, do_test
from cmark import CMark
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cmark roundtrip tests.')
    parser.add_argument('-p', '--program', dest='program', nargs='?', default=None,
            help='program to test')
    parser.add_argument('-s', '--spec', dest='spec', nargs='?', default='spec.txt',
            help='path to spec')
    parser.add_argument('-P', '--pattern', dest='pattern', nargs='?',
            default=None, help='limit to sections matching regex pattern')
    parser.add_argument('--library-dir', dest='library_dir', nargs='?',
            default=None, help='directory containing dynamic library')
    parser.add_argument('--extensions', dest='extensions', nargs='?',
            default=None, help='space separated list of extensions to enable')
    parser.add_argument('--no-normalize', dest='normalize',
            action='store_const', const=False, default=True,
            help='do not normalize HTML')
    parser.add_argument('-n', '--number', type=int, default=None,
            help='only consider the test with the given number')
    args = parser.parse_args(sys.argv[1:])

spec = sys.argv[1]

def converter(md, exts):
  cmark = CMark(prog=args.program, library_dir=args.library_dir, extensions=args.extensions)
  [ec, result, err] = cmark.to_commonmark(md, exts)
  if ec == 0:
    [ec, html, err] = cmark.to_html(result, exts)
    if ec == 0:
        # In the commonmark writer we insert dummy HTML
        # comments between lists, and between lists and code
        # blocks.  Strip these out, since the spec uses
        # two blank lines instead:
        return [ec, re.sub('<!-- end list -->\n', '', html), '']
    else:
        return [ec, html, err]
  else:
    return [ec, result, err]

tests = get_tests(args.spec)
result_counts = {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0}
for test in tests:
    do_test(converter, test, args.normalize, result_counts)

sys.stdout.buffer.write("{pass} passed, {fail} failed, {error} errored, {skip} skipped\n".format(**result_counts).encode('utf-8'))
exit(result_counts['fail'] + result_counts['error'])
