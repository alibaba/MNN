#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from difflib import unified_diff
import argparse
import re
import json
from cmark import CMark
from normalize import normalize_html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cmark tests.')
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
    parser.add_argument('-d', '--dump-tests', dest='dump_tests',
            action='store_const', const=True, default=False,
            help='dump tests in JSON format')
    parser.add_argument('--debug-normalization', dest='debug_normalization',
            action='store_const', const=True,
            default=False, help='filter stdin through normalizer for testing')
    parser.add_argument('-n', '--number', type=int, default=None,
            help='only consider the test with the given number')
    args = parser.parse_args(sys.argv[1:])

def out(str):
    sys.stdout.buffer.write(str.encode('utf-8')) 

def print_test_header(headertext, example_number, start_line, end_line):
    out("Example %d (lines %d-%d) %s\n" % (example_number,start_line,end_line,headertext))

def do_test(converter, test, normalize, result_counts):
    [retcode, actual_html, err] = converter(test['markdown'], test['extensions'])
    actual_html = re.sub(r'\r\n', '\n', actual_html)
    if retcode == 0:
        expected_html = re.sub(r'\r\n', '\n', test['html'])
        unicode_error = None
        if expected_html.strip() == '<IGNORE>':
            passed = True
        elif normalize:
            try:
                passed = normalize_html(actual_html) == normalize_html(expected_html)
            except UnicodeDecodeError as e:
                unicode_error = e
                passed = False
        else:
            passed = actual_html == expected_html
        if passed:
            result_counts['pass'] += 1
        else:
            print_test_header(test['section'], test['example'], test['start_line'], test['end_line'])
            out(test['markdown'] + '\n')
            if unicode_error:
                out("Unicode error: " + str(unicode_error) + '\n')
                out("Expected: " + repr(expected_html) + '\n')
                out("Got:      " + repr(actual_html) + '\n')
            else:
                expected_html_lines = expected_html.splitlines(True)
                actual_html_lines = actual_html.splitlines(True)
                for diffline in unified_diff(expected_html_lines, actual_html_lines,
                                "expected HTML", "actual HTML"):
                    out(diffline)
            out('\n')
            result_counts['fail'] += 1
    else:
        print_test_header(test['section'], test['example'], test['start_line'], test['end_line'])
        out("program returned error code %d\n" % retcode)
        sys.stdout.buffer.write(err)
        result_counts['error'] += 1

def get_tests(specfile):
    line_number = 0
    start_line = 0
    end_line = 0
    example_number = 0
    markdown_lines = []
    html_lines = []
    state = 0  # 0 regular text, 1 markdown example, 2 html output
    extensions = []
    headertext = ''
    tests = []

    header_re = re.compile('#+ ')

    with open(specfile, 'r', encoding='utf-8', newline='\n') as specf:
        for line in specf:
            line_number = line_number + 1
            l = line.strip()
            if l.startswith("`" * 32 + " example"):
                state = 1
                extensions = l[32 + len(" example"):].split()
            elif l == "`" * 32:
                state = 0
                example_number = example_number + 1
                end_line = line_number
                if 'disabled' not in extensions:
                    tests.append({
                        "markdown":''.join(markdown_lines).replace('→',"\t"),
                        "html":''.join(html_lines).replace('→',"\t"),
                        "example": example_number,
                        "start_line": start_line,
                        "end_line": end_line,
                        "section": headertext,
                        "extensions": extensions})
                start_line = 0
                markdown_lines = []
                html_lines = []
            elif l == ".":
                state = 2
            elif state == 1:
                if start_line == 0:
                    start_line = line_number - 1
                markdown_lines.append(line)
            elif state == 2:
                html_lines.append(line)
            elif state == 0 and re.match(header_re, line):
                headertext = header_re.sub('', line).strip()
    return tests

if __name__ == "__main__":
    if args.debug_normalization:
        out(normalize_html(sys.stdin.read()))
        exit(0)

    all_tests = get_tests(args.spec)
    if args.pattern:
        pattern_re = re.compile(args.pattern, re.IGNORECASE)
    else:
        pattern_re = re.compile('.')
    tests = [ test for test in all_tests if re.search(pattern_re, test['section']) and (not args.number or test['example'] == args.number) ]
    if args.dump_tests:
        out(json.dumps(tests, indent=2))
        exit(0)
    else:
        skipped = len(all_tests) - len(tests)
        converter = CMark(prog=args.program, library_dir=args.library_dir, extensions=args.extensions).to_html
        result_counts = {'pass': 0, 'fail': 0, 'error': 0, 'skip': skipped}
        for test in tests:
            do_test(converter, test, args.normalize, result_counts)
        out("{pass} passed, {fail} failed, {error} errored, {skip} skipped\n".format(**result_counts))
        exit(result_counts['fail'] + result_counts['error'])
