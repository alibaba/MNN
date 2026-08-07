#!/usr/bin/env python3
# Minify embedded Metal shader strings (R"metal(...)metal" blocks) in place.
# Strips // and /* */ comments, blank lines and indentation, collapses repeated spaces.
# Only touches the shader string bodies; all other file content is preserved.
import re
import sys

RAW_RE = re.compile(r'R"metal\((.*?)\)metal"', re.S)


def strip_comments(src: str) -> str:
    out = []
    i, n = 0, len(src)
    state = 'code'  # code | line | block | sq | dq
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ''
        if state == 'code':
            if c == '/' and nxt == '/':
                state = 'line'
                i += 2
                continue
            if c == '/' and nxt == '*':
                state = 'block'
                out.append(' ')  # keep token separation
                i += 2
                continue
            if c == '\'':
                state = 'sq'
            elif c == '"':
                state = 'dq'
            out.append(c)
            i += 1
        elif state == 'line':
            if c == '\n':
                state = 'code'
                out.append('\n')
            i += 1
        elif state == 'block':
            if c == '*' and nxt == '/':
                state = 'code'
                i += 2
            else:
                if c == '\n':
                    out.append('\n')  # preserve line structure
                i += 1
        elif state == 'sq':
            out.append(c)
            if c == '\\':
                i += 1
                if i < n:
                    out.append(src[i])
            elif c == '\'':
                state = 'code'
            i += 1
        else:  # dq
            out.append(c)
            if c == '\\':
                i += 1
                if i < n:
                    out.append(src[i])
            elif c == '"':
                state = 'code'
            i += 1
    return ''.join(out)


def minify(body: str) -> str:
    body = strip_comments(body)
    out = []
    for line in body.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r' {2,}', ' ', line)
        out.append(line)
    return '\n' + '\n'.join(out) + '\n'


def process(path: str) -> None:
    with open(path, 'r') as f:
        text = f.read()

    def sub(m):
        return 'R"metal(' + minify(m.group(1)) + ')metal"'

    new_text = RAW_RE.sub(sub, text)
    if new_text != text:
        with open(path, 'w') as f:
            f.write(new_text)
        print('minified: %s' % path)
    else:
        print('unchanged: %s' % path)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        process(p)
