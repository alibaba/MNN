#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ctypes import CDLL, c_char_p, c_size_t, c_int, c_void_p
from subprocess import *
import platform
import os

def pipe_through_prog(prog, text):
    p1 = Popen(prog.split(), stdout=PIPE, stdin=PIPE, stderr=PIPE)
    [result, err] = p1.communicate(input=text.encode('utf-8'))
    return [p1.returncode, result.decode('utf-8'), err]

def parse(lib, extlib, text, extensions):
    cmark_gfm_core_extensions_ensure_registered = extlib.cmark_gfm_core_extensions_ensure_registered

    find_syntax_extension = lib.cmark_find_syntax_extension
    find_syntax_extension.restype = c_void_p
    find_syntax_extension.argtypes = [c_char_p]

    parser_attach_syntax_extension = lib.cmark_parser_attach_syntax_extension
    parser_attach_syntax_extension.argtypes = [c_void_p, c_void_p]

    parser_new = lib.cmark_parser_new
    parser_new.restype = c_void_p
    parser_new.argtypes = [c_int]
    
    parser_feed = lib.cmark_parser_feed
    parser_feed.argtypes = [c_void_p, c_char_p, c_int]

    parser_finish = lib.cmark_parser_finish
    parser_finish.restype = c_void_p
    parser_finish.argtypes = [c_void_p]

    cmark_gfm_core_extensions_ensure_registered()

    parser = parser_new(0)
    for e in set(extensions):
        ext = find_syntax_extension(bytes(e, 'utf-8'))
        if not ext:
            raise Exception("Extension not found: '{}'".format(e))
        parser_attach_syntax_extension(parser, ext)

    textbytes = text.encode('utf-8')
    textlen = len(textbytes)
    parser_feed(parser, textbytes, textlen)

    return [parser_finish(parser), parser]

def to_html(lib, extlib, text, extensions):
    document, parser = parse(lib, extlib, text, extensions)
    parser_get_syntax_extensions = lib.cmark_parser_get_syntax_extensions
    parser_get_syntax_extensions.restype = c_void_p
    parser_get_syntax_extensions.argtypes = [c_void_p]
    syntax_extensions = parser_get_syntax_extensions(parser)

    render_html = lib.cmark_render_html
    render_html.restype = c_char_p
    render_html.argtypes = [c_void_p, c_int, c_void_p]
    # 1 << 17 == CMARK_OPT_UNSAFE
    result = render_html(document, 1 << 17, syntax_extensions).decode('utf-8')
    return [0, result, '']

def to_commonmark(lib, extlib, text, extensions):
    document, _ = parse(lib, extlib, text, extensions)

    render_commonmark = lib.cmark_render_commonmark
    render_commonmark.restype = c_char_p
    render_commonmark.argtypes = [c_void_p, c_int, c_int]
    result = render_commonmark(document, 0, 0).decode('utf-8')
    return [0, result, '']

class CMark:
    def __init__(self, prog=None, library_dir=None, extensions=None):
        self.prog = prog
        self.extensions = []
        if extensions:
            self.extensions = extensions.split()

        if prog:
            prog += ' --unsafe'
            extsfun = lambda exts: ''.join([' -e ' + e for e in set(exts)])
            self.to_html = lambda x, exts=[]: pipe_through_prog(prog + extsfun(exts + self.extensions), x)
            self.to_commonmark = lambda x, exts=[]: pipe_through_prog(prog + ' -t commonmark' + extsfun(exts + self.extensions), x)
        else:
            sysname = platform.system()
            if sysname == 'Darwin':
                libnames = [ ["lib", ".dylib" ] ]
            elif sysname == 'Windows':
                libnames = [ ["", ".dll"], ["lib", ".dll"] ]
            else:
                libnames = [ ["lib", ".so"] ]
            if not library_dir:
                library_dir = os.path.join("..", "build", "src")
            for prefix, suffix in libnames:
                candidate = os.path.join(library_dir, prefix + "cmark-gfm" + suffix)
                if os.path.isfile(candidate):
                    libpath = candidate
                    break
            cmark = CDLL(libpath)
            extlib = CDLL(os.path.join(
                library_dir, "..", "extensions", prefix + "cmark-gfm-extensions" + suffix))
            self.to_html = lambda x, exts=[]: to_html(cmark, extlib, x, exts + self.extensions)
            self.to_commonmark = lambda x, exts=[]: to_commonmark(cmark, extlib, x, exts + self.extensions)

