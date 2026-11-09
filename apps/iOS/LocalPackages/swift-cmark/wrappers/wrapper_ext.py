#!/usr/bin/env python

#
# Example for using the shared library from python.
# Will work with either python 2 or python 3.
# Requires cmark-gfm and cmark-gfm-extensions libraries to be installed.
#
# This particular example uses the GitHub extensions from the gfm-extensions
# library. EXTENSIONS specifies which to use, and the sample shows how to
# connect them into a parser.
#

import sys
import ctypes

if sys.platform == 'darwin':
    libname = 'libcmark-gfm.dylib'
    extname = 'libcmark-gfm-extensions.dylib'
elif sys.platform == 'win32':
    libname = 'cmark-gfm.dll'
    extname = 'cmark-gfm-extensions.dll'
else:
    libname = 'libcmark-gfm.so'
    extname = 'libcmark-gfm-extensions.so'
cmark = ctypes.CDLL(libname)
cmark_ext = ctypes.CDLL(extname)

# Options for the GFM rendering call
OPTS = 0  # defaults

# The GFM extensions that we want to use
EXTENSIONS = (
  'autolink',
  'table',
  'strikethrough',
  'tagfilter',
  )

# Use ctypes to access the functions in libcmark-gfm

F_cmark_parser_new = cmark.cmark_parser_new
F_cmark_parser_new.restype = ctypes.c_void_p
F_cmark_parser_new.argtypes = (ctypes.c_int,)

F_cmark_parser_feed = cmark.cmark_parser_feed
F_cmark_parser_feed.restype = None
F_cmark_parser_feed.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t)

F_cmark_parser_finish = cmark.cmark_parser_finish
F_cmark_parser_finish.restype = ctypes.c_void_p
F_cmark_parser_finish.argtypes = (ctypes.c_void_p,)

F_cmark_parser_attach_syntax_extension = cmark.cmark_parser_attach_syntax_extension
F_cmark_parser_attach_syntax_extension.restype = ctypes.c_int
F_cmark_parser_attach_syntax_extension.argtypes = (ctypes.c_void_p, ctypes.c_void_p)

F_cmark_parser_get_syntax_extensions = cmark.cmark_parser_get_syntax_extensions
F_cmark_parser_get_syntax_extensions.restype = ctypes.c_void_p
F_cmark_parser_get_syntax_extensions.argtypes = (ctypes.c_void_p,)

F_cmark_parser_free = cmark.cmark_parser_free
F_cmark_parser_free.restype = None
F_cmark_parser_free.argtypes = (ctypes.c_void_p,)

F_cmark_node_free = cmark.cmark_node_free
F_cmark_node_free.restype = None
F_cmark_node_free.argtypes = (ctypes.c_void_p,)

F_cmark_find_syntax_extension = cmark.cmark_find_syntax_extension
F_cmark_find_syntax_extension.restype = ctypes.c_void_p
F_cmark_find_syntax_extension.argtypes = (ctypes.c_char_p,)

F_cmark_render_html = cmark.cmark_render_html
F_cmark_render_html.restype = ctypes.c_char_p
F_cmark_render_html.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)


# Set up the libcmark-gfm library and its extensions
F_register = cmark_ext.cmark_gfm_core_extensions_ensure_registered
F_register.restype = None
F_register.argtypes = ( )
F_register()


def md2html(text):
  "Use cmark-gfm to render the Markdown into an HTML fragment."

  parser = F_cmark_parser_new(OPTS)
  assert parser
  for name in EXTENSIONS:
    ext = F_cmark_find_syntax_extension(name)
    assert ext
    rv = F_cmark_parser_attach_syntax_extension(parser, ext)
    assert rv
  exts = F_cmark_parser_get_syntax_extensions(parser)

  F_cmark_parser_feed(parser, text, len(text))
  doc = F_cmark_parser_finish(parser)
  assert doc

  output = F_cmark_render_html(doc, OPTS, exts)

  F_cmark_parser_free(parser)
  F_cmark_node_free(doc)

  return output


sys.stdout.write(md2html(sys.stdin.read()))
