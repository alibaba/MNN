import os, sys, token, tokenize
def do_file(fname):
    source = open(fname)
    mod = open(fname + ".tmp", "w")
    prev_toktype = token.INDENT
    first_line = None
    last_lineno = -1
    last_col = 0
    tokgen = tokenize.generate_tokens(source.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if 0:   # Change to if 1 to see the tokens fly by.
            print("%10s %-14s %-20r %r" % (
                tokenize.tok_name.get(toktype, toktype),
                "%d.%d-%d.%d" % (slineno, scol, elineno, ecol),
                ttext, ltext
                ))
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            mod.write(" " * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            # Docstring
            # mod.write("#--")
            pass
        elif toktype == tokenize.COMMENT:
            # Comment
            # mod.write("##\n")
            pass
        else:
            mod.write(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno
    os.rename(fname + '.tmp', fname)
if __name__ == '__main__':
    src_dir = sys.argv[1] 
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f[-3:] == '.py':
                dir_path = os.path.join(root, f) 
                do_file(dir_path)
