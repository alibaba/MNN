contents1 = open("test1.txt", "rt").read().split("\n")
contents2 = open("test2.txt", "rt").read().split("\n")
log0 = [content.split(' ') for content in contents1]
log1 = [content.split(' ') for content in contents2]
for layer, ln in enumerate(zip(log0, log1)):
    ln0, ln1 = ln
    for n0, n1 in zip(ln0,ln1):
        if n0 == "" or n1 == "":
            continue
        n = float(n0) - float(n1)
        if n != 0.:
            print(layer, n0, n1, n)