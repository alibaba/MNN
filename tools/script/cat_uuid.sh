strings -a $1 | grep "[[:alnum:]]\{8\}-[[:alnum:]]\{4\}-4[[:alnum:]]\{3\}-[[:alnum:]]\{4\}-[[:alnum:]]\{12\}"
strings -a $1 | grep "[0-9a-f]\{32\}"
