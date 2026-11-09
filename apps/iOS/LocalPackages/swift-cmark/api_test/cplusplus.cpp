#include <cstdlib>

#include <cmark-gfm.h>
#include "cplusplus.h"
#include "harness.h"

void
test_cplusplus(test_batch_runner *runner)
{
    static const char md[] = "paragraph\n";
    char *html = cmark_markdown_to_html(md, sizeof(md) - 1, CMARK_OPT_DEFAULT);
    STR_EQ(runner, html, "<p>paragraph</p>\n", "libcmark works with C++");
    free(html);
}

