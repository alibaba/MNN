#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "cmark-gfm.h"
#include "cmark-gfm-core-extensions.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

const char *extension_names[] = {
  "autolink",
  "strikethrough",
  "table",
  "tagfilter",
  NULL,
};

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  cmark_gfm_core_extensions_ensure_registered();
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  struct __attribute__((packed)) {
    int options;
    int width;
    uint8_t splitpoint;
    uint8_t repeatlen;
  } fuzz_config;

  if (size >= sizeof(fuzz_config)) {
    /* The beginning of `data` is treated as fuzzer configuration */
    memcpy(&fuzz_config, data, sizeof(fuzz_config));

    /* Test options that are used by GitHub. */
    fuzz_config.options = CMARK_OPT_UNSAFE | CMARK_OPT_FOOTNOTES | CMARK_OPT_GITHUB_PRE_LANG | CMARK_OPT_HARDBREAKS;

    /* Remainder of input is the markdown */
    const char *markdown0 = (const char *)(data + sizeof(fuzz_config));
    const size_t markdown_size0 = size - sizeof(fuzz_config);
    char markdown[0x80000];
    if (markdown_size0 <= sizeof(markdown)) {
      size_t markdown_size = 0;
      if (fuzz_config.splitpoint <= markdown_size0 && 0 < fuzz_config.repeatlen &&
          fuzz_config.repeatlen <= markdown_size0 - fuzz_config.splitpoint) {
        const size_t size_after_splitpoint = markdown_size0 - fuzz_config.splitpoint - fuzz_config.repeatlen;
        memcpy(&markdown[markdown_size], &markdown0[0], fuzz_config.splitpoint);
        markdown_size += fuzz_config.splitpoint;

        while (markdown_size + fuzz_config.repeatlen + size_after_splitpoint <= sizeof(markdown)) {
          memcpy(&markdown[markdown_size], &markdown0[fuzz_config.splitpoint],
                 fuzz_config.repeatlen);
          markdown_size += fuzz_config.repeatlen;
        }
        memcpy(&markdown[markdown_size], &markdown0[fuzz_config.splitpoint + fuzz_config.repeatlen],
               size_after_splitpoint);
        markdown_size += size_after_splitpoint;
      } else {
        markdown_size = markdown_size0;
        memcpy(markdown, markdown0, markdown_size);
      }

      cmark_parser *parser = cmark_parser_new(fuzz_config.options);

      for (const char **it = extension_names; *it; ++it) {
        const char *extension_name = *it;
        cmark_syntax_extension *syntax_extension = cmark_find_syntax_extension(extension_name);
        if (!syntax_extension) {
          fprintf(stderr, "%s is not a valid syntax extension\n", extension_name);
          abort();
        }
        cmark_parser_attach_syntax_extension(parser, syntax_extension);
      }

      cmark_parser_feed(parser, markdown, markdown_size);
      cmark_node *doc = cmark_parser_finish(parser);

      free(cmark_render_html(doc, fuzz_config.options, NULL));
      free(cmark_render_xml(doc, fuzz_config.options));
      free(cmark_render_man(doc, fuzz_config.options, 80));
      free(cmark_render_commonmark(doc, fuzz_config.options, 80));
      free(cmark_render_plaintext(doc, fuzz_config.options, 80));
      free(cmark_render_latex(doc, fuzz_config.options, 80));

      cmark_node_free(doc);
      cmark_parser_free(parser);
    }
  }
  return 0;
}
