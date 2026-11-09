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
    uint8_t startlen;
    uint8_t openlen;
    uint8_t middlelen;
    uint8_t closelen;
  } fuzz_config;

  if (size >= sizeof(fuzz_config)) {
    /* The beginning of `data` is treated as fuzzer configuration */
    memcpy(&fuzz_config, data, sizeof(fuzz_config));

    /* Test options that are used by GitHub. */
    fuzz_config.options = CMARK_OPT_UNSAFE | CMARK_OPT_FOOTNOTES | CMARK_OPT_GITHUB_PRE_LANG | CMARK_OPT_HARDBREAKS;
    fuzz_config.openlen = fuzz_config.openlen & 0x7;
    fuzz_config.middlelen = fuzz_config.middlelen & 0x7;
    fuzz_config.closelen = fuzz_config.closelen & 0x7;

    /* Remainder of input is the markdown */
    const char *markdown0 = (const char *)(data + sizeof(fuzz_config));
    const size_t markdown_size0 = size - sizeof(fuzz_config);
    char markdown[0x80000];
    if (markdown_size0 <= sizeof(markdown)) {
      size_t markdown_size = 0;
      const size_t componentslen = fuzz_config.startlen + fuzz_config.openlen + fuzz_config.middlelen + fuzz_config.closelen;
      if (componentslen <= markdown_size0) {
        size_t offset = 0;
        const size_t endlen = markdown_size0 - componentslen;
        memcpy(&markdown[markdown_size], &markdown0[offset], fuzz_config.startlen);
        markdown_size += fuzz_config.startlen;
        offset += fuzz_config.startlen;

        if (0 < fuzz_config.openlen) {
          while (markdown_size + fuzz_config.openlen <= sizeof(markdown)/2) {
            memcpy(&markdown[markdown_size], &markdown0[offset],
                   fuzz_config.openlen);
            markdown_size += fuzz_config.openlen;
          }
          offset += fuzz_config.openlen;
        }
        memcpy(&markdown[markdown_size], &markdown0[offset],
               fuzz_config.middlelen);
        markdown_size += fuzz_config.middlelen;
        offset += fuzz_config.middlelen;
        if (0 < fuzz_config.closelen) {
          while (markdown_size + fuzz_config.closelen + endlen <= sizeof(markdown)) {
            memcpy(&markdown[markdown_size], &markdown0[offset],
                   fuzz_config.closelen);
            markdown_size += fuzz_config.closelen;
          }
          offset += fuzz_config.closelen;
        }
        if (markdown_size + endlen <= sizeof(markdown)) {
          memcpy(&markdown[markdown_size], &markdown0[offset],
                 endlen);
          markdown_size += endlen;
        }
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

      cmark_node_free(doc);
      cmark_parser_free(parser);
    }
  }
  return 0;
}
