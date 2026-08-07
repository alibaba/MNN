#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CMARK_NO_SHORT_NAMES
#include <cmark-gfm.h>
#include "node.h"
#include <cmark-gfm-core-extensions.h>

#include "harness.h"
#include "cplusplus.h"

#define UTF8_REPL "\xEF\xBF\xBD"

static const cmark_node_type node_types[] = {
    CMARK_NODE_DOCUMENT,  CMARK_NODE_BLOCK_QUOTE, CMARK_NODE_LIST,
    CMARK_NODE_ITEM,      CMARK_NODE_CODE_BLOCK,  CMARK_NODE_HTML_BLOCK,
    CMARK_NODE_PARAGRAPH, CMARK_NODE_HEADING,     CMARK_NODE_THEMATIC_BREAK,
    CMARK_NODE_TEXT,      CMARK_NODE_SOFTBREAK,   CMARK_NODE_LINEBREAK,
    CMARK_NODE_CODE,      CMARK_NODE_HTML_INLINE, CMARK_NODE_EMPH,
    CMARK_NODE_STRONG,    CMARK_NODE_LINK,        CMARK_NODE_IMAGE};
static const int num_node_types = sizeof(node_types) / sizeof(*node_types);

static void test_md_to_html(test_batch_runner *runner, const char *markdown,
                            const char *expected_html, const char *msg);

static void test_content(test_batch_runner *runner, cmark_node_type type,
                         unsigned int *allowed_content);

static void test_char(test_batch_runner *runner, int valid, const char *utf8,
                      const char *msg);

static void test_incomplete_char(test_batch_runner *runner, const char *utf8,
                                 const char *msg);

static void test_continuation_byte(test_batch_runner *runner, const char *utf8);

static void version(test_batch_runner *runner) {
  INT_EQ(runner, cmark_version(), CMARK_GFM_VERSION, "cmark_version");
  STR_EQ(runner, cmark_version_string(), CMARK_GFM_VERSION_STRING,
         "cmark_version_string");
}

static void constructor(test_batch_runner *runner) {
  for (int i = 0; i < num_node_types; ++i) {
    cmark_node_type type = node_types[i];
    cmark_node *node = cmark_node_new(type);
    OK(runner, node != NULL, "new type %d", type);
    INT_EQ(runner, cmark_node_get_type(node), type, "get_type %d", type);

    switch (node->type) {
    case CMARK_NODE_HEADING:
      INT_EQ(runner, cmark_node_get_heading_level(node), 1,
             "default heading level is 1");
      node->as.heading.level = 1;
      break;

    case CMARK_NODE_LIST:
      INT_EQ(runner, cmark_node_get_list_type(node), CMARK_BULLET_LIST,
             "default is list type is bullet");
      INT_EQ(runner, cmark_node_get_list_delim(node), CMARK_NO_DELIM,
             "default is list delim is NO_DELIM");
      INT_EQ(runner, cmark_node_get_list_start(node), 0,
             "default is list start is 0");
      INT_EQ(runner, cmark_node_get_list_tight(node), 0,
             "default is list is loose");
      break;

    default:
      break;
    }

    cmark_node_free(node);
  }
}

static void accessors(test_batch_runner *runner) {
  static const char markdown[] = "## Header\n"
                                 "\n"
                                 "* Item 1\n"
                                 "* Item 2\n"
                                 "\n"
                                 "2. Item 1\n"
                                 "\n"
                                 "3. Item 2\n"
                                 "\n"
                                 "``` lang\n"
                                 "fenced\n"
                                 "```\n"
                                 "    code\n"
                                 "\n"
                                 "<div>html</div>\n"
                                 "\n"
                                 "[link](url 'title')\n";

  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  // Getters

  cmark_node *heading = cmark_node_first_child(doc);
  INT_EQ(runner, cmark_node_get_heading_level(heading), 2, "get_heading_level");

  cmark_node *bullet_list = cmark_node_next(heading);
  INT_EQ(runner, cmark_node_get_list_type(bullet_list), CMARK_BULLET_LIST,
         "get_list_type bullet");
  INT_EQ(runner, cmark_node_get_list_tight(bullet_list), 1,
         "get_list_tight tight");
  INT_EQ(runner, cmark_node_get_list_marker(bullet_list), CMARK_ASTERISK_LIST_MARKER,
         "get_list_marker asterisk");

  cmark_node *ordered_list = cmark_node_next(bullet_list);
  INT_EQ(runner, cmark_node_get_list_type(ordered_list), CMARK_ORDERED_LIST,
         "get_list_type ordered");
  INT_EQ(runner, cmark_node_get_list_delim(ordered_list), CMARK_PERIOD_DELIM,
         "get_list_delim ordered");
  INT_EQ(runner, cmark_node_get_list_start(ordered_list), 2, "get_list_start");
  INT_EQ(runner, cmark_node_get_list_tight(ordered_list), 0,
         "get_list_tight loose");

  cmark_node *fenced = cmark_node_next(ordered_list);
  STR_EQ(runner, cmark_node_get_literal(fenced), "fenced\n",
         "get_literal fenced code");
  STR_EQ(runner, cmark_node_get_fence_info(fenced), "lang", "get_fence_info");

  cmark_node *code = cmark_node_next(fenced);
  STR_EQ(runner, cmark_node_get_literal(code), "code\n",
         "get_literal indented code");

  cmark_node *html = cmark_node_next(code);
  STR_EQ(runner, cmark_node_get_literal(html), "<div>html</div>\n",
         "get_literal html");

  cmark_node *paragraph = cmark_node_next(html);
  INT_EQ(runner, cmark_node_get_start_line(paragraph), 17, "get_start_line");
  INT_EQ(runner, cmark_node_get_start_column(paragraph), 1, "get_start_column");
  INT_EQ(runner, cmark_node_get_end_line(paragraph), 17, "get_end_line");

  cmark_node *link = cmark_node_first_child(paragraph);
  STR_EQ(runner, cmark_node_get_url(link), "url", "get_url");
  STR_EQ(runner, cmark_node_get_title(link), "title", "get_title");

  cmark_node *string = cmark_node_first_child(link);
  STR_EQ(runner, cmark_node_get_literal(string), "link", "get_literal string");

  // Setters

  OK(runner, cmark_node_set_heading_level(heading, 3), "set_heading_level");

  OK(runner, cmark_node_set_list_marker(bullet_list, CMARK_PLUS_LIST_MARKER), "set_list_marker plus");
  OK(runner, cmark_node_set_list_type(bullet_list, CMARK_ORDERED_LIST),
     "set_list_type ordered");
  OK(runner, cmark_node_set_list_delim(bullet_list, CMARK_PAREN_DELIM),
     "set_list_delim paren");
  OK(runner, cmark_node_set_list_start(bullet_list, 3), "set_list_start");
  OK(runner, cmark_node_set_list_tight(bullet_list, 0), "set_list_tight loose");

  OK(runner, cmark_node_set_list_type(ordered_list, CMARK_BULLET_LIST),
     "set_list_type bullet");
  OK(runner, cmark_node_set_list_tight(ordered_list, 1),
     "set_list_tight tight");

  OK(runner, cmark_node_set_literal(code, "CODE\n"),
     "set_literal indented code");

  OK(runner, cmark_node_set_literal(fenced, "FENCED\n"),
     "set_literal fenced code");
  OK(runner, cmark_node_set_fence_info(fenced, "LANG"), "set_fence_info");

  OK(runner, cmark_node_set_literal(html, "<div>HTML</div>\n"),
     "set_literal html");

  OK(runner, cmark_node_set_url(link, "URL"), "set_url");
  OK(runner, cmark_node_set_title(link, "TITLE"), "set_title");

  OK(runner, cmark_node_set_literal(string, "prefix-LINK"),
     "set_literal string");

  // Set literal to suffix of itself (issue #139).
  const char *literal = cmark_node_get_literal(string);
  OK(runner, cmark_node_set_literal(string, literal + sizeof("prefix")),
     "set_literal suffix");

  char *rendered_html = cmark_render_html(doc, CMARK_OPT_DEFAULT | CMARK_OPT_UNSAFE, NULL);
  static const char expected_html[] =
      "<h3>Header</h3>\n"
      "<ol start=\"3\">\n"
      "<li>\n"
      "<p>Item 1</p>\n"
      "</li>\n"
      "<li>\n"
      "<p>Item 2</p>\n"
      "</li>\n"
      "</ol>\n"
      "<ul>\n"
      "<li>Item 1</li>\n"
      "<li>Item 2</li>\n"
      "</ul>\n"
      "<pre><code class=\"language-LANG\">FENCED\n"
      "</code></pre>\n"
      "<pre><code>CODE\n"
      "</code></pre>\n"
      "<div>HTML</div>\n"
      "<p><a href=\"URL\" title=\"TITLE\">LINK</a></p>\n";
  STR_EQ(runner, rendered_html, expected_html, "setters work");
  free(rendered_html);

  // Getter errors

  INT_EQ(runner, cmark_node_get_heading_level(bullet_list), 0,
         "get_heading_level error");
  INT_EQ(runner, cmark_node_get_list_type(heading), CMARK_NO_LIST,
         "get_list_type error");
  INT_EQ(runner, cmark_node_get_list_start(code), 0, "get_list_start error");
  INT_EQ(runner, cmark_node_get_list_tight(fenced), 0, "get_list_tight error");
  INT_EQ(runner, cmark_node_get_list_marker(heading), CMARK_NO_LIST_MARKER, "get_list_marker error");
  OK(runner, cmark_node_get_literal(ordered_list) == NULL, "get_literal error");
  OK(runner, cmark_node_get_fence_info(paragraph) == NULL,
     "get_fence_info error");
  OK(runner, cmark_node_get_url(html) == NULL, "get_url error");
  OK(runner, cmark_node_get_title(heading) == NULL, "get_title error");

  // Setter errors

  OK(runner, !cmark_node_set_heading_level(bullet_list, 3),
     "set_heading_level error");
  OK(runner, !cmark_node_set_list_type(heading, CMARK_ORDERED_LIST),
     "set_list_type error");
  OK(runner, !cmark_node_set_list_start(code, 3), "set_list_start error");
  OK(runner, !cmark_node_set_list_tight(fenced, 0), "set_list_tight error");
  OK(runner, !cmark_node_set_list_marker(heading, CMARK_PLUS_LIST_MARKER), "set_list_marker error");
  OK(runner, !cmark_node_set_literal(ordered_list, "content\n"),
     "set_literal error");
  OK(runner, !cmark_node_set_fence_info(paragraph, "lang"),
     "set_fence_info error");
  OK(runner, !cmark_node_set_url(html, "url"), "set_url error");
  OK(runner, !cmark_node_set_title(heading, "title"), "set_title error");

  OK(runner, !cmark_node_set_heading_level(heading, 0),
     "set_heading_level too small");
  OK(runner, !cmark_node_set_heading_level(heading, 7),
     "set_heading_level too large");
  OK(runner, !cmark_node_set_list_type(bullet_list, CMARK_NO_LIST),
     "set_list_type invalid");
  OK(runner, !cmark_node_set_list_start(bullet_list, -1),
     "set_list_start negative");
  OK(runner, !cmark_node_set_list_marker(bullet_list, CMARK_NO_LIST_MARKER),
     "set_list_marker invalid");

  cmark_node_free(doc);
}

static void node_check(test_batch_runner *runner) {
  // Construct an incomplete tree.
  cmark_node *doc = cmark_node_new(CMARK_NODE_DOCUMENT);
  cmark_node *p1 = cmark_node_new(CMARK_NODE_PARAGRAPH);
  cmark_node *p2 = cmark_node_new(CMARK_NODE_PARAGRAPH);
  doc->first_child = p1;
  p1->next = p2;

  INT_EQ(runner, cmark_node_check(doc, NULL), 4, "node_check works");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "node_check fixes tree");

  cmark_node_free(doc);
}

static void iterator(test_batch_runner *runner) {
  cmark_node *doc = cmark_parse_document("> a *b*\n\nc", 10, CMARK_OPT_DEFAULT);
  int parnodes = 0;
  cmark_event_type ev_type;
  cmark_iter *iter = cmark_iter_new(doc);
  cmark_node *cur;

  while ((ev_type = cmark_iter_next(iter)) != CMARK_EVENT_DONE) {
    cur = cmark_iter_get_node(iter);
    if (cur->type == CMARK_NODE_PARAGRAPH && ev_type == CMARK_EVENT_ENTER) {
      parnodes += 1;
    }
  }
  INT_EQ(runner, parnodes, 2, "iterate correctly counts paragraphs");

  cmark_iter_free(iter);
  cmark_node_free(doc);
}

static void iterator_delete(test_batch_runner *runner) {
  static const char md[] = "a *b* c\n"
                           "\n"
                           "* item1\n"
                           "* item2\n"
                           "\n"
                           "a `b` c\n"
                           "\n"
                           "* item1\n"
                           "* item2\n";
  cmark_node *doc = cmark_parse_document(md, sizeof(md) - 1, CMARK_OPT_DEFAULT);
  cmark_iter *iter = cmark_iter_new(doc);
  cmark_event_type ev_type;

  while ((ev_type = cmark_iter_next(iter)) != CMARK_EVENT_DONE) {
    cmark_node *node = cmark_iter_get_node(iter);
    // Delete list, emph, and code nodes.
    if ((ev_type == CMARK_EVENT_EXIT && node->type == CMARK_NODE_LIST) ||
        (ev_type == CMARK_EVENT_EXIT && node->type == CMARK_NODE_EMPH) ||
        (ev_type == CMARK_EVENT_ENTER && node->type == CMARK_NODE_CODE)) {
      cmark_node_free(node);
    }
  }

  char *html = cmark_render_html(doc, CMARK_OPT_DEFAULT, NULL);
  static const char expected[] = "<p>a  c</p>\n"
                                 "<p>a  c</p>\n";
  STR_EQ(runner, html, expected, "iterate and delete nodes");

  free(html);
  cmark_iter_free(iter);
  cmark_node_free(doc);
}

static void create_tree(test_batch_runner *runner) {
  char *html;
  cmark_node *doc = cmark_node_new(CMARK_NODE_DOCUMENT);

  cmark_node *p = cmark_node_new(CMARK_NODE_PARAGRAPH);
  OK(runner, !cmark_node_insert_before(doc, p), "insert before root fails");
  OK(runner, !cmark_node_insert_after(doc, p), "insert after root fails");
  OK(runner, cmark_node_append_child(doc, p), "append1");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "append1 consistent");
  OK(runner, cmark_node_parent(p) == doc, "node_parent");

  cmark_node *emph = cmark_node_new(CMARK_NODE_EMPH);
  OK(runner, cmark_node_prepend_child(p, emph), "prepend1");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "prepend1 consistent");

  cmark_node *str1 = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(str1, "Hello, ");
  OK(runner, cmark_node_prepend_child(p, str1), "prepend2");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "prepend2 consistent");

  cmark_node *str3 = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(str3, "!");
  OK(runner, cmark_node_append_child(p, str3), "append2");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "append2 consistent");

  cmark_node *str2 = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(str2, "world");
  OK(runner, cmark_node_append_child(emph, str2), "append3");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "append3 consistent");

  html = cmark_render_html(doc, CMARK_OPT_DEFAULT, NULL);
  STR_EQ(runner, html, "<p>Hello, <em>world</em>!</p>\n", "render_html");
  free(html);

  OK(runner, cmark_node_insert_before(str1, str3), "ins before1");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "ins before1 consistent");
  // 31e
  OK(runner, cmark_node_first_child(p) == str3, "ins before1 works");

  OK(runner, cmark_node_insert_before(str1, emph), "ins before2");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "ins before2 consistent");
  // 3e1
  OK(runner, cmark_node_last_child(p) == str1, "ins before2 works");

  OK(runner, cmark_node_insert_after(str1, str3), "ins after1");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "ins after1 consistent");
  // e13
  OK(runner, cmark_node_next(str1) == str3, "ins after1 works");

  OK(runner, cmark_node_insert_after(str1, emph), "ins after2");
  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "ins after2 consistent");
  // 1e3
  OK(runner, cmark_node_previous(emph) == str1, "ins after2 works");

  cmark_node *str4 = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(str4, "brzz");
  OK(runner, cmark_node_replace(str1, str4), "replace");
  // The replaced node is not freed
  cmark_node_free(str1);

  INT_EQ(runner, cmark_node_check(doc, NULL), 0, "replace consistent");
  OK(runner, cmark_node_previous(emph) == str4, "replace works");
  INT_EQ(runner, cmark_node_replace(p, str4), 0, "replace str for p fails");

  cmark_node_unlink(emph);

  html = cmark_render_html(doc, CMARK_OPT_DEFAULT, NULL);
  STR_EQ(runner, html, "<p>brzz!</p>\n", "render_html after shuffling");
  free(html);

  cmark_node_free(doc);

  // TODO: Test that the contents of an unlinked inline are valid
  // after the parent block was destroyed. This doesn't work so far.
  cmark_node_free(emph);
}

static void custom_nodes(test_batch_runner *runner) {
  char *html;
  char *man;
  cmark_node *doc = cmark_node_new(CMARK_NODE_DOCUMENT);
  cmark_node *p = cmark_node_new(CMARK_NODE_PARAGRAPH);
  cmark_node_append_child(doc, p);
  cmark_node *ci = cmark_node_new(CMARK_NODE_CUSTOM_INLINE);
  cmark_node *str1 = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(str1, "Hello");
  OK(runner, cmark_node_append_child(ci, str1), "append1");
  OK(runner, cmark_node_set_on_enter(ci, "<ON ENTER|"), "set_on_enter");
  OK(runner, cmark_node_set_on_exit(ci, "|ON EXIT>"), "set_on_exit");
  STR_EQ(runner, cmark_node_get_on_enter(ci), "<ON ENTER|", "get_on_enter");
  STR_EQ(runner, cmark_node_get_on_exit(ci), "|ON EXIT>", "get_on_exit");
  cmark_node_append_child(p, ci);
  cmark_node *cb = cmark_node_new(CMARK_NODE_CUSTOM_BLOCK);
  cmark_node_set_on_enter(cb, "<on enter|");
  // leave on_exit unset
  STR_EQ(runner, cmark_node_get_on_exit(cb), "", "get_on_exit (empty)");
  cmark_node_append_child(doc, cb);

  html = cmark_render_html(doc, CMARK_OPT_DEFAULT, NULL);
  STR_EQ(runner, html, "<p><ON ENTER|Hello|ON EXIT></p>\n<on enter|\n",
         "render_html");
  free(html);

  man = cmark_render_man(doc, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, man, ".PP\n<ON ENTER|Hello|ON EXIT>\n<on enter|\n",
         "render_man");
  free(man);

  cmark_node_free(doc);
}

void hierarchy(test_batch_runner *runner) {
  cmark_node *bquote1 = cmark_node_new(CMARK_NODE_BLOCK_QUOTE);
  cmark_node *bquote2 = cmark_node_new(CMARK_NODE_BLOCK_QUOTE);
  cmark_node *bquote3 = cmark_node_new(CMARK_NODE_BLOCK_QUOTE);

  OK(runner, cmark_node_append_child(bquote1, bquote2), "append bquote2");
  OK(runner, cmark_node_append_child(bquote2, bquote3), "append bquote3");
  OK(runner, !cmark_node_append_child(bquote3, bquote3),
     "adding a node as child of itself fails");
  OK(runner, !cmark_node_append_child(bquote3, bquote1),
     "adding a parent as child fails");

  cmark_node_free(bquote1);

  unsigned int list_item_flag[] = {CMARK_NODE_ITEM, 0};
  unsigned int top_level_blocks[] = {
    CMARK_NODE_BLOCK_QUOTE, CMARK_NODE_LIST,
    CMARK_NODE_CODE_BLOCK, CMARK_NODE_HTML_BLOCK,
    CMARK_NODE_PARAGRAPH, CMARK_NODE_HEADING,
    CMARK_NODE_THEMATIC_BREAK, 0};
  unsigned int all_inlines[] = {
    CMARK_NODE_TEXT, CMARK_NODE_SOFTBREAK,
    CMARK_NODE_LINEBREAK, CMARK_NODE_CODE,
    CMARK_NODE_HTML_INLINE, CMARK_NODE_EMPH,
    CMARK_NODE_STRONG, CMARK_NODE_LINK,
    CMARK_NODE_IMAGE, 0};

  test_content(runner, CMARK_NODE_DOCUMENT, top_level_blocks);
  test_content(runner, CMARK_NODE_BLOCK_QUOTE, top_level_blocks);
  test_content(runner, CMARK_NODE_LIST, list_item_flag);
  test_content(runner, CMARK_NODE_ITEM, top_level_blocks);
  test_content(runner, CMARK_NODE_CODE_BLOCK, 0);
  test_content(runner, CMARK_NODE_HTML_BLOCK, 0);
  test_content(runner, CMARK_NODE_PARAGRAPH, all_inlines);
  test_content(runner, CMARK_NODE_HEADING, all_inlines);
  test_content(runner, CMARK_NODE_THEMATIC_BREAK, 0);
  test_content(runner, CMARK_NODE_TEXT, 0);
  test_content(runner, CMARK_NODE_SOFTBREAK, 0);
  test_content(runner, CMARK_NODE_LINEBREAK, 0);
  test_content(runner, CMARK_NODE_CODE, 0);
  test_content(runner, CMARK_NODE_HTML_INLINE, 0);
  test_content(runner, CMARK_NODE_EMPH, all_inlines);
  test_content(runner, CMARK_NODE_STRONG, all_inlines);
  test_content(runner, CMARK_NODE_LINK, all_inlines);
  test_content(runner, CMARK_NODE_IMAGE, all_inlines);
}

static void test_content(test_batch_runner *runner, cmark_node_type type,
                         unsigned int *allowed_content) {
  cmark_node *node = cmark_node_new(type);

  for (int i = 0; i < num_node_types; ++i) {
    cmark_node_type child_type = node_types[i];
    cmark_node *child = cmark_node_new(child_type);

    int got = cmark_node_append_child(node, child);
    int expected = 0;
    if (allowed_content)
        for (unsigned int *p = allowed_content; *p; ++p)
            expected |= *p == (unsigned int)child_type;

    INT_EQ(runner, got, expected, "add %d as child of %d", child_type, type);

    cmark_node_free(child);
  }

  cmark_node_free(node);
}

static void parser(test_batch_runner *runner) {
  test_md_to_html(runner, "No newline", "<p>No newline</p>\n",
                  "document without trailing newline");
}

static void render_html(test_batch_runner *runner) {
  char *html;

  static const char markdown[] = "foo *bar*\n"
                                 "\n"
                                 "paragraph 2\n";
  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  cmark_node *paragraph = cmark_node_first_child(doc);
  html = cmark_render_html(paragraph, CMARK_OPT_DEFAULT, NULL);
  STR_EQ(runner, html, "<p>foo <em>bar</em></p>\n", "render single paragraph");
  free(html);

  cmark_node *string = cmark_node_first_child(paragraph);
  html = cmark_render_html(string, CMARK_OPT_DEFAULT, NULL);
  STR_EQ(runner, html, "foo ", "render single inline");
  free(html);

  cmark_node *emph = cmark_node_next(string);
  html = cmark_render_html(emph, CMARK_OPT_DEFAULT, NULL);
  STR_EQ(runner, html, "<em>bar</em>", "render inline with children");
  free(html);

  cmark_node_free(doc);
}

static void render_xml(test_batch_runner *runner) {
  char *xml;

  static const char markdown[] = "foo *bar*\n"
                                 "\n"
                                 "paragraph 2\n"
                                 "\n"
                                 "```\ncode\n```\n";
  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT);
  STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                      "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                      "<document xmlns=\"http://commonmark.org/xml/1.0\">\n"
                      "  <paragraph>\n"
                      "    <text xml:space=\"preserve\">foo </text>\n"
                      "    <emph>\n"
                      "      <text xml:space=\"preserve\">bar</text>\n"
                      "    </emph>\n"
                      "  </paragraph>\n"
                      "  <paragraph>\n"
                      "    <text xml:space=\"preserve\">paragraph 2</text>\n"
                      "  </paragraph>\n"
                      "  <code_block xml:space=\"preserve\">code\n"
                      "</code_block>\n"
                      "</document>\n",
         "render document");
  free(xml);
  cmark_node *paragraph = cmark_node_first_child(doc);
  xml = cmark_render_xml(paragraph, CMARK_OPT_DEFAULT | CMARK_OPT_SOURCEPOS);
  STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                      "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                      "<paragraph sourcepos=\"1:1-1:9\">\n"
                      "  <text sourcepos=\"1:1-1:4\" xml:space=\"preserve\">foo </text>\n"
                      "  <emph sourcepos=\"1:5-1:9\">\n"
                      "    <text sourcepos=\"1:6-1:8\" xml:space=\"preserve\">bar</text>\n"
                      "  </emph>\n"
                      "</paragraph>\n",
         "render first paragraph with source pos");
  free(xml);
  cmark_node_free(doc);
}

static void render_man(test_batch_runner *runner) {
  char *man;

  static const char markdown[] = "foo *bar*\n"
                                 "\n"
                                 "- Lorem ipsum dolor sit amet,\n"
                                 "  consectetur adipiscing elit,\n"
                                 "- sed do eiusmod tempor incididunt\n"
                                 "  ut labore et dolore magna aliqua.\n";
  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  man = cmark_render_man(doc, CMARK_OPT_DEFAULT, 20);
  STR_EQ(runner, man, ".PP\n"
                      "foo \\f[I]bar\\f[]\n"
                      ".IP \\[bu] 2\n"
                      "Lorem ipsum dolor\n"
                      "sit amet,\n"
                      "consectetur\n"
                      "adipiscing elit,\n"
                      ".IP \\[bu] 2\n"
                      "sed do eiusmod\n"
                      "tempor incididunt ut\n"
                      "labore et dolore\n"
                      "magna aliqua.\n",
         "render document with wrapping");
  free(man);
  man = cmark_render_man(doc, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, man, ".PP\n"
                      "foo \\f[I]bar\\f[]\n"
                      ".IP \\[bu] 2\n"
                      "Lorem ipsum dolor sit amet,\n"
                      "consectetur adipiscing elit,\n"
                      ".IP \\[bu] 2\n"
                      "sed do eiusmod tempor incididunt\n"
                      "ut labore et dolore magna aliqua.\n",
         "render document without wrapping");
  free(man);
  cmark_node_free(doc);
}

static void render_latex(test_batch_runner *runner) {
  char *latex;

  static const char markdown[] = "foo *bar* $%\n"
                                 "\n"
                                 "- Lorem ipsum dolor sit amet,\n"
                                 "  consectetur adipiscing elit,\n"
                                 "- sed do eiusmod tempor incididunt\n"
                                 "  ut labore et dolore magna aliqua.\n";
  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  latex = cmark_render_latex(doc, CMARK_OPT_DEFAULT, 20);
  STR_EQ(runner, latex, "foo \\emph{bar} \\$\\%\n"
                        "\n"
                        "\\begin{itemize}\n"
                        "\\item Lorem ipsum\n"
                        "dolor sit amet,\n"
                        "consectetur\n"
                        "adipiscing elit,\n"
                        "\n"
                        "\\item sed do eiusmod\n"
                        "tempor incididunt ut\n"
                        "labore et dolore\n"
                        "magna aliqua.\n"
                        "\n"
                        "\\end{itemize}\n",
         "render document with wrapping");
  free(latex);
  latex = cmark_render_latex(doc, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, latex, "foo \\emph{bar} \\$\\%\n"
                        "\n"
                        "\\begin{itemize}\n"
                        "\\item Lorem ipsum dolor sit amet,\n"
                        "consectetur adipiscing elit,\n"
                        "\n"
                        "\\item sed do eiusmod tempor incididunt\n"
                        "ut labore et dolore magna aliqua.\n"
                        "\n"
                        "\\end{itemize}\n",
         "render document without wrapping");
  free(latex);
  cmark_node_free(doc);
}

static void render_commonmark(test_batch_runner *runner) {
  char *commonmark;

  static const char markdown[] = "> \\- foo *bar* \\*bar\\*\n"
                                 "\n"
                                 "- Lorem ipsum dolor sit amet,\n"
                                 "  consectetur adipiscing elit,\n"
                                 "- sed do eiusmod tempor incididunt\n"
                                 "  ut labore et dolore magna aliqua.\n";
  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  commonmark = cmark_render_commonmark(doc, CMARK_OPT_DEFAULT, 26);
  STR_EQ(runner, commonmark, "> \\- foo *bar* \\*bar\\*\n"
                             "\n"
                             "  - Lorem ipsum dolor sit\n"
                             "    amet, consectetur\n"
                             "    adipiscing elit,\n"
                             "  - sed do eiusmod tempor\n"
                             "    incididunt ut labore\n"
                             "    et dolore magna\n"
                             "    aliqua.\n",
         "render document with wrapping");
  free(commonmark);
  commonmark = cmark_render_commonmark(doc, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, commonmark, "> \\- foo *bar* \\*bar\\*\n"
                             "\n"
                             "  - Lorem ipsum dolor sit amet,\n"
                             "    consectetur adipiscing elit,\n"
                             "  - sed do eiusmod tempor incididunt\n"
                             "    ut labore et dolore magna aliqua.\n",
         "render document without wrapping");
  free(commonmark);

  cmark_node *text = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(text, "Hi");
  commonmark = cmark_render_commonmark(text, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, commonmark, "Hi\n", "render single inline node");
  free(commonmark);

  cmark_node_free(text);
  cmark_node_free(doc);
}

static void render_plaintext(test_batch_runner *runner) {
  char *plaintext;

  static const char markdown[] = "> \\- foo *bar* \\*bar\\*\n"
                                 "\n"
                                 "- Lorem ipsum dolor sit amet,\n"
                                 "  consectetur adipiscing elit,\n"
                                 "- sed do eiusmod tempor incididunt\n"
                                 "  ut labore et dolore magna aliqua.\n";
  cmark_node *doc =
      cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);

  plaintext = cmark_render_plaintext(doc, CMARK_OPT_DEFAULT, 26);
  STR_EQ(runner, plaintext, "- foo bar *bar*\n"
                             "\n"
                             "  - Lorem ipsum dolor sit\n"
                             "    amet, consectetur\n"
                             "    adipiscing elit,\n"
                             "  - sed do eiusmod tempor\n"
                             "    incididunt ut labore\n"
                             "    et dolore magna\n"
                             "    aliqua.\n",
         "render document with wrapping");
  free(plaintext);
  plaintext = cmark_render_plaintext(doc, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, plaintext, "- foo bar *bar*\n"
                             "\n"
                             "  - Lorem ipsum dolor sit amet,\n"
                             "    consectetur adipiscing elit,\n"
                             "  - sed do eiusmod tempor incididunt\n"
                             "    ut labore et dolore magna aliqua.\n",
         "render document without wrapping");
  free(plaintext);

  cmark_node *text = cmark_node_new(CMARK_NODE_TEXT);
  cmark_node_set_literal(text, "Hi");
  plaintext = cmark_render_plaintext(text, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, plaintext, "Hi\n", "render single inline node");
  free(plaintext);

  cmark_node_free(text);
  cmark_node_free(doc);
}

static void utf8(test_batch_runner *runner) {
  // Ranges
  test_char(runner, 1, "\x01", "valid utf8 01");
  test_char(runner, 1, "\x7F", "valid utf8 7F");
  test_char(runner, 0, "\x80", "invalid utf8 80");
  test_char(runner, 0, "\xBF", "invalid utf8 BF");
  test_char(runner, 0, "\xC0\x80", "invalid utf8 C080");
  test_char(runner, 0, "\xC1\xBF", "invalid utf8 C1BF");
  test_char(runner, 1, "\xC2\x80", "valid utf8 C280");
  test_char(runner, 1, "\xDF\xBF", "valid utf8 DFBF");
  test_char(runner, 0, "\xE0\x80\x80", "invalid utf8 E08080");
  test_char(runner, 0, "\xE0\x9F\xBF", "invalid utf8 E09FBF");
  test_char(runner, 1, "\xE0\xA0\x80", "valid utf8 E0A080");
  test_char(runner, 1, "\xED\x9F\xBF", "valid utf8 ED9FBF");
  test_char(runner, 0, "\xED\xA0\x80", "invalid utf8 EDA080");
  test_char(runner, 0, "\xED\xBF\xBF", "invalid utf8 EDBFBF");
  test_char(runner, 0, "\xF0\x80\x80\x80", "invalid utf8 F0808080");
  test_char(runner, 0, "\xF0\x8F\xBF\xBF", "invalid utf8 F08FBFBF");
  test_char(runner, 1, "\xF0\x90\x80\x80", "valid utf8 F0908080");
  test_char(runner, 1, "\xF4\x8F\xBF\xBF", "valid utf8 F48FBFBF");
  test_char(runner, 0, "\xF4\x90\x80\x80", "invalid utf8 F4908080");
  test_char(runner, 0, "\xF7\xBF\xBF\xBF", "invalid utf8 F7BFBFBF");
  test_char(runner, 0, "\xF8", "invalid utf8 F8");
  test_char(runner, 0, "\xFF", "invalid utf8 FF");

  // Incomplete byte sequences at end of input
  test_incomplete_char(runner, "\xE0\xA0", "invalid utf8 E0A0");
  test_incomplete_char(runner, "\xF0\x90\x80", "invalid utf8 F09080");

  // Invalid continuation bytes
  test_continuation_byte(runner, "\xC2\x80");
  test_continuation_byte(runner, "\xE0\xA0\x80");
  test_continuation_byte(runner, "\xF0\x90\x80\x80");

  // Test string containing null character
  static const char string_with_null[] = "((((\0))))";
  char *html = cmark_markdown_to_html(
      string_with_null, sizeof(string_with_null) - 1, CMARK_OPT_DEFAULT);
  STR_EQ(runner, html, "<p>((((" UTF8_REPL "))))</p>\n", "utf8 with U+0000");
  free(html);

  // Test NUL followed by newline
  static const char string_with_nul_lf[] = "```\n\0\n```\n";
  html = cmark_markdown_to_html(
      string_with_nul_lf, sizeof(string_with_nul_lf) - 1, CMARK_OPT_DEFAULT);
  STR_EQ(runner, html, "<pre><code>\xef\xbf\xbd\n</code></pre>\n",
         "utf8 with \\0\\n");
  free(html);

  // Test byte-order marker
  static const char string_with_bom[] = "\xef\xbb\xbf# Hello\n";
  html = cmark_markdown_to_html(
      string_with_bom, sizeof(string_with_bom) - 1, CMARK_OPT_DEFAULT);
  STR_EQ(runner, html, "<h1>Hello</h1>\n", "utf8 with BOM");
  free(html);
}

static void test_char(test_batch_runner *runner, int valid, const char *utf8,
                      const char *msg) {
  char buf[20];
  sprintf(buf, "((((%s))))", utf8);

  if (valid) {
    char expected[30];
    sprintf(expected, "<p>((((%s))))</p>\n", utf8);
    test_md_to_html(runner, buf, expected, msg);
  } else {
    test_md_to_html(runner, buf, "<p>((((" UTF8_REPL "))))</p>\n", msg);
  }
}

static void test_incomplete_char(test_batch_runner *runner, const char *utf8,
                                 const char *msg) {
  char buf[20];
  sprintf(buf, "----%s", utf8);
  test_md_to_html(runner, buf, "<p>----" UTF8_REPL "</p>\n", msg);
}

static void test_continuation_byte(test_batch_runner *runner,
                                   const char *utf8) {
  size_t len = strlen(utf8);

  for (size_t pos = 1; pos < len; ++pos) {
    char buf[20];
    sprintf(buf, "((((%s))))", utf8);
    buf[4 + pos] = '\x20';

    char expected[50];
    strcpy(expected, "<p>((((" UTF8_REPL "\x20");
    for (size_t i = pos + 1; i < len; ++i) {
      strcat(expected, UTF8_REPL);
    }
    strcat(expected, "))))</p>\n");

    char *html =
        cmark_markdown_to_html(buf, strlen(buf), CMARK_OPT_VALIDATE_UTF8);
    STR_EQ(runner, html, expected, "invalid utf8 continuation byte %zu/%zu", pos,
           len);
    free(html);
  }
}

static void line_endings(test_batch_runner *runner) {
  // Test list with different line endings
  static const char list_with_endings[] = "- a\n- b\r\n- c\r- d";
  char *html = cmark_markdown_to_html(
      list_with_endings, sizeof(list_with_endings) - 1, CMARK_OPT_DEFAULT);
  STR_EQ(runner, html,
         "<ul>\n<li>a</li>\n<li>b</li>\n<li>c</li>\n<li>d</li>\n</ul>\n",
         "list with different line endings");
  free(html);

  static const char crlf_lines[] = "line\r\nline\r\n";
  html = cmark_markdown_to_html(crlf_lines, sizeof(crlf_lines) - 1,
                                CMARK_OPT_DEFAULT | CMARK_OPT_HARDBREAKS);
  STR_EQ(runner, html, "<p>line<br />\nline</p>\n",
         "crlf endings with CMARK_OPT_HARDBREAKS");
  free(html);
  html = cmark_markdown_to_html(crlf_lines, sizeof(crlf_lines) - 1,
                                CMARK_OPT_DEFAULT | CMARK_OPT_NOBREAKS);
  STR_EQ(runner, html, "<p>line line</p>\n",
         "crlf endings with CMARK_OPT_NOBREAKS");
  free(html);

  static const char no_line_ending[] = "```\nline\n```";
  html = cmark_markdown_to_html(no_line_ending, sizeof(no_line_ending) - 1,
                                CMARK_OPT_DEFAULT);
  STR_EQ(runner, html, "<pre><code>line\n</code></pre>\n",
         "fenced code block with no final newline");
  free(html);
}

static void numeric_entities(test_batch_runner *runner) {
  test_md_to_html(runner, "&#0;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 0");
  test_md_to_html(runner, "&#55295;", "<p>\xED\x9F\xBF</p>\n",
                  "Valid numeric entity 0xD7FF");
  test_md_to_html(runner, "&#xD800;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 0xD800");
  test_md_to_html(runner, "&#xDFFF;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 0xDFFF");
  test_md_to_html(runner, "&#57344;", "<p>\xEE\x80\x80</p>\n",
                  "Valid numeric entity 0xE000");
  test_md_to_html(runner, "&#x10FFFF;", "<p>\xF4\x8F\xBF\xBF</p>\n",
                  "Valid numeric entity 0x10FFFF");
  test_md_to_html(runner, "&#x110000;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 0x110000");
  test_md_to_html(runner, "&#x80000000;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 0x80000000");
  test_md_to_html(runner, "&#xFFFFFFFF;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 0xFFFFFFFF");
  test_md_to_html(runner, "&#99999999;", "<p>" UTF8_REPL "</p>\n",
                  "Invalid numeric entity 99999999");

  test_md_to_html(runner, "&#;", "<p>&amp;#;</p>\n",
                  "Min decimal entity length");
  test_md_to_html(runner, "&#x;", "<p>&amp;#x;</p>\n",
                  "Min hexadecimal entity length");
  test_md_to_html(runner, "&#999999999;", "<p>&amp;#999999999;</p>\n",
                  "Max decimal entity length");
  test_md_to_html(runner, "&#x000000041;", "<p>&amp;#x000000041;</p>\n",
                  "Max hexadecimal entity length");
}

static void test_safe(test_batch_runner *runner) {
  // Test safe mode
  static const char raw_html[] = "<div>\nhi\n</div>\n\n<a>hi</"
                                 "a>\n[link](JAVAscript:alert('hi'))\n![image]("
                                 "file:my.js)\n";
  char *html = cmark_markdown_to_html(raw_html, sizeof(raw_html) - 1,
                                      CMARK_OPT_DEFAULT);
  STR_EQ(runner, html, "<!-- raw HTML omitted -->\n<p><!-- raw HTML omitted "
                       "-->hi<!-- raw HTML omitted -->\n<a "
                       "href=\"\">link</a>\n<img src=\"\" alt=\"image\" "
                       "/></p>\n",
         "input with raw HTML and dangerous links");
  free(html);
}

static void test_md_to_html(test_batch_runner *runner, const char *markdown,
                            const char *expected_html, const char *msg) {
  char *html = cmark_markdown_to_html(markdown, strlen(markdown),
                                      CMARK_OPT_VALIDATE_UTF8);
  STR_EQ(runner, html, expected_html, msg);
  free(html);
}

static void test_feed_across_line_ending(test_batch_runner *runner) {
  // See #117
  cmark_parser *parser = cmark_parser_new(CMARK_OPT_DEFAULT);
  cmark_parser_feed(parser, "line1\r", 6);
  cmark_parser_feed(parser, "\nline2\r\n", 8);
  cmark_node *document = cmark_parser_finish(parser);
  OK(runner, document->first_child->next == NULL, "document has one paragraph");
  cmark_parser_free(parser);
  cmark_node_free(document);
}

#if !defined(_WIN32) || defined(__CYGWIN__)
#  include <sys/time.h>
static struct timeval _before, _after;
static int _timing;
#  define START_TIMING() \
       gettimeofday(&_before, NULL)

#  define END_TIMING() \
        do { \
          gettimeofday(&_after, NULL); \
          _timing = (_after.tv_sec - _before.tv_sec) * 1000 + (_after.tv_usec - _before.tv_usec) / 1000; \
        } while (0)

#  define TIMING _timing
#else
#  define START_TIMING()
#  define END_TIMING()
#  define TIMING 0
#endif

static void test_pathological_regressions(test_batch_runner *runner) {
  {
    // I don't care what the output is, so long as it doesn't take too long.
    char path[] = "[a](b";
    char *input = (char *)calloc(1, (sizeof(path) - 1) * 50000);
    for (int i = 0; i < 50000; ++i)
      memcpy(input + i * (sizeof(path) - 1), path, sizeof(path) - 1);

    START_TIMING();
    char *html = cmark_markdown_to_html(input, (sizeof(path) - 1) * 50000,
                                        CMARK_OPT_VALIDATE_UTF8);
    END_TIMING();
    free(html);
    free(input);

    OK(runner, TIMING < 1000, "takes less than 1000ms to run");
  }

  {
    char path[] = "[a](<b";
    char *input = (char *)calloc(1, (sizeof(path) - 1) * 50000);
    for (int i = 0; i < 50000; ++i)
      memcpy(input + i * (sizeof(path) - 1), path, sizeof(path) - 1);

    START_TIMING();
    char *html = cmark_markdown_to_html(input, (sizeof(path) - 1) * 50000,
                                        CMARK_OPT_VALIDATE_UTF8);
    END_TIMING();
    free(html);
    free(input);

    OK(runner, TIMING < 1000, "takes less than 1000ms to run");
  }
}

static void source_pos(test_batch_runner *runner) {
  static const char markdown[] =
    "# Hi *there*.\n"
    "\n"
    "Hello &ldquo; <http://www.google.com>\n"
    "there `hi` -- [okay](www.google.com (ok)).\n"
    "\n"
    "> 1. Okay.\n"
    ">    Sure.\n"
    ">\n"
    "> 2. Yes, okay.\n"
    ">    ![ok](hi \"yes\")\n"
    "<!-- HTML Comment -->\n"
    "\n"
    "what happens if we spread a link [across multiple\n"
    "lines][anchor]\n"
    "\n"
    "[anchor]: http://example.com\n";

  cmark_node *doc = cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);
  char *xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT | CMARK_OPT_SOURCEPOS);
  STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                      "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                      "<document sourcepos=\"1:1-16:28\" xmlns=\"http://commonmark.org/xml/1.0\">\n"
                      "  <heading sourcepos=\"1:1-1:13\" level=\"1\">\n"
                      "    <text sourcepos=\"1:3-1:5\" xml:space=\"preserve\">Hi </text>\n"
                      "    <emph sourcepos=\"1:6-1:12\">\n"
                      "      <text sourcepos=\"1:7-1:11\" xml:space=\"preserve\">there</text>\n"
                      "    </emph>\n"
                      "    <text sourcepos=\"1:13-1:13\" xml:space=\"preserve\">.</text>\n"
                      "  </heading>\n"
                      "  <paragraph sourcepos=\"3:1-4:42\">\n"
                      "    <text sourcepos=\"3:1-3:14\" xml:space=\"preserve\">Hello \xe2\x80\x9c </text>\n"
                      "    <link sourcepos=\"3:15-3:37\" destination=\"http://www.google.com\" title=\"\">\n"
                      "      <text sourcepos=\"3:16-3:36\" xml:space=\"preserve\">http://www.google.com</text>\n"
                      "    </link>\n"
                      "    <softbreak />\n"
                      "    <text sourcepos=\"4:1-4:6\" xml:space=\"preserve\">there </text>\n"
                      "    <code sourcepos=\"4:8-4:9\" xml:space=\"preserve\">hi</code>\n"
                      "    <text sourcepos=\"4:11-4:14\" xml:space=\"preserve\"> -- </text>\n"
                      "    <link sourcepos=\"4:15-4:41\" destination=\"www.google.com\" title=\"ok\">\n"
                      "      <text sourcepos=\"4:16-4:19\" xml:space=\"preserve\">okay</text>\n"
                      "    </link>\n"
                      "    <text sourcepos=\"4:42-4:42\" xml:space=\"preserve\">.</text>\n"
                      "  </paragraph>\n"
                      "  <block_quote sourcepos=\"6:1-10:20\">\n"
                      "    <list sourcepos=\"6:3-10:20\" type=\"ordered\" start=\"1\" delim=\"period\" tight=\"false\">\n"
                      "      <item sourcepos=\"6:3-8:1\">\n"
                      "        <paragraph sourcepos=\"6:6-7:10\">\n"
                      "          <text sourcepos=\"6:6-6:10\" xml:space=\"preserve\">Okay.</text>\n"
                      "          <softbreak />\n"
                      "          <text sourcepos=\"7:6-7:10\" xml:space=\"preserve\">Sure.</text>\n"
                      "        </paragraph>\n"
                      "      </item>\n"
                      "      <item sourcepos=\"9:3-10:20\">\n"
                      "        <paragraph sourcepos=\"9:6-10:20\">\n"
                      "          <text sourcepos=\"9:6-9:15\" xml:space=\"preserve\">Yes, okay.</text>\n"
                      "          <softbreak />\n"
                      "          <image sourcepos=\"10:6-10:20\" destination=\"hi\" title=\"yes\">\n"
                      "            <text sourcepos=\"10:8-10:9\" xml:space=\"preserve\">ok</text>\n"
                      "          </image>\n"
                      "        </paragraph>\n"
                      "      </item>\n"
                      "    </list>\n"
                      "  </block_quote>\n"
                      "  <html_block sourcepos=\"11:1-11:21\" xml:space=\"preserve\">&lt;!-- HTML Comment --&gt;\n"
                      "</html_block>\n"
                      "  <paragraph sourcepos=\"13:1-14:14\">\n"
                      "    <text sourcepos=\"13:1-13:33\" xml:space=\"preserve\">what happens if we spread a link </text>\n"
                      "    <link sourcepos=\"13:34-14:14\" destination=\"http://example.com\" title=\"\">\n"
                      "      <text sourcepos=\"13:35-13:49\" xml:space=\"preserve\">across multiple</text>\n"
                      "      <softbreak />\n"
                      "      <text sourcepos=\"14:1-14:5\" xml:space=\"preserve\">lines</text>\n"
                      "    </link>\n"
                      "  </paragraph>\n"
                      "</document>\n",
         "sourcepos are as expected");
  free(xml);
  cmark_node_free(doc);
}

static void source_pos_inlines(test_batch_runner *runner) {
  {
    static const char markdown[] =
      "*first*\n"
      "second\n"
      "\n"
      "   <http://example.com>";

    cmark_node *doc = cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);
    char *xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT | CMARK_OPT_SOURCEPOS);
    STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                        "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                        "<document sourcepos=\"1:1-4:23\" xmlns=\"http://commonmark.org/xml/1.0\">\n"
                        "  <paragraph sourcepos=\"1:1-2:6\">\n"
                        "    <emph sourcepos=\"1:1-1:7\">\n"
                        "      <text sourcepos=\"1:2-1:6\" xml:space=\"preserve\">first</text>\n"
                        "    </emph>\n"
                        "    <softbreak />\n"
                        "    <text sourcepos=\"2:1-2:6\" xml:space=\"preserve\">second</text>\n"
                        "  </paragraph>\n"
                        "  <paragraph sourcepos=\"4:4-4:23\">\n"
                        "    <link sourcepos=\"4:4-4:23\" destination=\"http://example.com\" title=\"\">\n"
                        "      <text sourcepos=\"4:5-4:22\" xml:space=\"preserve\">http://example.com</text>\n"
                        "    </link>\n"
                        "  </paragraph>\n"
                        "</document>\n",
                        "sourcepos are as expected");
    free(xml);
    cmark_node_free(doc);
  }
  {
    static const char markdown[] =
      "*first\n"
      "second*\n";

    cmark_node *doc = cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);
    char *xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT | CMARK_OPT_SOURCEPOS);
    STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                        "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                        "<document sourcepos=\"1:1-2:7\" xmlns=\"http://commonmark.org/xml/1.0\">\n"
                        "  <paragraph sourcepos=\"1:1-2:7\">\n"
                        "    <emph sourcepos=\"1:1-2:7\">\n"
                        "      <text sourcepos=\"1:2-1:6\" xml:space=\"preserve\">first</text>\n"
                        "      <softbreak />\n"
                        "      <text sourcepos=\"2:1-2:6\" xml:space=\"preserve\">second</text>\n"
                        "    </emph>\n"
                        "  </paragraph>\n"
                        "</document>\n",
                        "sourcepos are as expected");
    free(xml);
    cmark_node_free(doc);
  }
  {
    static const char markdown[] =
      "` It is one backtick\n"
      "`` They are two backticks\n";

    cmark_node *doc = cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);
    char *xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT | CMARK_OPT_SOURCEPOS);
    STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                        "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                        "<document sourcepos=\"1:1-2:25\" xmlns=\"http://commonmark.org/xml/1.0\">\n"
                        "  <paragraph sourcepos=\"1:1-2:25\">\n"
                        "    <text sourcepos=\"1:1-1:20\" xml:space=\"preserve\">` It is one backtick</text>\n"
                        "    <softbreak />\n"
                        "    <text sourcepos=\"2:1-2:25\" xml:space=\"preserve\">`` They are two backticks</text>\n"
                        "  </paragraph>\n"
                        "</document>\n",
                        "sourcepos are as expected");
    free(xml);
    cmark_node_free(doc);
  }
}

static void ref_source_pos(test_batch_runner *runner) {
  static const char markdown[] =
    "Let's try [reference] links.\n"
    "\n"
    "[reference]: https://github.com (GitHub)\n";

  cmark_node *doc = cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_DEFAULT);
  char *xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT | CMARK_OPT_SOURCEPOS);
  STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                      "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
                      "<document sourcepos=\"1:1-3:40\" xmlns=\"http://commonmark.org/xml/1.0\">\n"
                      "  <paragraph sourcepos=\"1:1-1:28\">\n"
                      "    <text sourcepos=\"1:1-1:10\" xml:space=\"preserve\">Let's try </text>\n"
                      "    <link sourcepos=\"1:11-1:21\" destination=\"https://github.com\" title=\"GitHub\">\n"
                      "      <text sourcepos=\"1:12-1:20\" xml:space=\"preserve\">reference</text>\n"
                      "    </link>\n"
                      "    <text sourcepos=\"1:22-1:28\" xml:space=\"preserve\"> links.</text>\n"
                      "  </paragraph>\n"
                      "</document>\n",
         "sourcepos are as expected");
  free(xml);
  cmark_node_free(doc);
}

static void inline_only_opt(test_batch_runner *runner) {
  static const char markdown[] =
    "# My heading\n"
    "> My block quote\n\n"
    "- List item\n\n"
    "[link](https://github.com)\n";
  
  cmark_node *doc = cmark_parse_document(markdown, sizeof(markdown) - 1, CMARK_OPT_INLINE_ONLY);
  char *html = cmark_render_html(doc, CMARK_OPT_DEFAULT, 0);
  STR_EQ(runner, html, "<p># My heading\n"
         "&gt; My block quote\n"
         "\n"
         "- List item\n"
         "\n"
         "<a href=\"https://github.com\">link</a>\n"
         "</p>\n", "html is as expected");
  free(html);
  cmark_node_free(doc);
}

static void check_markdown_plaintext(test_batch_runner *runner, const char *markdown) {
  cmark_node *doc = cmark_parse_document(markdown, strlen(markdown), CMARK_OPT_PRESERVE_WHITESPACE);
  cmark_node *pg = cmark_node_first_child(doc);
  INT_EQ(runner, cmark_node_get_type(pg), CMARK_NODE_PARAGRAPH, "markdown '%s' did not produce a paragraph node", markdown);
  cmark_node *textNode = cmark_node_first_child(pg);
  INT_EQ(runner, cmark_node_get_type(textNode), CMARK_NODE_TEXT, "markdown '%s' did not produce a text node inside the paragraph node", markdown);
  const char *text = cmark_node_get_literal(textNode);
  OK(runner, (text != NULL), "Text literal for '%s' was null", markdown);
  if (text) {
    STR_EQ(runner, text, markdown, "markdown '%s' resulted in '%s'", markdown, text);
  } else {
    SKIP(runner, 1);
  }
  cmark_node_free(doc);
}

static void preserve_whitespace_opt(test_batch_runner *runner) {
  check_markdown_plaintext(runner, " ");
  check_markdown_plaintext(runner, "    ");
  check_markdown_plaintext(runner, "hello");
  check_markdown_plaintext(runner, "hello ");
  check_markdown_plaintext(runner, " hello");
  check_markdown_plaintext(runner, "    hello");
  check_markdown_plaintext(runner, "hello    ");
  check_markdown_plaintext(runner, "hel\nlo");
  check_markdown_plaintext(runner, "hel\n\nlo");
  check_markdown_plaintext(runner, "hel\nworld\nlo");
  check_markdown_plaintext(runner, " hel \n world \n lo ");
  check_markdown_plaintext(runner, "  hello \n  \n world  ");
  check_markdown_plaintext(runner, "\n");
  check_markdown_plaintext(runner, "\n\n\n");
  check_markdown_plaintext(runner, "\nHello");
  check_markdown_plaintext(runner, "Hello\n");
  check_markdown_plaintext(runner, "\nHello\n");
}

static void check_markdown_attributes_node(test_batch_runner *runner, const char *markdown, cmark_node_type expectedType, const char *expectedAttributes) {
  cmark_node *doc = cmark_parse_document(markdown, strlen(markdown), CMARK_OPT_DEFAULT);
  cmark_node *pg = cmark_node_first_child(doc);
  INT_EQ(runner, cmark_node_get_type(pg), CMARK_NODE_PARAGRAPH, "markdown '%s' did not produce a paragraph node", markdown);
  cmark_node *attributeNode = cmark_node_first_child(pg);
  cmark_node_type nodeType = cmark_node_get_type(attributeNode);
  INT_EQ(runner, nodeType, expectedType, "markdown '%s' did not produce the correct node type: got %d, expecting %d", markdown, nodeType, expectedType);
  const char *attributeContent = cmark_node_get_attributes(attributeNode);
  if (attributeContent == NULL) {
    OK(runner, expectedAttributes == NULL, "markdown '%s' produced an unexpected NULL attribute", markdown);
  } else if (expectedAttributes == NULL) {
    OK(runner, attributeContent == NULL, "markdown '%s' produced an unexpected NULL attribute", markdown);
  } else {
    STR_EQ(runner, attributeContent, expectedAttributes, "markdown '%s' did not produce the correct attributes: got %s, expecting: %s", markdown, attributeContent, expectedAttributes);
  }

  cmark_node_free(doc);
}

static void verify_custom_attributes_node(test_batch_runner *runner) {
  // Should produce a TEXT node since there's no `()` to signify attributes
  check_markdown_attributes_node(runner, "^[]", CMARK_NODE_TEXT, NULL);
  check_markdown_attributes_node(runner, "^[](", CMARK_NODE_TEXT, NULL);
  check_markdown_attributes_node(runner, "^[])", CMARK_NODE_TEXT, NULL);
  check_markdown_attributes_node(runner, "^[])(", CMARK_NODE_TEXT, NULL);
  // Should produce an ATTRIBUTE node with no attributes
  check_markdown_attributes_node(runner, "^[]()", CMARK_NODE_ATTRIBUTE, "");
  // Should produce an ATTRIBUTE node with attributes
  check_markdown_attributes_node(runner, "^[](rainbow: 'extreme')", CMARK_NODE_ATTRIBUTE, "rainbow: 'extreme'");
}

static cmark_node* parse_custom_attributues_footnote(test_batch_runner *runner, const char *markdown, cmark_node **retdoc) {
  cmark_node *doc = cmark_parse_document(markdown, strlen(markdown), CMARK_OPT_DEFAULT);
  cmark_node *pg = cmark_node_first_child(doc);
  INT_EQ(runner, cmark_node_get_type(pg), CMARK_NODE_PARAGRAPH, "markdown '%s' did not produce a paragraph node", markdown);
  *retdoc = doc;
  return cmark_node_first_child(pg);
}

static void verify_custom_attributes_footnote_basic(test_batch_runner *runner) {
  static const char markdown[] =
    "^[caffe][1]\n"
    "\n"
    "^[1]: rainbow: 'extreme', colors: { r: 255, g: 0, b: 0 }, corgicopter: true";
  cmark_node *doc;
  cmark_node *attributeNode = parse_custom_attributues_footnote(runner, markdown, &doc);
  INT_EQ(runner, cmark_node_get_type(attributeNode), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(attributeNode),
         "rainbow: 'extreme', colors: { r: 255, g: 0, b: 0 }, corgicopter: true",
         "markdown '%s' did not produce the right attribute in footnote", markdown);

  cmark_node_free(doc);
}

static void verify_custom_attributes_footnote_multiple_footnotes(test_batch_runner *runner) {
  static const char markdown[] =
    "^[food][1] and ^[drinks][2]\n"
    "\n"
    "^[1]: rainbow: 'fun'\n"
    "^[2]: magic: 42";
  cmark_node *doc;
  cmark_node *attributeNode1 = parse_custom_attributues_footnote(runner, markdown, &doc);
  INT_EQ(runner, cmark_node_get_type(attributeNode1), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(attributeNode1), "rainbow: 'fun'", "markdown '%s' did not produce the right attribute in footnote", markdown);
  cmark_node *textNode = cmark_node_next(attributeNode1); // "and"
  cmark_node *attributeNode2 = cmark_node_next(textNode);
  INT_EQ(runner, cmark_node_get_type(attributeNode2), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(attributeNode2), "magic: 42", "markdown '%s' did not produce the right attribute in footnote", markdown);

  cmark_node_free(doc);
}

static void verify_custom_attributes_footnote_reuse(test_batch_runner *runner) {
  static const char markdown[] =
    "^[pizza][1], ^[sandwich][2], ^[ice cream][2], and ^[salad][1]\n"
    "\n"
    "^[1]: has_tomato: true\n"
    "^[2]: price: 12";
  cmark_node *doc;
  cmark_node *pizzaNode = parse_custom_attributues_footnote(runner, markdown, &doc);
  INT_EQ(runner, cmark_node_get_type(pizzaNode), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(pizzaNode), "has_tomato: true", "markdown '%s' did not produce the right attribute in footnote", markdown);
  cmark_node *sandwichNode = cmark_node_next(cmark_node_next(pizzaNode));
  INT_EQ(runner, cmark_node_get_type(sandwichNode), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(sandwichNode), "price: 12", "markdown '%s' did not produce the right attribute in footnote", markdown);
  cmark_node *icecreamNode = cmark_node_next(cmark_node_next(sandwichNode));
  INT_EQ(runner, cmark_node_get_type(icecreamNode), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(icecreamNode), "price: 12", "markdown '%s' did not produce the right attribute in footnote", markdown);
  cmark_node *saladNode = cmark_node_next(cmark_node_next(icecreamNode));
  INT_EQ(runner, cmark_node_get_type(saladNode), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(saladNode), "has_tomato: true", "markdown '%s' did not produce the right attribute in footnote", markdown);

  cmark_node_free(doc);
}

static void verify_custom_attributes_footnote_mixed_content(test_batch_runner *runner) {
  static const char markdown[] =
    "^[attribute1][1], [a link][2], ^[attribute2][3], ^[attribute3][1]\n"
    "\n"
    "^[1]: rainbow: 'fun'\n"
    "[2]: https://www.example.com\n"
    "Lorem ipsum\n"
    "\n"
    "^[3]: universe: 42\n";
  cmark_node *doc;
  cmark_node *attributeNode1 = parse_custom_attributues_footnote(runner, markdown, &doc);
  INT_EQ(runner, cmark_node_get_type(attributeNode1), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(attributeNode1), "rainbow: 'fun'", "markdown '%s' did not produce the right attribute in footnote", markdown);
  cmark_node *linkNode = cmark_node_next(cmark_node_next(attributeNode1));
  INT_EQ(runner, cmark_node_get_type(linkNode), CMARK_NODE_LINK, "markdown '%s' did not produce an link node", markdown);
  STR_EQ(runner, cmark_node_get_url(linkNode), "https://www.example.com", "markdown '%s' did not produce the right link in footnote", markdown);
  cmark_node *attributeNode2 = cmark_node_next(cmark_node_next(linkNode));
  INT_EQ(runner, cmark_node_get_type(attributeNode2), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(attributeNode2), "universe: 42", "markdown '%s' did not produce the right attribute in footnote", markdown);
  cmark_node *attributeNode3 = cmark_node_next(cmark_node_next(attributeNode2));
  INT_EQ(runner, cmark_node_get_type(attributeNode3), CMARK_NODE_ATTRIBUTE, "markdown '%s' did not produce an attribute node", markdown);
  STR_EQ(runner, cmark_node_get_attributes(attributeNode3), "rainbow: 'fun'", "markdown '%s' did not produce the right attribute in footnote", markdown);

  cmark_node_free(doc);
}

static void verify_custom_attributes_node_with_footnote(test_batch_runner *runner) {
  verify_custom_attributes_footnote_basic(runner);
  verify_custom_attributes_footnote_multiple_footnotes(runner);
  verify_custom_attributes_footnote_reuse(runner);
  verify_custom_attributes_footnote_mixed_content(runner);
}

typedef void (*reentrant_call_func) (void);

static cmark_node *reentrant_parse_inline_ext(cmark_syntax_extension *self, cmark_parser *parser,
                                              cmark_node *parent, unsigned char character,
                                              cmark_inline_parser *inline_parser) {
  void *priv = cmark_syntax_extension_get_private(self);
  if (priv) {
    reentrant_call_func func = (reentrant_call_func)priv;
    func();
    cmark_syntax_extension_set_private(self, NULL, NULL);
  }

  return NULL;
}

static void run_inner_parser() {
  cmark_parser *parser = cmark_parser_new(CMARK_OPT_DEFAULT);
  cmark_parser_attach_syntax_extension(parser, cmark_find_syntax_extension("strikethrough"));

  static const char markdown[] = "this is the ~~outer~~ inner document";
  cmark_parser_feed(parser, markdown, sizeof(markdown) - 1);

  cmark_node *doc = cmark_parser_finish(parser);
  cmark_node_free(doc);
  cmark_parser_free(parser);
}

static void parser_interrupt(test_batch_runner *runner) {
  cmark_gfm_core_extensions_ensure_registered();

  cmark_syntax_extension *my_ext = cmark_syntax_extension_new("interrupt");
  cmark_syntax_extension_set_private(my_ext, (void *)&run_inner_parser, NULL);
  cmark_syntax_extension_set_match_inline_func(my_ext, reentrant_parse_inline_ext);

  cmark_parser *parser = cmark_parser_new(CMARK_OPT_DEFAULT);
  cmark_parser_attach_syntax_extension(parser, cmark_find_syntax_extension("strikethrough"));
  cmark_parser_attach_syntax_extension(parser, my_ext);

  static const char markdown[] = "this is the ~~inner~~ outer document";
  cmark_parser_feed(parser, markdown, sizeof(markdown) - 1);

  cmark_node *doc = cmark_parser_finish(parser);
  char *xml = cmark_render_xml(doc, CMARK_OPT_DEFAULT);
  STR_EQ(runner, xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
         "<!DOCTYPE document SYSTEM \"CommonMark.dtd\">\n"
         "<document xmlns=\"http://commonmark.org/xml/1.0\">\n"
         "  <paragraph>\n"
         "    <text xml:space=\"preserve\">this is the </text>\n"
         "    <strikethrough>\n"
         "      <text xml:space=\"preserve\">inner</text>\n"
         "    </strikethrough>\n"
         "    <text xml:space=\"preserve\"> outer document</text>\n"
         "  </paragraph>\n"
         "</document>\n", "interrupting the parser should still allow extensions");

  free(xml);
  cmark_node_free(doc);
  cmark_parser_free(parser);
  cmark_syntax_extension_free(cmark_get_default_mem_allocator(), my_ext);
}

static void compare_table_spans_html(test_batch_runner *runner, const char *markdown, bool use_ditto,
                                     const char *expected_html, const char *msg) {
  int options = CMARK_OPT_TABLE_SPANS;
  if (use_ditto)
    options |= CMARK_OPT_TABLE_ROWSPAN_DITTO;
  cmark_parser *parser = cmark_parser_new(options);
  cmark_parser_attach_syntax_extension(parser, cmark_find_syntax_extension("table"));

  cmark_parser_feed(parser, markdown, strlen(markdown));

  cmark_node *doc = cmark_parser_finish(parser);
  char *html = cmark_render_html(doc, options, NULL);
  STR_EQ(runner, html, expected_html, msg);

  free(html);
  cmark_node_free(doc);
  cmark_parser_free(parser);
}

static void table_spans(test_batch_runner *runner) {
  {
    static const char markdown[] =
      "| one | two |\n"
      "| --- | --- |\n"
      "| hello    ||\n";
    static const char html[] =
      "<table>\n"
      "<thead>\n"
      "<tr>\n"
      "<th>one</th>\n"
      "<th>two</th>\n"
      "</tr>\n"
      "</thead>\n"
      "<tbody>\n"
      "<tr>\n"
      "<td colspan=\"2\">hello</td>\n"
      "</tr>\n"
      "</tbody>\n"
      "</table>\n";
    compare_table_spans_html(runner, markdown, false, html,
                             "table colspans should work when enabled");
  }
  {
    static const char markdown[] =
      "| one | two   |\n"
      "| --- | ----- |\n"
      "| big | small |\n"
      "| ^   | small |\n";
    static const char html[] =
      "<table>\n"
      "<thead>\n"
      "<tr>\n"
      "<th>one</th>\n"
      "<th>two</th>\n"
      "</tr>\n"
      "</thead>\n"
      "<tbody>\n"
      "<tr>\n"
      "<td rowspan=\"2\">big</td>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "<tr>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "</tbody>\n"
      "</table>\n";
    compare_table_spans_html(runner, markdown, false, html,
                             "table rowspans should work when enabled");
  }
  {
    static const char markdown[] =
      "| one | two   |\n"
      "| --- | ----- |\n"
      "| big | small |\n"
      "| \"   | small |\n";
    static const char html[] =
      "<table>\n"
      "<thead>\n"
      "<tr>\n"
      "<th>one</th>\n"
      "<th>two</th>\n"
      "</tr>\n"
      "</thead>\n"
      "<tbody>\n"
      "<tr>\n"
      "<td rowspan=\"2\">big</td>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "<tr>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "</tbody>\n"
      "</table>\n";
    compare_table_spans_html(runner, markdown, true, html,
                             "rowspan ditto marks should work when enabled");
  }
  {
    static const char markdown[] =
      "| one | two | three |\n"
      "| --- | --- | ----- |\n"
      "| big      || small |\n"
      "| ^        || small |\n";
    static const char html[] =
      "<table>\n"
      "<thead>\n"
      "<tr>\n"
      "<th>one</th>\n"
      "<th>two</th>\n"
      "<th>three</th>\n"
      "</tr>\n"
      "</thead>\n"
      "<tbody>\n"
      "<tr>\n"
      "<td colspan=\"2\" rowspan=\"2\">big</td>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "<tr>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "</tbody>\n"
      "</table>\n";
    compare_table_spans_html(runner, markdown, false, html,
                             "colspan and rowspan should combine sensibly");
  }
  {
    static const char markdown[] =
      "| one | two | three |\n"
      "| --- | --- | ----- |\n"
      "| big      || small |\n"
      "| \"        || small |\n";
    static const char html[] =
      "<table>\n"
      "<thead>\n"
      "<tr>\n"
      "<th>one</th>\n"
      "<th>two</th>\n"
      "<th>three</th>\n"
      "</tr>\n"
      "</thead>\n"
      "<tbody>\n"
      "<tr>\n"
      "<td colspan=\"2\" rowspan=\"2\">big</td>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "<tr>\n"
      "<td>small</td>\n"
      "</tr>\n"
      "</tbody>\n"
      "</table>\n";
    compare_table_spans_html(runner, markdown, true, html,
                             "colspan and rowspan should combine when ditto marks are enabled");
  }
}

int main() {
  int retval;
  test_batch_runner *runner = test_batch_runner_new();

  cmark_enable_safety_checks(true);
  version(runner);
  constructor(runner);
  accessors(runner);
  node_check(runner);
  iterator(runner);
  iterator_delete(runner);
  create_tree(runner);
  custom_nodes(runner);
  hierarchy(runner);
  parser(runner);
  render_html(runner);
  render_xml(runner);
  render_man(runner);
  render_latex(runner);
  render_commonmark(runner);
  render_plaintext(runner);
  utf8(runner);
  line_endings(runner);
  numeric_entities(runner);
  test_cplusplus(runner);
  test_safe(runner);
  test_feed_across_line_ending(runner);
  test_pathological_regressions(runner);
  source_pos(runner);
  source_pos_inlines(runner);
  ref_source_pos(runner);
  inline_only_opt(runner);
  preserve_whitespace_opt(runner);
  verify_custom_attributes_node(runner);
  verify_custom_attributes_node_with_footnote(runner);
  parser_interrupt(runner);
  table_spans(runner);

  test_print_summary(runner);
  retval = test_ok(runner) ? 0 : 1;
  free(runner);

  return retval;
}
