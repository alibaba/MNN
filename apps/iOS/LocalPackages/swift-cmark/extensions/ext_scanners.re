/*!re2c re2c:flags:no-debug-info = 1; */
/*!re2c re2c:indent:string = '  '; */

#include <stdlib.h>
#include "ext_scanners.h"

bufsize_t _ext_scan_at(bufsize_t (*scanner)(const unsigned char *), unsigned char *ptr, int len, bufsize_t offset)
{
	bufsize_t res;

        if (ptr == NULL || offset >= len) {
          return 0;
        } else {
	  unsigned char lim = ptr[len];

	  ptr[len] = '\0';
	  res = scanner(ptr + offset);
	  ptr[len] = lim;
        }

	return res;
}

/*!re2c
  re2c:define:YYCTYPE  = "unsigned char";
  re2c:define:YYCURSOR = p;
  re2c:define:YYMARKER = marker;
  re2c:yyfill:enable = 0;

  spacechar = [ \t\v\f];
  newline = [\r]?[\n];
  escaped_char = [\\][|!"#$%&'()*+,./:;<=>?@[\\\]^_`{}~-];

  table_marker = (spacechar*[:]?[-]+[:]?spacechar*);
  table_cell = (escaped_char|[^|\r\n])+;

  tasklist = spacechar*("-"|"+"|"*"|[0-9]+.)spacechar+("[ ]"|"[x]")spacechar+;
*/

bufsize_t _scan_table_start(const unsigned char *p)
{
  const unsigned char *marker = NULL;
  const unsigned char *start = p;
  /*!re2c
    [|]? table_marker ([|] table_marker)* [|]? spacechar* newline {
      return (bufsize_t)(p - start);
    }
    * { return 0; }
  */
}

bufsize_t _scan_table_cell(const unsigned char *p)
{
  const unsigned char *marker = NULL;
  const unsigned char *start = p;
  /*!re2c
    // In fact, `table_cell` matches non-empty table cells only. The empty
    // string is also a valid table cell, but is handled by the default rule.
    // This approach prevents re2c's match-empty-string warning.
    table_cell { return (bufsize_t)(p - start); }
    * { return 0; }
  */
}

bufsize_t _scan_table_cell_end(const unsigned char *p)
{
  const unsigned char *start = p;
  /*!re2c
    [|] spacechar* { return (bufsize_t)(p - start); }
    * { return 0; }
  */
}

bufsize_t _scan_table_row_end(const unsigned char *p)
{
  const unsigned char *marker = NULL;
  const unsigned char *start = p;
  /*!re2c
    spacechar* newline { return (bufsize_t)(p - start); }
    * { return 0; }
  */
}

bufsize_t _scan_tasklist(const unsigned char *p)
{
  const unsigned char *marker = NULL;
  const unsigned char *start = p;
  /*!re2c
    tasklist { return (bufsize_t)(p - start); }
    * { return 0; }
  */
}
