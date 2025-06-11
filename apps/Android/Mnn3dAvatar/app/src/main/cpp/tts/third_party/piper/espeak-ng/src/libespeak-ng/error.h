/* Internal error APIs.
 *
 * Copyright (C) 2016 Reece H. Dunn
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see: <http://www.gnu.org/licenses/>.
 */
#ifndef ESPEAK_NG_ERROR_API
#define ESPEAK_NG_ERROR_API

#include <espeak-ng/espeak_ng.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
	ERROR_CONTEXT_FILE,
	ERROR_CONTEXT_VERSION,
} espeak_ng_CONTEXT_TYPE;

typedef struct espeak_ng_ERROR_CONTEXT_
{
	espeak_ng_CONTEXT_TYPE type;
	char *name;
	int version;
	int expected_version;
} espeak_ng_ERROR_CONTEXT_;

espeak_ng_STATUS
create_file_error_context(espeak_ng_ERROR_CONTEXT *context,
                          espeak_ng_STATUS status,
                          const char *filename);

espeak_ng_STATUS
create_version_mismatch_error_context(espeak_ng_ERROR_CONTEXT *context,
                                      const char *path,
                                      int version,
                                      int expected_version);

#ifdef __cplusplus
}
#endif

#endif
