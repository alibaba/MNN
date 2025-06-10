/* Error handling APIs.
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

#include "config.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>

#include "error.h"
#include "common.h"           // for strncpy0

espeak_ng_STATUS
create_file_error_context(espeak_ng_ERROR_CONTEXT *context,
                          espeak_ng_STATUS status,
                          const char *filename)
{
	if (context) {
		if (*context) {
			free((*context)->name);
		} else {
			*context = malloc(sizeof(espeak_ng_ERROR_CONTEXT_));
			if (!*context)
				return ENOMEM;
		}
		(*context)->type = ERROR_CONTEXT_FILE;
		(*context)->name = strdup(filename);
		(*context)->version = 0;
		(*context)->expected_version = 0;
	}
	return status;
}

espeak_ng_STATUS
create_version_mismatch_error_context(espeak_ng_ERROR_CONTEXT *context,
                                      const char *path_home,
                                      int version,
                                      int expected_version)
{
	if (context) {
		if (*context) {
			free((*context)->name);
		} else {
			*context = malloc(sizeof(espeak_ng_ERROR_CONTEXT_));
			if (!*context)
				return ENOMEM;
		}
		(*context)->type = ERROR_CONTEXT_VERSION;
		(*context)->name = strdup(path_home);
		(*context)->version = version;
		(*context)->expected_version = expected_version;
	}
	return ENS_VERSION_MISMATCH;
}

#pragma GCC visibility push(default)

ESPEAK_NG_API void
espeak_ng_ClearErrorContext(espeak_ng_ERROR_CONTEXT *context)
{
	if (context && *context) {
		free((*context)->name);
		free(*context);
		*context = NULL;
	}
}

ESPEAK_NG_API void
espeak_ng_GetStatusCodeMessage(espeak_ng_STATUS status,
                               char *buffer,
                               size_t length)
{
	switch (status)
	{
	case ENS_COMPILE_ERROR:
		strncpy0(buffer, "Compile error", length);
		break;
	case ENS_VERSION_MISMATCH:
		strncpy0(buffer, "Wrong version of espeak-ng-data", length);
		break;
	case ENS_FIFO_BUFFER_FULL:
		strncpy0(buffer, "The FIFO buffer is full", length);
		break;
	case ENS_NOT_INITIALIZED:
		strncpy0(buffer, "The espeak-ng library has not been initialized", length);
		break;
	case ENS_AUDIO_ERROR:
		strncpy0(buffer, "Cannot initialize the audio device", length);
		break;
	case ENS_VOICE_NOT_FOUND:
		strncpy0(buffer, "The specified espeak-ng voice does not exist", length);
		break;
	case ENS_MBROLA_NOT_FOUND:
		strncpy0(buffer, "Could not load the mbrola.dll file", length);
		break;
	case ENS_MBROLA_VOICE_NOT_FOUND:
		strncpy0(buffer, "Could not load the specified mbrola voice file", length);
		break;
	case ENS_EVENT_BUFFER_FULL:
		strncpy0(buffer, "The event buffer is full", length);
		break;
	case ENS_NOT_SUPPORTED:
		strncpy0(buffer, "The requested functionality has not been built into espeak-ng", length);
		break;
	case ENS_UNSUPPORTED_PHON_FORMAT:
		strncpy0(buffer, "The phoneme file is not in a supported format", length);
		break;
	case ENS_NO_SPECT_FRAMES:
		strncpy0(buffer, "The spectral file does not contain any frame data", length);
		break;
	case ENS_EMPTY_PHONEME_MANIFEST:
		strncpy0(buffer, "The phoneme manifest file does not contain any phonemes", length);
		break;
	case ENS_UNKNOWN_PHONEME_FEATURE:
		strncpy0(buffer, "The phoneme feature is not recognised", length);
		break;
	case ENS_UNKNOWN_TEXT_ENCODING:
		strncpy0(buffer, "The text encoding is not supported", length);
		break;
	default:
		if ((status & ENS_GROUP_MASK) == ENS_GROUP_ERRNO)
			strerror_r(status, buffer, length);
		else
			snprintf(buffer, length, "Unspecified error 0x%x", status);
		break;
	}
}

ESPEAK_NG_API void
espeak_ng_PrintStatusCodeMessage(espeak_ng_STATUS status,
                                 FILE *out,
                                 espeak_ng_ERROR_CONTEXT context)
{
	char error[512];
	espeak_ng_GetStatusCodeMessage(status, error, sizeof(error));
	if (context) {
		switch (context->type)
		{
		case ERROR_CONTEXT_FILE:
			fprintf(out, "Error processing file '%s': %s.\n", context->name, error);
			break;
		case ERROR_CONTEXT_VERSION:
			fprintf(out, "Error: %s at '%s' (expected 0x%x, got 0x%x).\n",
			        error, context->name, context->expected_version, context->version);
			break;
		}
	} else
		fprintf(out, "Error: %s.\n", error);
}

#pragma GCC visibility pop
