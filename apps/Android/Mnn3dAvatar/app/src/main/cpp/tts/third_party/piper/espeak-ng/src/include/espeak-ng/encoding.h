/*
 * Copyright (C) 2017 Reece H. Dunn
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
#ifndef ESPEAK_NG_ENCODING_H
#define ESPEAK_NG_ENCODING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
	ESPEAKNG_ENCODING_UNKNOWN,
	ESPEAKNG_ENCODING_US_ASCII,
	ESPEAKNG_ENCODING_ISO_8859_1,
	ESPEAKNG_ENCODING_ISO_8859_2,
	ESPEAKNG_ENCODING_ISO_8859_3,
	ESPEAKNG_ENCODING_ISO_8859_4,
	ESPEAKNG_ENCODING_ISO_8859_5,
	ESPEAKNG_ENCODING_ISO_8859_6,
	ESPEAKNG_ENCODING_ISO_8859_7,
	ESPEAKNG_ENCODING_ISO_8859_8,
	ESPEAKNG_ENCODING_ISO_8859_9,
	ESPEAKNG_ENCODING_ISO_8859_10,
	ESPEAKNG_ENCODING_ISO_8859_11,
	// ISO-8859-12 is not a valid encoding.
	ESPEAKNG_ENCODING_ISO_8859_13,
	ESPEAKNG_ENCODING_ISO_8859_14,
	ESPEAKNG_ENCODING_ISO_8859_15,
	ESPEAKNG_ENCODING_ISO_8859_16,
	ESPEAKNG_ENCODING_KOI8_R,
	ESPEAKNG_ENCODING_ISCII,
	ESPEAKNG_ENCODING_UTF_8,
	ESPEAKNG_ENCODING_ISO_10646_UCS_2,
} espeak_ng_ENCODING;

ESPEAK_NG_API espeak_ng_ENCODING
espeak_ng_EncodingFromName(const char *encoding);

typedef struct espeak_ng_TEXT_DECODER_ espeak_ng_TEXT_DECODER;

ESPEAK_NG_API espeak_ng_TEXT_DECODER *
create_text_decoder(void);

ESPEAK_NG_API void
destroy_text_decoder(espeak_ng_TEXT_DECODER *decoder);

ESPEAK_NG_API espeak_ng_STATUS
text_decoder_decode_string(espeak_ng_TEXT_DECODER *decoder,
                           const char *string,
                           int length,
                           espeak_ng_ENCODING encoding);

ESPEAK_NG_API espeak_ng_STATUS
text_decoder_decode_string_auto(espeak_ng_TEXT_DECODER *decoder,
                                const char *string,
                                int length,
                                espeak_ng_ENCODING encoding);

ESPEAK_NG_API espeak_ng_STATUS
text_decoder_decode_wstring(espeak_ng_TEXT_DECODER *decoder,
                            const wchar_t *string,
                            int length);

ESPEAK_NG_API espeak_ng_STATUS
text_decoder_decode_string_multibyte(espeak_ng_TEXT_DECODER *decoder,
                                     const void *input,
                                     espeak_ng_ENCODING encoding,
                                     int flags);

ESPEAK_NG_API int
text_decoder_eof(espeak_ng_TEXT_DECODER *decoder);

ESPEAK_NG_API uint32_t
text_decoder_getc(espeak_ng_TEXT_DECODER *decoder);

ESPEAK_NG_API uint32_t
text_decoder_peekc(espeak_ng_TEXT_DECODER *decoder);

ESPEAK_NG_API const void *
text_decoder_get_buffer(espeak_ng_TEXT_DECODER *decoder);

#ifdef __cplusplus
}
#endif

#endif
