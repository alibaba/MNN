/* SSML (Speech Synthesis Markup Language) processing APIs.
 *
 * Copyright (C) 2005 to 2015 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2018 Reece H. Dunn
 * Copyright (C) 2018 Juho Hiltunen
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
#ifndef ESPEAK_NG_SSML_API
#define ESPEAK_NG_SSML_API

#include <stdbool.h>
#include <wchar.h>

#include <espeak-ng/speak_lib.h>

#ifdef __cplusplus
extern "C"
{
#endif

// stack for language and voice properties
// frame 0 is for the defaults, before any ssml tags.
typedef struct {
        int tag_type;
        int voice_variant_number;
        int voice_gender;
        int voice_age;
        char voice_name[40];
        char language[20];
} SSML_STACK;

#define N_PARAM_STACK  20

#define SSML_SPEAK        1
#define SSML_VOICE        2
#define SSML_PROSODY      3
#define SSML_SAYAS        4
#define SSML_MARK         5
#define SSML_SENTENCE     6
#define SSML_PARAGRAPH    7
#define SSML_PHONEME      8
#define SSML_SUB          9
#define SSML_STYLE       10
#define SSML_AUDIO       11
#define SSML_EMPHASIS    12
#define SSML_BREAK       13
#define SSML_IGNORE_TEXT 14
#define HTML_BREAK       15
#define HTML_NOSPACE     16   // don't insert a space for this element, so it doesn't break a word
#define SSML_CLOSE       0x20 // for a closing tag, OR this with the tag type

int ProcessSsmlTag(wchar_t *xml_buf,
                   char *outbuf,
                   int *outix,
                   int n_outbuf,
                   const char *xmlbase,
                   bool *audio_text,
                   char *current_voice_id,
                   espeak_VOICE *base_voice,
                   char *base_voice_variant_name,
                   bool *ignore_text,
                   bool *clear_skipping_text,
                   int *sayas_mode,
                   int *sayas_start,
                   SSML_STACK *ssml_stack,
                   int *n_ssml_stack,
                   int *n_param_stack,
                   int *speech_parameters);

int ParseSsmlReference(char *ref,
                       int *c1,
                       int *c2);

#ifdef __cplusplus
}
#endif

#endif
