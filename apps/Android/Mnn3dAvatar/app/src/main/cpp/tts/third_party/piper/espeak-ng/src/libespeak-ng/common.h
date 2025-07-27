/*
 * Copyright (C) 2005 to 2007 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2015 Reece H. Dunn
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

#ifndef ESPEAK_NG_COMMON_H
#define ESPEAK_NG_COMMON_H

#include "espeak-ng/espeak_ng.h"
#include "translate.h"

extern ESPEAK_NG_API int GetFileLength(const char *filename);
extern ESPEAK_NG_API void strncpy0(char *to, const char *from, int size);

void espeak_srand(long seed);
long espeak_rand(long min, long max);

int IsAlpha(unsigned int c);
int IsBracket(int c);
int IsDigit(unsigned int c);
int IsDigit09(unsigned int c);
int IsSpace(unsigned int c);
int isspace2(unsigned int c);
int is_str_totally_null(const char* str, int size); // Tests if all bytes of str up to size are null
int Read4Bytes(FILE *f);
unsigned int StringToWord(const char *string);
int towlower2(unsigned int c, Translator *translator); // Supports Turkish I

ESPEAK_NG_API int utf8_in(int *c, const char *buf);
int utf8_in2(int *c, const char *buf, int backwards);
int utf8_out(unsigned int c, char *buf);

#ifdef __cplusplus
}
#endif

#endif // SPEECH_H
