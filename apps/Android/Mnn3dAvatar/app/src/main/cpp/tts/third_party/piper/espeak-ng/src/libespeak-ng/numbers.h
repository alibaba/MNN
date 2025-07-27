/*
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

#ifndef ESPEAK_NG_NUMBERS_H
#define ESPEAK_NG_NUMBERS_H

#include "translate.h"

#ifdef __cplusplus
extern "C"
{
#endif

void LookupAccentedLetter(Translator *tr, unsigned int letter, char *ph_buf);
void LookupLetter(Translator *tr, unsigned int letter, int next_byte, char *ph_buf1, int control);
int IsSuperscript(int letter);
void SetSpellingStress(Translator *tr, char *phonemes, int control, int n_chars);
int TranslateRoman(Translator *tr, char *word, char *ph_out, char *ph_out_end, WORD_TAB *wtab);
int TranslateNumber(Translator *tr, char *word1, char *ph_out, char *ph_out_end, unsigned int *flags, WORD_TAB *wtab, int control);


#ifdef __cplusplus
}
#endif

#endif

