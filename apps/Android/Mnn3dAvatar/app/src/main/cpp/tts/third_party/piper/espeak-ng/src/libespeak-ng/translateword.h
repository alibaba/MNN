/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2017, 2020 Reece H. Dunn
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

#ifndef ESPEAK_NG_TRANSLATEWORD_H
#define ESPEAK_NG_TRANSLATEWORD_H

#include <stdbool.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/encoding.h>

#ifdef __cplusplus
extern "C"
{
#endif

int TranslateWord3(Translator *tr, char *word_start, WORD_TAB *wtab, char *word_out, bool *any_stressed_words, ALPHABET *current_alphabet, char word_phonemes[], size_t size_word_phonemes);

#ifdef __cplusplus
}
#endif

#endif
