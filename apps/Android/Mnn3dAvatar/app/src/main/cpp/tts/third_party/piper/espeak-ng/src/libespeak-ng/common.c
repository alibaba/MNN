/*
 * Copyright (C) 2005 to 2013 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2017 Reece H. Dunn
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

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <wctype.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>
#include <ucd/ucd.h>

#include "common.h"
#include "translate.h"

#pragma GCC visibility push(default)

int GetFileLength(const char *filename)
{
	struct stat statbuf;

	if (stat(filename, &statbuf) != 0)
		return -errno;

	if (S_ISDIR(statbuf.st_mode))
		return -EISDIR;

	return statbuf.st_size;
}

void strncpy0(char *to, const char *from, int size)
{
	// strcpy with limit, ensures a zero terminator
	strncpy(to, from, size);
	to[size-1] = 0;
}

int utf8_in(int *c, const char *buf)
{
	/* Read a unicode characater from a UTF8 string
	 * Returns the number of UTF8 bytes used.
	 * buf: position of buffer is moved, if character is read
	 * c: holds UTF-16 representation of multibyte character by
	 * skipping UTF-8 header bits of bytes in following way:
	 * 2-byte character "ā":
	 * hex            binary
	 * c481           1100010010000001
	 *    |           11000100  000001
	 *    V              \    \ |    |
	 * 0101           0000000100000001
	 * 3-byte character "ꙅ":
	 * ea9985 111010101001100110000101
	 *            1010  011001  000101
	 *    |       +  +--.\   \  |    |
	 *    V        `--.  \`.  `.|    |
	 *   A645         1010011001000101
	 * 4-byte character "𠜎":
	 * f0a09c8e 11110000101000001001110010001110
	 *    V          000  100000  011100  001110
	 *   02070e         000000100000011100001110
	 */
	return utf8_in2(c, buf, 0);
}
#pragma GCC visibility pop

int utf8_out(unsigned int c, char *buf)
{
	// write a UTF-16 character into a buffer as UTF-8
	// returns the number of bytes written

	int n_bytes;
	int j;
	int shift;
	static const char unsigned code[4] = { 0, 0xc0, 0xe0, 0xf0 };

	if (c < 0x80) {
		buf[0] = c;
		return 1;
	}
	if (c >= 0x110000) {
		buf[0] = ' '; // out of range character code
		return 1;
	}
	if (c < 0x0800)
		n_bytes = 1;
	else if (c < 0x10000)
		n_bytes = 2;
	else
		n_bytes = 3;

	shift = 6*n_bytes;
	buf[0] = code[n_bytes] | (c >> shift);
	for (j = 0; j < n_bytes; j++) {
		shift -= 6;
		buf[j+1] = 0x80 + ((c >> shift) & 0x3f);
	}
	return n_bytes+1;
}

int utf8_in2(int *c, const char *buf, int backwards)
{
	// Reads a unicode characater from a UTF8 string
	// Returns the number of UTF8 bytes used.
	// c: holds integer representation of multibyte character
	// buf: position of buffer is moved, if character is read
	// backwards: set if we are moving backwards through the UTF8 string

	int c1;
	int n_bytes;
	static const unsigned char mask[4] = { 0xff, 0x1f, 0x0f, 0x07 };

	// find the start of the next/previous character
	while ((*buf & 0xc0) == 0x80) {
		// skip over non-initial bytes of a multi-byte utf8 character
		if (backwards)
			buf--;
		else
			buf++;
	}

	n_bytes = 0;

	if ((c1 = *buf++) & 0x80) {
		if ((c1 & 0xe0) == 0xc0)
			n_bytes = 1;
		else if ((c1 & 0xf0) == 0xe0)
			n_bytes = 2;
		else if ((c1 & 0xf8) == 0xf0)
			n_bytes = 3;

		c1 &= mask[n_bytes];
		int ix;
		for (ix = 0; ix < n_bytes; ix++)
		{
			if (!*buf)
				/* Oops, truncated */
				break;
			c1 = (c1 << 6) + (*buf++ & 0x3f);
		}
		n_bytes = ix;
	}
	*c = c1;
	return n_bytes+1;
}


int IsAlpha(unsigned int c)
{
	// Replacement for iswalph() which also checks for some in-word symbols

	static const unsigned short extra_indic_alphas[] = {
		0xa70, 0xa71, // Gurmukhi: tippi, addak
		0
	};

	if (iswalpha(c))
		return 1;

	if (c < 0x300)
		return 0;

	if ((c >= 0x901) && (c <= 0xdf7)) {
		// Indic scripts: Devanagari, Tamil, etc
		if ((c & 0x7f) < 0x64)
			return 1;
		if (lookupwchar(extra_indic_alphas, c) != 0)
			return 1;
		if ((c >= 0xd7a) && (c <= 0xd7f))
			return 1; // malaytalam chillu characters

		return 0;
	}

	if ((c >= 0x5b0) && (c <= 0x5c2))
		return 1; // Hebrew vowel marks

	if (c == 0x0605)
		return 1;

	if ((c == 0x670) || ((c >= 0x64b) && (c <= 0x65e)))
		return 1; // arabic vowel marks

	if ((c >= 0x300) && (c <= 0x36f))
		return 1; // combining accents

	if ((c >= 0xf40) && (c <= 0xfbc))
		return 1; // tibetan

	if ((c >= 0x1100) && (c <= 0x11ff))
		return 1; // Korean jamo

	if ((c >= 0x2800) && (c <= 0x28ff))
		return 1; // braille

	if ((c > 0x3040) && (c <= 0xa700))
		return 1; // Chinese/Japanese.  Should never get here, but Mac OS 10.4's iswalpha seems to be broken, so just make sure

	return 0;
}

// brackets, also 0x2014 to 0x021f which don't need to be in this list
static const unsigned short brackets[] = {
	'(', ')', '[', ']', '{', '}', '<', '>', '"', '\'', '`',
	0xab,   0xbb,   // double angle brackets
	0x300a, 0x300b, // double angle brackets (ideograph)
	0xe000+'<',     // private usage area
	0
};

int IsBracket(int c)
{
	if ((c >= 0x2014) && (c <= 0x201f))
		return 1;
	return lookupwchar(brackets, c);
}

int IsDigit09(unsigned int c)
{
	if ((c >= '0') && (c <= '9'))
		return 1;
	return 0;
}

int IsDigit(unsigned int c)
{
	if (iswdigit(c))
		return 1;

	if ((c >= 0x966) && (c <= 0x96f))
		return 1;

	return 0;
}

int IsSpace(unsigned int c)
{
	if (c == 0)
		return 0;
	if ((c >= 0x2500) && (c < 0x25a0))
		return 1; // box drawing characters
	if ((c >= 0xfff9) && (c <= 0xffff))
		return 1; // unicode specials
	return iswspace(c);
}

int isspace2(unsigned int c)
{
	// can't use isspace() because on Windows, isspace(0xe1) gives TRUE !
	if ( ((c & 0xff) == 0) || (c > ' '))
		return 0;
	return 1;
}

int is_str_totally_null(const char* str, int size) {
	// Tests if all bytes of str are null up to size
	// This should never be reimplemented with integers, because
	// this function has to work with unaligned char*
	// (casting to int when unaligned may result in ungaranteed behaviors)
	return (*str == 0 && memcmp(str, str+1, size-1) == 0);
}

int Read4Bytes(FILE *f)
{
	// Read 4 bytes (least significant first) into a word
	int ix;
	int acc = 0;

	for (ix = 0; ix < 4; ix++) {
		unsigned char c;
		c = fgetc(f) & 0xff;
		acc += (c << (ix*8));
	}
	return acc;
}

unsigned int StringToWord(const char *string)
{
	// Pack 4 characters into a word
	int ix;
	unsigned char c;
	unsigned int word;

	if (string == NULL)
		return 0;

	word = 0;
	for (ix = 0; ix < 4; ix++) {
		if (string[ix] == 0) break;
		c = string[ix];
		word |= (c << (ix*8));
	}
	return word;
}

int towlower2(unsigned int c, Translator *translator)
{
	// check for non-standard upper to lower case conversions
	if (c == 'I' && translator->langopts.dotless_i)
		return 0x131; // I -> ı

	return ucd_tolower(c);
}

static uint32_t espeak_rand_state = 0;

long espeak_rand(long min, long max) {
	// Ref: https://github.com/bminor/glibc/blob/glibc-2.36/stdlib/random_r.c#L364
	espeak_rand_state = (((uint64_t)espeak_rand_state * 1103515245) + 12345) % 0x7fffffff;
	long res = (long)espeak_rand_state;
	return (res % (max-min+1))-min;
}

void espeak_srand(long seed) {
	espeak_rand_state = (uint32_t)(seed);
	(void)espeak_rand(0, 1); // Dummy flush a generator
}

#pragma GCC visibility push(default)
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_SetRandSeed(long seed) {
	espeak_srand(seed);
	return ENS_OK;
}
#pragma GCC visibility pop
