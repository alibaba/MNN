/*
 * Copyright (C) 2005 to 2015 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2016, 2020 Reece H. Dunn
 * Copyright (C) 2021 Juho Hiltunen
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
#include <locale.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "common.h"
#include "setlengths.h"          // for SetLengthMods
#include "translate.h"           // for Translator, LANGUAGE_OPTIONS, L, NUM...

// start of unicode pages for character sets
#define OFFSET_GREEK    0x380
#define OFFSET_CYRILLIC 0x420
#define OFFSET_ARMENIAN 0x530
#define OFFSET_HEBREW   0x590
#define OFFSET_ARABIC   0x600
#define OFFSET_SYRIAC   0x700
#define OFFSET_DEVANAGARI  0x900
#define OFFSET_BENGALI  0x980
#define OFFSET_GURMUKHI 0xa00
#define OFFSET_GUJARATI 0xa80
#define OFFSET_ORIYA    0xb00
#define OFFSET_TAMIL    0xb80
#define OFFSET_TELUGU   0xc00
#define OFFSET_KANNADA  0xc80
#define OFFSET_MALAYALAM 0xd00
#define OFFSET_SINHALA  0x0d80
#define OFFSET_THAI     0x0e00
#define OFFSET_LAO      0x0e80
#define OFFSET_TIBET    0x0f00
#define OFFSET_MYANMAR  0x1000
#define OFFSET_GEORGIAN 0x10a0
#define OFFSET_KOREAN   0x1100
#define OFFSET_ETHIOPIC 0x1200
#define OFFSET_SHAVIAN  0x10450

// character ranges must be listed in ascending unicode order
static const ALPHABET alphabets[] = {
	{ "_el",    OFFSET_GREEK,    0x380, 0x3ff,  L('e', 'l'), AL_DONT_NAME | AL_NOT_LETTERS | AL_WORDS },
	{ "_cyr",   OFFSET_CYRILLIC, 0x400, 0x52f,  0, 0 },
	{ "_hy",    OFFSET_ARMENIAN, 0x530, 0x58f,  L('h', 'y'), AL_WORDS },
	{ "_he",    OFFSET_HEBREW,   0x590, 0x5ff,  L('h', 'e'), 0 },
	{ "_ar",    OFFSET_ARABIC,   0x600, 0x6ff,  0, 0 },
	{ "_syc",   OFFSET_SYRIAC,   0x700, 0x74f,  0, 0 },
	{ "_hi",    OFFSET_DEVANAGARI, 0x900, 0x97f, L('h', 'i'), AL_WORDS },
	{ "_bn",    OFFSET_BENGALI,  0x0980, 0x9ff, L('b', 'n'), AL_WORDS },
	{ "_gur",   OFFSET_GURMUKHI, 0xa00, 0xa7f,  L('p', 'a'), AL_WORDS },
	{ "_gu",    OFFSET_GUJARATI, 0xa80, 0xaff,  L('g', 'u'), AL_WORDS },
	{ "_or",    OFFSET_ORIYA,    0xb00, 0xb7f,  0, 0 },
	{ "_ta",    OFFSET_TAMIL,    0xb80, 0xbff,  L('t', 'a'), AL_WORDS },
	{ "_te",    OFFSET_TELUGU,   0xc00, 0xc7f,  L('t', 'e'), 0 },
	{ "_kn",    OFFSET_KANNADA,  0xc80, 0xcff,  L('k', 'n'), AL_WORDS },
	{ "_ml",    OFFSET_MALAYALAM, 0xd00, 0xd7f,  L('m', 'l'), AL_WORDS },
	{ "_si",    OFFSET_SINHALA,  0xd80, 0xdff,  L('s', 'i'), AL_WORDS },
	{ "_th",    OFFSET_THAI,     0xe00, 0xe7f,  0, 0 },
	{ "_lo",    OFFSET_LAO,      0xe80, 0xeff,  0, 0 },
	{ "_ti",    OFFSET_TIBET,    0xf00, 0xfff,  0, 0 },
	{ "_my",    OFFSET_MYANMAR,  0x1000, 0x109f, 0, 0 },
	{ "_ka",    OFFSET_GEORGIAN, 0x10a0, 0x10ff, L('k', 'a'), AL_WORDS },
	{ "_ko",    OFFSET_KOREAN,   0x1100, 0x11ff, L('k', 'o'), AL_WORDS },
	{ "_eth",   OFFSET_ETHIOPIC, 0x1200, 0x139f, 0, 0 },
	{ "_braille", 0x2800,        0x2800, 0x28ff, 0, AL_NO_SYMBOL },
	{ "_ja",    0x3040,          0x3040, 0x30ff, 0, AL_NOT_CODE },
	{ "_zh",    0x3100,          0x3100, 0x9fff, 0, AL_NOT_CODE },
	{ "_ko",    0xa700,          0xa700, 0xd7ff, L('k', 'o'), AL_NOT_CODE | AL_WORDS },
	{ "_shaw",  OFFSET_SHAVIAN,  0x10450, 0x1047F, L('e', 'n'), 0 },
	{ NULL, 0, 0, 0, 0, 0 }
};

const ALPHABET *AlphabetFromChar(int c)
{
	// Find the alphabet from a character.
	const ALPHABET *alphabet = alphabets;

	while (alphabet->name != NULL) {
		if (c <= alphabet->range_max) {
			if (c >= alphabet->range_min)
				return alphabet;
			else
				break;
		}
		alphabet++;
	}
	return NULL;
}

static void Translator_Russian(Translator *tr);

static void SetLetterVowel(Translator *tr, int c)
{
	tr->letter_bits[c] = (tr->letter_bits[c] & 0x40) | 0x81; // keep value for group 6 (front vowels e,i,y)
}

static void ResetLetterBits(Translator *tr, int groups)
{
	// Clear all the specified groups
	unsigned int ix;
	unsigned int mask;

	mask = ~groups;

	for (ix = 0; ix < sizeof(tr->letter_bits); ix++)
		tr->letter_bits[ix] &= mask;
}

static void SetLetterBits(Translator *tr, int group, const char *string)
{
	int bits;
	unsigned char c;

	bits = (1L << group);
	while ((c = *string++) != 0)
		tr->letter_bits[c] |= bits;
}

static void SetLetterBitsRange(Translator *tr, int group, int first, int last)
{
	int bits;
	int ix;

	bits = (1L << group);
	for (ix = first; ix <= last; ix++)
		tr->letter_bits[ix] |= bits;
}

static void SetLetterBitsUTF8(Translator *tr, int group, const char *letters, int offset)
{
	// Add the letters to the specified letter group.
	const char *p = letters;
	int code = -1;
	while (code != 0) {
		int bytes = utf8_in(&code, p);
		if (code > 0x20)
			tr->letter_bits[code - offset] |= (1L << group);
		p += bytes;
	}
}

// ignore these characters
static const unsigned short chars_ignore_default[] = {
	// U+00AD SOFT HYPHEN
	//     Used to mark hyphenation points in words for where to split a
	//     word at the end of a line to provide readable justified text.
	0xad,   1,
	// U+200C ZERO WIDTH NON-JOINER
	//     Used to prevent combined ligatures being displayed in their
	//     combined form.
	0x200c, 1,
	// U+200D ZERO WIDTH JOINER
	//     Used to indicate an alternative connected form made up of the
	//     characters surrounding the ZWJ in Devanagari, Kannada, Malayalam
	//     and Emoji.
//	0x200d, 1, // Not ignored.
	// End of the ignored character list.
	0,      0
};

// alternatively, ignore characters but allow zero-width-non-joiner (lang-fa)
static const unsigned short chars_ignore_zwnj_hyphen[] = {
	// U+00AD SOFT HYPHEN
	//     Used to mark hyphenation points in words for where to split a
	//     word at the end of a line to provide readable justified text.
	0xad,   1,
	// U+0640 TATWEEL (KASHIDA)
	//     Used in Arabic scripts to stretch characters for justifying
	//     the text.
	0x640,  1,
	// U+200C ZERO WIDTH NON-JOINER
	//     Used to prevent combined ligatures being displayed in their
	//     combined form.
	0x200c, '-',
	// U+200D ZERO WIDTH JOINER
	//     Used to indicate an alternative connected form made up of the
	//     characters surrounding the ZWJ in Devanagari, Kannada, Malayalam
	//     and Emoji.
//	0x200d, 1, // Not ignored.
	// End of the ignored character list.
	0,      0
};

static const unsigned char utf8_ordinal[] = { 0xc2, 0xba, 0 }; // masculine ordinal character, UTF-8
static const unsigned char utf8_null[] = { 0 }; // null string, UTF-8

static Translator *NewTranslator(void)
{
	Translator *tr;
	int ix;
	static const unsigned char stress_amps2[] = { 18, 18, 20, 20, 20, 22, 22, 20 };
	static const short stress_lengths2[8] = { 182, 140, 220, 220, 220, 240, 260, 280 };
	static const wchar_t empty_wstring[1] = { 0 };
	static const wchar_t punct_in_word[2] = { '\'', 0 };  // allow hyphen within words
	static const unsigned char default_tunes[6] = { 0, 1, 2, 3, 0, 0 };

	// Translates character codes in the range transpose_min to transpose_max to
	// a number in the range 1 to 63.  0 indicates there is no translation.
	// Used up to 57 (max of 63)
	static const char transpose_map_latin[] = {
		 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, // 0x60
		16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,  0,  0,  0,  0,  0, // 0x70
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x80
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x90
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0xa0
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0xb0
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0xc0
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0xd0
		27, 28, 29,  0,  0, 30, 31, 32, 33, 34, 35, 36,  0, 37, 38,  0, // 0xe0
		 0,  0,  0, 39,  0,  0, 40,  0, 41,  0, 42,  0, 43,  0,  0,  0, // 0xf0
		 0,  0,  0, 44,  0, 45,  0, 46,  0,  0,  0,  0,  0, 47,  0,  0, // 0x100
		 0, 48,  0,  0,  0,  0,  0,  0,  0, 49,  0,  0,  0,  0,  0,  0, // 0x110
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x120
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x130
		 0,  0, 50,  0, 51,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x140
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 52,  0,  0,  0,  0, // 0x150
		 0, 53,  0, 54,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x160
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 55,  0, 56,  0, 57,  0, // 0x170
	};

	if ((tr = (Translator *)malloc(sizeof(Translator))) == NULL)
		return NULL;

	tr->encoding = ESPEAKNG_ENCODING_ISO_8859_1;
	dictionary_name[0] = 0;
	tr->dictionary_name[0] = 0;
	tr->phonemes_repeat[0] = 0;
	tr->dict_condition = 0;
	tr->dict_min_size = 0;
	tr->data_dictrules = NULL; // language_1   translation rules file
	tr->data_dictlist = NULL;  // language_2   dictionary lookup file

	tr->transpose_min = 0x60;
	tr->transpose_max = 0x17f;
	tr->transpose_map = transpose_map_latin;
	tr->frequent_pairs = NULL;

	tr->expect_verb = 0;
	tr->expect_past = 0;
	tr->expect_verb_s = 0;
	tr->expect_noun = 0;

	tr->clause_upper_count = 0;
	tr->clause_lower_count = 0;

	// only need lower case
	tr->letter_bits_offset = 0;
	memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
	memset(tr->letter_groups, 0, sizeof(tr->letter_groups));

	// 0-6 sets of characters matched by A B C H F G Y  in pronunciation rules
	// these may be set differently for different languages
	SetLetterBits(tr, 0, "aeiou"); // A  vowels, except y
	SetLetterBits(tr, 1, "bcdfgjklmnpqstvxz"); // B  hard consonants, excluding h,r,w
	SetLetterBits(tr, 2, "bcdfghjklmnpqrstvwxz"); // C  all consonants
	SetLetterBits(tr, 3, "hlmnr"); // H  'soft' consonants
	SetLetterBits(tr, 4, "cfhkpqstx"); // F  voiceless consonants
	SetLetterBits(tr, 5, "bdgjlmnrvwyz"); // G voiced
	SetLetterBits(tr, 6, "eiy"); // Letter group Y, front vowels
	SetLetterBits(tr, 7, "aeiouy"); // vowels, including y

	tr->char_plus_apostrophe = empty_wstring;
	tr->punct_within_word = punct_in_word;
	tr->chars_ignore = chars_ignore_default;

	for (ix = 0; ix < 8; ix++) {
		tr->stress_amps[ix] = stress_amps2[ix];
		tr->stress_lengths[ix] = stress_lengths2[ix];
	}
	memset(&(tr->langopts), 0, sizeof(tr->langopts));
	tr->langopts.max_lengthmod = 500;
	tr->langopts.lengthen_tonic = 20;

	tr->langopts.stress_rule = STRESSPOSN_2R;
	tr->langopts.unstressed_wd1 = 1;
	tr->langopts.unstressed_wd2 = 3;
	tr->langopts.param[LOPT_SONORANT_MIN] = 95;
	tr->langopts.param[LOPT_LONG_VOWEL_THRESHOLD] = 190/2;
	tr->langopts.param[LOPT_MAXAMP_EOC] = 19;
	tr->langopts.param[LOPT_UNPRONOUNCABLE] = 's'; // don't count this character at start of word
	tr->langopts.param[LOPT_BRACKET_PAUSE] = 4; // pause at bracket
	tr->langopts.param[LOPT_BRACKET_PAUSE_ANNOUNCED] = 2; // pauses when announcing bracket names
	tr->langopts.spelling_stress = false;
	tr->langopts.max_initial_consonants = 3;
	tr->langopts.replace_chars = NULL;
	tr->langopts.alt_alphabet_lang = L('e', 'n');
	tr->langopts.roman_suffix = utf8_null;
	tr->langopts.lowercase_sentence = false;

	SetLengthMods(tr, 201);

	tr->langopts.long_stop = 100;

	tr->langopts.max_roman = 49;
	tr->langopts.min_roman = 2;
	tr->langopts.thousands_sep = ',';
	tr->langopts.decimal_sep = '.';
	tr->langopts.numbers = NUM_DEFAULT;
	tr->langopts.break_numbers = BREAK_THOUSANDS;
	tr->langopts.max_digits = 14;

	// index by 0=. 1=, 2=?, 3=! 4=none, 5=emphasized
	unsigned char punctuation_to_tone[INTONATION_TYPES][PUNCT_INTONATIONS] = {
		{  0,  1,  2,  3, 0, 4 },
		{  0,  1,  2,  3, 0, 4 },
		{  5,  6,  2,  3, 0, 4 },
		{  5,  7,  1,  3, 0, 4 },
		{  8,  9, 10,  3, 0, 0 },
		{  8,  8, 10,  3, 0, 0 },
		{ 11, 11, 11, 11, 0, 0 }, // 6 test
		{ 12, 12, 12, 12, 0, 0 }
	};

	memcpy(tr->punct_to_tone, punctuation_to_tone, sizeof(tr->punct_to_tone));

	memcpy(tr->langopts.tunes, default_tunes, sizeof(tr->langopts.tunes));

	return tr;
}

// common letter pairs, encode these as a single byte
//  2 bytes, using the transposed character codes
static const short pairs_ru[] = {
	0x010c, //  ла   21052  0x23
	0x010e, //  на   18400
	0x0113, //  та   14254
	0x0301, //  ав   31083
	0x030f, //  ов   13420
	0x060e, //  не   21798
	0x0611, //  ре   19458
	0x0903, //  ви   16226
	0x0b01, //  ак   14456
	0x0b0f, //  ок   17836
	0x0c01, //  ал   13324
	0x0c09, //  ил   16877
	0x0e01, //  ан   15359
	0x0e06, //  ен   13543  0x30
	0x0e09, //  ин   17168
	0x0e0e, //  нн   15973
	0x0e0f, //  он   22373
	0x0e1c, //  ын   15052
	0x0f03, //  во   24947
	0x0f11, //  ро   13552
	0x0f12, //  со   16368
	0x100f, //  оп   19054
	0x1011, //  рп   17067
	0x1101, //  ар   23967
	0x1106, //  ер   18795
	0x1109, //  ир   13797
	0x110f, //  ор   21737
	0x1213, //  тс   25076
	0x1220, //  яс   14310
	0x7fff
};

static const unsigned char ru_vowels[] = { // (also kazakh) offset by 0x420 -- а е ё и о у ы э ю я ә ө ұ ү і
	0x10, 0x15, 0x31, 0x18, 0x1e, 0x23, 0x2b, 0x2d, 0x2e, 0x2f,  0xb9, 0xc9, 0x91, 0x8f, 0x36, 0
};
static const unsigned char ru_consonants[] = { // б в г д ж з й к л м н п р с т ф х ц ч ш щ ъ ь қ ң һ
	0x11, 0x12, 0x13, 0x14, 0x16, 0x17, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x21, 0x22, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2c, 0x73, 0x7b, 0x83, 0x9b, 0
};

static void SetArabicLetters(Translator *tr)
{
	const char *arab_vowel_letters = "َ  ُ  ِ";
	const char *arab_consonant_vowel_letters = "ا و ي";
	const char *arab_consonant_letters = "ب پ ت ة ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ئ ؤ ء أ آ إ ه";
	const char *arab_thick_letters = "ص ض ط ظ";
	const char *arab_shadda_letter = " ّ ";
	const char *arab_hamza_letter = " ّ ";
	const char *arab_sukun_letter = " ّ ";

	SetLetterBitsUTF8(tr, LETTERGP_A, arab_vowel_letters, OFFSET_ARABIC);
	SetLetterBitsUTF8(tr, LETTERGP_B, arab_consonant_vowel_letters, OFFSET_ARABIC);
	SetLetterBitsUTF8(tr, LETTERGP_C, arab_consonant_letters, OFFSET_ARABIC);
	SetLetterBitsUTF8(tr, LETTERGP_F, arab_thick_letters, OFFSET_ARABIC);
	SetLetterBitsUTF8(tr, LETTERGP_G, arab_shadda_letter, OFFSET_ARABIC);
	SetLetterBitsUTF8(tr, LETTERGP_H, arab_hamza_letter, OFFSET_ARABIC);
	SetLetterBitsUTF8(tr, LETTERGP_Y, arab_sukun_letter, OFFSET_ARABIC);
}

static void SetCyrillicLetters(Translator *tr)
{
	// Set letter types for Cyrillic script languages: bg (Bulgarian), ru (Russian), tt (Tatar), uk (Ukrainian).

	// character codes offset by 0x420
	static const char cyrl_soft[] = { 0x2c, 0x19, 0x27, 0x29, 0 }; // letter group B  [k ts; s;] -- ь й ч щ
	static const char cyrl_hard[] = { 0x2a, 0x16, 0x26, 0x28, 0 }; // letter group H  [S Z ts] -- ъ ж ц ш
	static const char cyrl_nothard[] = { 0x11, 0x12, 0x13, 0x14, 0x17, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x21, 0x22, 0x24, 0x25, 0x27, 0x29, 0x2c, 0 }; // б в г д з й к л м н п р с т ф х ч щ ь
	static const char cyrl_voiced[] = { 0x11, 0x12, 0x13, 0x14, 0x16, 0x17, 0 };    // letter group G  (voiced obstruents) -- б в г д ж з
	static const char cyrl_ivowels[] = { 0x2c, 0x2e, 0x2f, 0x31, 0 };   // letter group Y  (iotated vowels & soft-sign) -- ь ю я ё
	tr->encoding = ESPEAKNG_ENCODING_KOI8_R;
	tr->transpose_min = 0x430;  // convert cyrillic from unicode into range 0x01 to 0x22
	tr->transpose_max = 0x451;
	tr->transpose_map = NULL;
	tr->frequent_pairs = pairs_ru;

	tr->letter_bits_offset = OFFSET_CYRILLIC;
	memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
	SetLetterBits(tr, LETTERGP_A, (char *)ru_vowels);
	SetLetterBits(tr, LETTERGP_B, cyrl_soft);
	SetLetterBits(tr, LETTERGP_C, (char *)ru_consonants);
	SetLetterBits(tr, LETTERGP_H, cyrl_hard);
	SetLetterBits(tr, LETTERGP_F, cyrl_nothard);
	SetLetterBits(tr, LETTERGP_G, cyrl_voiced);
	SetLetterBits(tr, LETTERGP_Y, cyrl_ivowels);
	SetLetterBits(tr, LETTERGP_VOWEL2, (char *)ru_vowels);
}

static void SetIndicLetters(Translator *tr)
{
	// Set letter types for Devanagari (Indic) script languages: Devanagari, Tamill, etc.

	static const char deva_consonants2[] = { 0x02, 0x03, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x7b, 0x7c, 0x7e, 0x7f, 0 };
	static const char deva_vowels2[] = { 0x60, 0x61, 0x55, 0x56, 0x57, 0x62, 0x63, 0 };  // non-consecutive vowels and vowel-signs

	memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
	SetLetterBitsRange(tr, LETTERGP_A, 0x04, 0x14); // vowel letters
	SetLetterBitsRange(tr, LETTERGP_A, 0x3e, 0x4d); // + vowel signs, and virama
	SetLetterBits(tr, LETTERGP_A, deva_vowels2);     // + extra vowels and vowel signs

	SetLetterBitsRange(tr, LETTERGP_B, 0x3e, 0x4d); // vowel signs, and virama
	SetLetterBits(tr, LETTERGP_B, deva_vowels2);     // + extra vowels and vowel signs

	SetLetterBitsRange(tr, LETTERGP_C, 0x15, 0x39); // the main consonant range
	SetLetterBits(tr, LETTERGP_C, deva_consonants2); // + additional consonants

	SetLetterBitsRange(tr, LETTERGP_Y, 0x04, 0x14); // vowel letters
	SetLetterBitsRange(tr, LETTERGP_Y, 0x3e, 0x4c); // + vowel signs
	SetLetterBits(tr, LETTERGP_Y, deva_vowels2);     // + extra vowels and vowel signs

	tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1;    // disable check for unpronouncable words
	tr->langopts.suffix_add_e = tr->letter_bits_offset + 0x4d; // virama
}

static void SetupTranslator(Translator *tr, const short *lengths, const unsigned char *amps)
{
	if (lengths != NULL)
		memcpy(tr->stress_lengths, lengths, sizeof(tr->stress_lengths));
	if (amps != NULL)
		memcpy(tr->stress_amps, amps, sizeof(tr->stress_amps));
}

Translator *SelectTranslator(const char *name)
{
	int name2 = 0;
	Translator *tr;

	static const short stress_lengths_equal[8] = { 230, 230,  230, 230,  0, 0,  230, 230 };
	static const unsigned char stress_amps_equal[8] = { 19, 19, 19, 19, 19, 19, 19, 19 };

	static const short stress_lengths_fr[8] = { 190, 170,  190, 200,  0, 0,  190, 240 };
	static const unsigned char stress_amps_fr[8] = { 18, 16, 18, 18, 18, 18, 18, 18 };

	static const unsigned char stress_amps_sk[8] = { 17, 16, 20, 20, 20, 22, 22, 21 };
	static const short stress_lengths_sk[8] = { 190, 190, 210, 210, 0, 0, 210, 210 };

	static const short stress_lengths_ta[8] = { 200, 200,  210, 210,  0, 0,  230, 230 };
	static const short stress_lengths_ta2[8] = { 230, 230,  240, 240,  0, 0,  260, 260 };
	static const unsigned char stress_amps_ta[8] = { 18, 18, 18, 18, 20, 20, 22, 22 };

	tr = NewTranslator();
	strcpy(tr->dictionary_name, name);

	// convert name string into a word of up to 4 characters, for the switch()
	while (*name != 0)
		name2 = (name2 << 8) + *name++;

	switch (name2)
	{
	case L('m', 'i'):
	case L('m', 'y'):
	case L4('p', 'i', 'q', 'd'): // piqd
	case L('p', 'y'):
	case L('q', 'u'):
	case L3('q', 'u', 'c'):
	case L('t', 'h'):
	case L('u', 'z'):
	{
		tr->langopts.numbers = 0; // disable numbers until the definition are complete in _list file
	}
		break;
	case L('a', 'f'):
	{
		static const short stress_lengths_af[8] = { 170, 140, 220, 220,  0, 0, 250, 270 };
		SetupTranslator(tr, stress_lengths_af, NULL);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.vowel_pause = 0x30;
		tr->langopts.param[LOPT_DIERESES] = 1;
		tr->langopts.param[LOPT_PREFIXES] = 1;
		SetLetterVowel(tr, 'y'); // add 'y' to vowels

		tr->langopts.numbers = NUM_SWAP_TENS | NUM_HUNDRED_AND | NUM_SINGLE_AND | NUM_ROMAN | NUM_1900;
		tr->langopts.accents = 1;
	}
		break;
	case L('a', 'm'): // Amharic, Ethiopia
	{
		SetupTranslator(tr, stress_lengths_fr, stress_amps_fr);
		tr->letter_bits_offset = OFFSET_ETHIOPIC;
		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_NO_AUTO_2 | S_FINAL_DIM; // don't use secondary stress
		tr->langopts.length_mods0 = tr->langopts.length_mods;  // don't lengthen vowels in the last syllable
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1;           // disable check for unpronouncable words
		tr->langopts.numbers = NUM_OMIT_1_HUNDRED;
	}
		break;
	case L('a', 'r'): // Arabic
		tr->transpose_min = OFFSET_ARABIC; // for ar_list, use 6-bit character codes
		tr->transpose_max = 0x65f;
		tr->transpose_map = NULL;
		tr->letter_bits_offset = OFFSET_ARABIC;
		tr->langopts.numbers = NUM_SWAP_TENS | NUM_AND_UNITS | NUM_HUNDRED_AND | NUM_OMIT_1_HUNDRED | NUM_AND_HUNDRED | NUM_THOUSAND_AND | NUM_OMIT_1_THOUSAND;
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_6;
		SetArabicLetters(tr);
		break;
	case L('b', 'e'): // Belarusian
	{
		static const unsigned char stress_amps_be[8] = { 12, 10, 8, 8, 0, 0, 16, 17 };
		static const short stress_lengths_be[8] = { 160, 140, 200, 140, 0, 0, 240, 160 };
		static const wchar_t vowels_be[] = { // offset by 0x420 -- а е ё о у ы э ю я і
			0x10, 0x15, 0x31, 0x1e, 0x23, 0x2b, 0x2d, 0x2e, 0x2f, 0x36, 0
		};
		static const unsigned char consonants_be[] = { // б в г д ж з й к л м н п р с т ф х ц ч ш ў
			0x11, 0x12, 0x13, 0x14, 0x16, 0x17, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x21, 0x22, 0x24, 0x25, 0x26, 0x27, 0x28, 0x3e, 0
		};

		tr->langopts.stress_flags = S_NO_AUTO_2 | S_NO_DIM; // don't use secondary stress
		tr->letter_bits_offset = OFFSET_CYRILLIC;
		tr->transpose_min = 0x430;  // convert cyrillic from unicode into range 0x01 to 0x2f
		tr->transpose_max = 0x45e;
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBits(tr, LETTERGP_A, (char *)vowels_be);
		SetLetterBits(tr, LETTERGP_C, (char *)consonants_be);
		SetLetterBits(tr, LETTERGP_VOWEL2, (char *)vowels_be);

		SetupTranslator(tr, stress_lengths_be, stress_amps_be);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_5;
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED;
		tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_THOUSANDS | NUM2_THOUSANDS_VAR1; // variant numbers before thousands
	}
		break;
	case L('b', 'g'): // Bulgarian
	{
		SetCyrillicLetters(tr);
		SetLetterVowel(tr, 0x2a);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_5;
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 0x432; // [v]  don't count this character at start of word
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x107; // devoice at end of word, and change voicing to match a following consonant (except v)
		tr->langopts.param[LOPT_REDUCE] = 2;
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_OMIT_1_HUNDRED | NUM_HUNDRED_AND | NUM_AND_UNITS | NUM_SINGLE_AND | NUM_ROMAN | NUM_ROMAN_ORDINAL | NUM_ROMAN_CAPITALS;
		tr->langopts.thousands_sep = ' '; // don't allow dot as thousands separator
	}
		break;
	case L('b', 'n'): // Bengali
	case L('a', 's'): // Assamese
	case L3('b', 'p', 'y'): // Manipuri  (temporary placement - it's not indo-european)
	{
		static const short stress_lengths_bn[8] = { 180, 180,  210, 210,  0, 0,  230, 240 };
		static const unsigned char stress_amps_bn[8] = { 18, 18, 18, 18, 20, 20, 22, 22 };
		static const char bn_consonants2[3] = { 0x70, 0x71, 0 };

		SetupTranslator(tr, stress_lengths_bn, stress_amps_bn);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags =  S_MID_DIM | S_FINAL_DIM; // use 'diminished' for unstressed final syllable
		tr->letter_bits_offset = OFFSET_BENGALI;
		SetIndicLetters(tr); // call this after setting OFFSET_BENGALI
		SetLetterBitsRange(tr, LETTERGP_B, 0x01, 0x01); // candranindu
		SetLetterBitsRange(tr, LETTERGP_F, 0x3e, 0x4c); // vowel signs, but not virama
		SetLetterBits(tr, LETTERGP_C, bn_consonants2);

		tr->langopts.numbers = NUM_SWAP_TENS;
		tr->langopts.break_numbers = BREAK_LAKH_BN;

		if (name2 == L3('b', 'p', 'y')) {
			tr->langopts.numbers = NUM_DEFAULT;
			tr->langopts.numbers2 = NUM2_SWAP_THOUSANDS;
		}

	}
		break;
	case L3('c', 'h', 'r'): // Cherokee
	{
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		tr->langopts.stress_flags = S_NO_AUTO_2 | S_FINAL_DIM | S_FINAL_DIM_ONLY | S_EO_CLAUSE1;
	}
		break;
	case L('c', 'y'): // Welsh
	{
		static const short stress_lengths_cy[8] = { 170, 220, 180, 180, 0, 0, 250, 270 };
		static const unsigned char stress_amps_cy[8] = { 17, 15, 18, 18, 0, 0, 22, 20 }; // 'diminished' is used to mark a quieter, final unstressed syllable

		SetupTranslator(tr, stress_lengths_cy, stress_amps_cy);

		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_14;
		tr->langopts.stress_rule = STRESSPOSN_2R;

		// 'diminished' is an unstressed final syllable
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.unstressed_wd1 = 0;
		tr->langopts.unstressed_wd2 = 2;
		tr->langopts.param[LOPT_SONORANT_MIN] = 120; // limit the shortening of sonorants before short vowels

		tr->langopts.numbers = NUM_OMIT_1_HUNDRED;

		SetLetterVowel(tr, 'w'); // add letter to vowels and remove from consonants
		SetLetterVowel(tr, 'y');
	}
		break;
	case L('d', 'a'): // Danish
	{
		static const short stress_lengths_da[8] = { 160, 140, 200, 200, 0, 0, 220, 230 };
		SetupTranslator(tr, stress_lengths_da, NULL);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.param[LOPT_PREFIXES] = 1;
		SetLetterVowel(tr, 'y');
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_SWAP_TENS | NUM_HUNDRED_AND | NUM_OMIT_1_HUNDRED | NUM_ORDINAL_DOT | NUM_1900 | NUM_ROMAN | NUM_ROMAN_CAPITALS | NUM_ROMAN_ORDINAL;
	}
		break;
	case L('d', 'e'):
	{
		static const short stress_lengths_de[8] = { 150, 130, 200, 200,  0, 0, 270, 270 };
		static const unsigned char stress_amps_de[] = { 20, 20, 20, 20, 20, 22, 22, 20 };
		SetupTranslator(tr, stress_lengths_de, stress_amps_de);
		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.word_gap = 0x8; // don't use linking phonemes
		tr->langopts.vowel_pause = 0x30;
		tr->langopts.param[LOPT_PREFIXES] = 1;
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x100; // devoice at end of word
		tr->langopts.param[LOPT_LONG_VOWEL_THRESHOLD] = 175/2;

		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_SWAP_TENS | NUM_ALLOW_SPACE | NUM_ORDINAL_DOT | NUM_ROMAN;
		SetLetterVowel(tr, 'y');
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 2; // use de_rules for unpronouncable rules
	}
		break;
	case L('e', 'n'):
	{
		static const short stress_lengths_en[8] = { 182, 140, 220, 220, 0, 0, 248, 275 };
		SetupTranslator(tr, stress_lengths_en, NULL);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = 0x08;
		tr->langopts.numbers = NUM_HUNDRED_AND | NUM_ROMAN | NUM_1900;
		tr->langopts.max_digits = 33;
		tr->langopts.param[LOPT_COMBINE_WORDS] = 2; // allow "mc" to cmbine with the following word
		tr->langopts.suffix_add_e = 'e';
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 2; // use en_rules for unpronouncable rules
		SetLetterBits(tr, 6, "aeiouy"); // Group Y: vowels, including y
	}
		break;
	case L('e', 'l'): // Greek
	case L3('g', 'r', 'c'): // Ancient Greek
	{
		static const short stress_lengths_el[8] = { 155, 180,  210, 210,  0, 0,  270, 300 };
		static const unsigned char stress_amps_el[8] = { 15, 12, 20, 20, 20, 22, 22, 21 }; // 'diminished' is used to mark a quieter, final unstressed syllable

		// character codes offset by 0x380
		static const char el_vowels[] = { 0x10, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x35, 0x37, 0x39, 0x3f, 0x45, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0 };
		static const char el_fvowels[] = { 0x2d, 0x2e, 0x2f, 0x35, 0x37, 0x39, 0x45, 0x4d, 0 }; // ε η ι υ  έ ή ί ύ _
		static const char el_voiceless[] = { 0x38, 0x3a, 0x3e, 0x40, 0x42, 0x43, 0x44, 0x46, 0x47, 0 }; // θ κ ξ π ς σ τ φ χ _
		static const char el_consonants[] = { 0x32, 0x33, 0x34, 0x36, 0x38, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x40, 0x41, 0x42, 0x43, 0x44, 0x46, 0x47, 0x48, 0 };
		static const wchar_t el_char_apostrophe[] = { 0x3c3, 0 }; // σ _

		SetupTranslator(tr, stress_lengths_el, stress_amps_el);

		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_7;
		tr->char_plus_apostrophe = el_char_apostrophe;

		tr->letter_bits_offset = OFFSET_GREEK;
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBits(tr, LETTERGP_A, el_vowels);
		SetLetterBits(tr, LETTERGP_VOWEL2, el_vowels);
		SetLetterBits(tr, LETTERGP_B, el_voiceless);
		SetLetterBits(tr, LETTERGP_C, el_consonants);
		SetLetterBits(tr, LETTERGP_Y, el_fvowels); // front vowels: ε η ι υ _

		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY; // mark unstressed final syllables as diminished
		tr->langopts.unstressed_wd1 = 0;
		tr->langopts.unstressed_wd2 = 2;
		tr->langopts.param[LOPT_SONORANT_MIN] = 130; // limit the shortening of sonorants before short vowels

		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA;
		tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_THOUSANDS | NUM2_MULTIPLE_ORDINAL | NUM2_ORDINAL_NO_AND;

		if (name2 == L3('g', 'r', 'c')) {
			// ancient greek
			tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1;
		}
	}
		break;
	case L('e', 'o'):
	{
		static const short stress_lengths_eo[8] = { 150, 140,  180, 180,    0,   0,  200, 200 };
		static const unsigned char stress_amps_eo[] = { 16, 14, 20, 20, 20, 22, 22, 21 };
		static const wchar_t eo_char_apostrophe[2] = { 'l', 0 };

		SetupTranslator(tr, stress_lengths_eo, stress_amps_eo);

		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_3;
		tr->char_plus_apostrophe = eo_char_apostrophe;

		tr->langopts.vowel_pause = 2;
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.unstressed_wd2 = 2;

		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_ALLOW_SPACE | NUM_ROMAN;
	}
		break;
	case L('e', 's'): // Spanish
	case L('a', 'n'): // Aragonese
	case L('c', 'a'): // Catalan
	case L('i', 'a'): // Interlingua
	case L3('p', 'a', 'p'): // Papiamento
	{
		static const short stress_lengths_es[8] = { 160, 145,  155, 150,  0, 0,  200, 245 };
		static const unsigned char stress_amps_es[8] = { 16, 14, 15, 16, 20, 20, 22, 22 }; // 'diminished' is used to mark a quieter, final unstressed syllable
		static const wchar_t ca_punct_within_word[] = { '\'', 0xb7, 0 }; // ca: allow middle-dot within word

		SetupTranslator(tr, stress_lengths_es, stress_amps_es);

		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable
		tr->langopts.stress_rule = STRESSPOSN_2R;

		// stress last syllable if it doesn't end in vowel or "s" or "n"
		// 'diminished' is an unstressed final syllable
		tr->langopts.stress_flags = S_FINAL_SPANISH | S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.unstressed_wd1 = 0;
		tr->langopts.unstressed_wd2 = 2;
		tr->langopts.param[LOPT_SONORANT_MIN] = 120; // limit the shortening of sonorants before short vowels

		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_AND_UNITS | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_ROMAN | NUM_ROMAN_AFTER | NUM_DFRACTION_4;
		tr->langopts.numbers2 = NUM2_MULTIPLE_ORDINAL | NUM2_ORDINAL_NO_AND;

		if (name2 == L('c', 'a')) {
			// stress last syllable unless word ends with a vowel
			tr->punct_within_word = ca_punct_within_word;
			tr->langopts.stress_flags = S_FINAL_SPANISH | S_FINAL_DIM_ONLY | S_FINAL_NO_2 | S_NO_AUTO_2 | S_FIRST_PRIMARY;
		} else if (name2 == L('i', 'a')) {
			tr->langopts.stress_flags = S_FINAL_SPANISH | S_FINAL_DIM_ONLY | S_FINAL_NO_2;
			tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_ROMAN | NUM_ROMAN_AFTER;
		} else if (name2 == L('a', 'n')) {
			tr->langopts.stress_flags = S_FINAL_SPANISH | S_FINAL_DIM_ONLY | S_FINAL_NO_2;
			tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_AND_UNITS | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_ROMAN | NUM_ROMAN_ORDINAL;
			tr->langopts.numbers2 = NUM2_ORDINAL_NO_AND;
			tr->langopts.roman_suffix = utf8_ordinal;
		} else if (name2 == L3('p', 'a', 'p')) {
			// stress last syllable unless word ends with a vowel
			tr->langopts.stress_rule = STRESSPOSN_1R;
			tr->langopts.stress_flags = S_FINAL_VOWEL_UNSTRESSED | S_FINAL_DIM_ONLY | S_FINAL_NO_2 | S_NO_AUTO_2;
		} else
			tr->langopts.param[LOPT_UNPRONOUNCABLE] = 2; // use es_rules for unpronouncable rules
	}
		break;
	case L('e', 'u'): // basque
	{
		static const short stress_lengths_eu[8] = { 200, 200,  200, 200,  0, 0,  210, 230 }; // very weak stress
		static const unsigned char stress_amps_eu[8] = { 16, 16, 18, 18, 18, 18, 18, 18 };
		SetupTranslator(tr, stress_lengths_eu, stress_amps_eu);
		tr->langopts.stress_flags = S_FINAL_VOWEL_UNSTRESSED | S_MID_DIM;
		tr->langopts.param[LOPT_SUFFIX] = 1;
		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_HUNDRED_AND | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_VIGESIMAL;
	}
		break;
	case L('f', 'a'): // Farsi
	{
		// Convert characters in the range 0x620 to 0x6cc to the range 1 to 63.
		// 0 indicates no translation for this character
		static const char transpose_map_fa[] = {
			 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, // 0x620
			16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,  0,  0,  0,  0,  0, // 0x630
			 0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, // 0x640
			42, 43,  0,  0, 44,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x650
			 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x660
			 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 45,  0, // 0x670
			 0,  0,  0,  0,  0,  0, 46,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x680
			 0,  0,  0,  0,  0,  0,  0,  0, 47,  0,  0,  0,  0,  0,  0,  0, // 0x690
			 0,  0,  0,  0,  0,  0,  0,  0,  0, 48,  0,  0,  0,  0,  0, 49, // 0x6a0
			 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, // 0x6b0
			50,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 51              // 0x6c0
		};
		tr->transpose_min = 0x620;
		tr->transpose_max = 0x6cc;
		tr->transpose_map = transpose_map_fa;
		tr->letter_bits_offset = OFFSET_ARABIC;

		tr->langopts.numbers = NUM_AND_UNITS | NUM_HUNDRED_AND;
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words

		tr->chars_ignore = chars_ignore_zwnj_hyphen; // replace ZWNJ by hyphen
	}
		break;
	case L('e', 't'): // Estonian
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_4;
		// fallthrough:
	case L('f', 'i'): // Finnish
	{
		tr->langopts.long_stop = 130;

		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_DFRACTION_2 | NUM_ORDINAL_DOT;
		SetLetterVowel(tr, 'y');
	}
		break;
	case L('f', 'o'): // Faroese
	{
		//static const short stress_lengths_da[8] = { 160, 140, 200, 200, 0, 0, 220, 230 };
		//SetupTranslator(tr, stress_lengths_da, NULL);

		//tr->langopts.stress_rule = STRESSPOSN_1L;
		//tr->langopts.param[LOPT_PREFIXES] = 1;
		//SetLetterVowel(tr, 'y');
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_SWAP_TENS | NUM_HUNDRED_AND | NUM_OMIT_1_HUNDRED | NUM_ORDINAL_DOT | NUM_1900 | NUM_ROMAN | NUM_ROMAN_CAPITALS | NUM_ROMAN_ORDINAL;
	}
		break;
	case L('f', 'r'): // french
	{
		SetupTranslator(tr, stress_lengths_fr, stress_amps_fr);
		tr->langopts.stress_rule = STRESSPOSN_1R; // stress on final syllable
		tr->langopts.stress_flags = S_NO_AUTO_2 | S_FINAL_DIM; // don't use secondary stress
		tr->langopts.param[LOPT_IT_LENGTHEN] = 1; // remove lengthen indicator from unstressed syllables
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable
		tr->langopts.accents = 2; // Say "Capital" after the letter.

		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_OMIT_1_HUNDRED | NUM_NOPAUSE | NUM_ROMAN | NUM_ROMAN_CAPITALS | NUM_ROMAN_AFTER | NUM_VIGESIMAL | NUM_DFRACTION_4;
		SetLetterVowel(tr, 'y');
	}
		break;
    case L3('h','a', 'k'): // Hakka Chinese
    {
        tr->langopts.stress_flags = S_NO_DIM; // don't automatically set diminished stress (may be set in the intonation module)
        tr->langopts.tone_language = 1; // Tone language, use  CalcPitches_Tone() rather than CalcPitches()
        tr->langopts.tone_numbers = 1; // a number after letters indicates a tone number (eg. pinyin or jyutping)
        tr->langopts.ideographs = 1;
    }
        break;
	case L('h','e'): // Hebrew
	{
		tr->langopts.param[LOPT_APOSTROPHE] = 2; // bit 1  Apostrophe at end of word is part of the word, for words like בָּגָאז׳
		tr->langopts.stress_flags = S_NO_AUTO_2; // don't use secondary stress
		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DFRACTION_2 | NUM_AND_UNITS | NUM_HUNDRED_AND | NUM_SINGLE_AND;
	}
		break;
	case L('g', 'a'): // irish
	case L('g', 'd'): // scots gaelic
	{
		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_NO_AUTO_2; // don't use secondary stress
		tr->langopts.numbers = NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND;
		tr->langopts.accents = 2; // 'capital' after letter name
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 3; // don't count apostrophe
		tr->langopts.param[LOPT_IT_LENGTHEN] = 1; // remove [:] phoneme from non-stressed syllables (Lang=gd)
	}
		break;
	case L('g','n'):   // guarani
		{
			tr->langopts.stress_rule = STRESSPOSN_1R;      // stress on final syllable
			tr->langopts.length_mods0 = tr->langopts.length_mods;  // don't lengthen vowels in the last syllable
		}
		break;
	case L('h', 'i'): // Hindi
	case L('n', 'e'): // Nepali
	case L('o', 'r'): // Oriya
	case L('p', 'a'): // Punjabi
	case L('g', 'u'): // Gujarati
	case L('m', 'r'): // Marathi
	{
		static const short stress_lengths_hi[8] = { 190, 190,  210, 210,  0, 0,  230, 250 };
		static const unsigned char stress_amps_hi[8] = { 17, 14, 20, 19, 20, 22, 22, 21 };

		SetupTranslator(tr, stress_lengths_hi, stress_amps_hi);
		tr->encoding = ESPEAKNG_ENCODING_ISCII;
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.stress_rule = STRESSPOSN_1RH; // stress on last heaviest syllable, excluding final syllable
		tr->langopts.stress_flags =  S_MID_DIM | S_FINAL_DIM; // use 'diminished' for unstressed final syllable
		tr->langopts.numbers = NUM_SWAP_TENS;
		tr->langopts.break_numbers = BREAK_LAKH_HI;
		tr->letter_bits_offset = OFFSET_DEVANAGARI;

		if (name2 == L('p', 'a'))
			tr->letter_bits_offset = OFFSET_GURMUKHI;
		else if (name2 == L('g', 'u')) {
			SetupTranslator(tr, stress_lengths_equal, stress_amps_equal);
			tr->letter_bits_offset = OFFSET_GUJARATI;
			tr->langopts.stress_rule = STRESSPOSN_2R;
		} else if (name2 == L('n', 'e')) {
			SetupTranslator(tr, stress_lengths_equal, stress_amps_equal);
			tr->langopts.break_numbers = BREAK_LAKH;
			tr->langopts.max_digits = 22;
			tr->langopts.numbers2 |= NUM2_ENGLISH_NUMERALS;
		} else if (name2 == L('o', 'r'))
			tr->letter_bits_offset = OFFSET_ORIYA;
		SetIndicLetters(tr);
	}
		break;
	case L('h', 'r'): // Croatian
	case L('b', 's'): // Bosnian
	case L('s', 'r'): // Serbian
	{
		static const unsigned char stress_amps_hr[8] = { 17, 17, 20, 20, 20, 22, 22, 21 };
		static const short stress_lengths_hr[8] = { 180, 160, 200, 200, 0, 0, 220, 230 };
		static const short stress_lengths_sr[8] = { 160, 150, 200, 200, 0, 0, 250, 260 };

		strcpy(tr->dictionary_name, "hbs");

		if (name2 == L('s', 'r'))
			SetupTranslator(tr, stress_lengths_sr, stress_amps_hr);
		else
			SetupTranslator(tr, stress_lengths_hr, stress_amps_hr);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_2;

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_FINAL_NO_2;
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x3;
		tr->langopts.max_initial_consonants = 5;
		tr->langopts.spelling_stress = true;
		tr->langopts.accents = 1;

		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_HUNDRED_AND | NUM_OMIT_1_HUNDRED | NUM_DECIMAL_COMMA | NUM_THOUS_SPACE | NUM_DFRACTION_2 | NUM_ROMAN_CAPITALS;
		tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_THOUSANDS | NUM2_THOUSANDPLEX_VAR_MILLIARDS | NUM2_THOUSANDS_VAR5;
		tr->langopts.our_alphabet = OFFSET_CYRILLIC; // don't say "cyrillic" before letter names

		SetLetterVowel(tr, 'y');
		SetLetterVowel(tr, 'r');
	}
		break;
	case L('h', 't'): // Haitian Creole
		tr->langopts.stress_rule = STRESSPOSN_1R; // stress on final syllable
		tr->langopts.stress_flags = S_NO_AUTO_2 | S_FINAL_DIM; // don't use secondary stress
		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_OMIT_1_HUNDRED | NUM_NOPAUSE | NUM_ROMAN | NUM_VIGESIMAL | NUM_DFRACTION_4;
		break;
	case L('h', 'u'): // Hungarian
	{
		static const unsigned char stress_amps_hu[8] = { 17, 17, 19, 19, 20, 22, 22, 21 };
		static const short stress_lengths_hu[8] = { 185, 195, 195, 190, 0, 0, 210, 220 };

		SetupTranslator(tr, stress_lengths_hu, stress_amps_hu);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_2;

		tr->langopts.vowel_pause = 0x20;
		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY | S_FINAL_NO_2 | S_NO_AUTO_2 | 0x8000 | S_HYPEN_UNSTRESS;
		tr->langopts.unstressed_wd1 = 2;
		tr->langopts.param[LOPT_ANNOUNCE_PUNCT] = 2; // don't break clause before announcing . ? !

		tr->langopts.numbers = NUM_DFRACTION_5 | NUM_ALLOW_SPACE | NUM_ROMAN | NUM_ROMAN_ORDINAL | NUM_ROMAN_CAPITALS | NUM_ORDINAL_DOT | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND;
		tr->langopts.thousands_sep = ' '; // don't allow dot as thousands separator
		tr->langopts.decimal_sep = ',';
		tr->langopts.max_roman = 899;
		tr->langopts.min_roman = 1;
		SetLetterVowel(tr, 'y');
		tr->langopts.spelling_stress = true;
		SetLengthMods(tr, 3); // all equal
	}
		break;
	case L('h', 'y'): // Armenian
	{
		static const short stress_lengths_hy[8] = { 250, 200,  250, 250,  0, 0,  250, 250 };
		static const char hy_vowels[] = { 0x31, 0x35, 0x37, 0x38, 0x3b, 0x48, 0x55, 0 };
		static const char hy_consonants[] = {
			0x32, 0x33, 0x34, 0x36, 0x39, 0x3a, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44,
			0x46, 0x47, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53, 0x54, 0x56, 0
		};
		static const char hy_consonants2[] = { 0x45, 0 };

		SetupTranslator(tr, stress_lengths_hy, NULL);
		tr->langopts.stress_rule = STRESSPOSN_1R; // default stress on final syllable

		tr->letter_bits_offset = OFFSET_ARMENIAN;
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBits(tr, LETTERGP_A, hy_vowels);
		SetLetterBits(tr, LETTERGP_VOWEL2, hy_vowels);
		SetLetterBits(tr, LETTERGP_B, hy_consonants); // not including 'j'
		SetLetterBits(tr, LETTERGP_C, hy_consonants);
		SetLetterBits(tr, LETTERGP_C, hy_consonants2); // add 'j'
		tr->langopts.max_initial_consonants = 6;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_OMIT_1_HUNDRED;
	}
		break;
			
	case L('i', 'o'): // Ido International Auxiliary Language 
	{
		static const short stress_lengths_eo[8] = { 150, 140,  180, 180,    0,   0,  200, 200 };
		static const unsigned char stress_amps_eo[] = { 16, 14, 20, 20, 20, 22, 22, 21 };
		static const wchar_t eo_char_apostrophe[2] = { 'l', 0 };

		SetupTranslator(tr, stress_lengths_eo, stress_amps_eo);

		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_3;
		tr->char_plus_apostrophe = eo_char_apostrophe;

		tr->langopts.vowel_pause = 2;
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.unstressed_wd2 = 2;

		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_ALLOW_SPACE | NUM_AND_UNITS| NUM_HUNDRED_AND | NUM_ROMAN;
	}
	break;
			
	case L('i', 'd'): // Indonesian
	case L('m', 's'): // Malay
	{
		static const short stress_lengths_id[8] = { 160, 200,  180, 180,  0, 0,  220, 240 };
		static const unsigned char stress_amps_id[8] = { 16, 18, 18, 18, 20, 22, 22, 21 };

		SetupTranslator(tr, stress_lengths_id, stress_amps_id);
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_ROMAN;
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.accents = 2; // "capital" after letter name
	}
		break;
	case L('i', 's'): // Icelandic
	{
		static const short stress_lengths_is[8] = { 180, 160, 200, 200, 0, 0, 240, 250 };
		static const wchar_t is_lettergroup_B[] = { 'c', 'f', 'h', 'k', 'p', 't', 'x', 0xfe, 0 }; // voiceless conants, including 'þ'  ?? 's'

		SetupTranslator(tr, stress_lengths_is, NULL);
		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_FINAL_NO_2;
		tr->langopts.param[LOPT_IT_LENGTHEN] = 0x11; // remove lengthen indicator from unstressed vowels
		tr->langopts.param[LOPT_REDUCE] = 2;

		ResetLetterBits(tr, 0x18);
		SetLetterBits(tr, 4, "kpst"); // Letter group F
		SetLetterBits(tr, 3, "jvr"); // Letter group H
		tr->letter_groups[1] = is_lettergroup_B;
		SetLetterVowel(tr, 'y');
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_SINGLE_AND | NUM_HUNDRED_AND | NUM_AND_UNITS | NUM_1900;
		tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_THOUSANDS;
	}
		break;
	case L('i', 't'): // Italian
	{
		static const short stress_lengths_it[8] = { 160, 140, 150, 165, 0, 0, 218, 305 };
		static const unsigned char stress_amps_it[8] = { 17, 15, 18, 16, 20, 22, 22, 22 };
		SetupTranslator(tr, stress_lengths_it, stress_amps_it);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_NO_AUTO_2 | S_FINAL_DIM_ONLY | S_PRIORITY_STRESS;
		tr->langopts.vowel_pause = 1;
		tr->langopts.unstressed_wd1 = 0;
		tr->langopts.unstressed_wd2 = 2;
		tr->langopts.param[LOPT_IT_LENGTHEN] = 2; // remove lengthen indicator from unstressed or non-penultimate syllables
		tr->langopts.param[LOPT_SONORANT_MIN] = 130; // limit the shortening of sonorants before short vowels
		tr->langopts.param[LOPT_REDUCE] = 1; // reduce vowels even if phonemes are specified in it_list
		tr->langopts.param[LOPT_ALT] = 2; // call ApplySpecialAttributes2() if a word has $alt or $alt2
		tr->langopts.numbers = NUM_SINGLE_VOWEL | NUM_OMIT_1_HUNDRED |NUM_DECIMAL_COMMA | NUM_DFRACTION_1 | NUM_ROMAN | NUM_ROMAN_CAPITALS | NUM_ROMAN_ORDINAL;
		tr->langopts.numbers2 = NUM2_NO_TEEN_ORDINALS;
		tr->langopts.roman_suffix = utf8_ordinal;
		tr->langopts.accents = 2; // Say "Capital" after the letter.
		SetLetterVowel(tr, 'y');
	}
		break;
	case L3('j', 'b', 'o'): // Lojban
	{
		static const short stress_lengths_jbo[8] = { 145, 145, 170, 160, 0, 0, 330, 350 };
		static const wchar_t jbo_punct_within_word[] = { '.', ',', '\'', 0x2c8, 0 }; // allow period and comma within a word, also stress marker (from LOPT_CAPS_IN_WORD)

		SetupTranslator(tr, stress_lengths_jbo, NULL);
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.vowel_pause = 0x20c; // pause before a word which starts with a vowel, or after a word which ends in a consonant
		tr->punct_within_word = jbo_punct_within_word;
		tr->langopts.param[LOPT_CAPS_IN_WORD] = 1; // capitals indicate stressed syllables
		SetLetterVowel(tr, 'y');
		tr->langopts.max_lengthmod = 368;
		tr->langopts.numbers = 0; // disable numbers until the definition are complete in _list file
	}
		break;
	case L('k', 'a'): // Georgian
	{
		// character codes offset by 0x1080
		static const char ka_vowels[] = { 0x30, 0x34, 0x38, 0x3d, 0x43, 0x55, 0x57, 0 };
		static const char ka_consonants[] =
		{ 0x31, 0x32, 0x33, 0x35, 0x36, 0x37, 0x39, 0x3a, 0x3b, 0x3c, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x44,
		  0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53, 0x54, 0x56, 0 };
		SetupTranslator(tr, stress_lengths_ta, stress_amps_ta);
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBits(tr, LETTERGP_A, ka_vowels);
		SetLetterBits(tr, LETTERGP_C, ka_consonants);
		SetLetterBits(tr, LETTERGP_VOWEL2, ka_vowels);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_FINAL_NO_2;
		tr->letter_bits_offset = OFFSET_GEORGIAN;
		tr->langopts.max_initial_consonants = 7;
		tr->langopts.max_digits = 32;
		tr->langopts.numbers = NUM_VIGESIMAL | NUM_AND_UNITS | NUM_OMIT_1_HUNDRED |NUM_OMIT_1_THOUSAND | NUM_DFRACTION_5 | NUM_ROMAN;

		tr->langopts.alt_alphabet = OFFSET_CYRILLIC;
		tr->langopts.alt_alphabet_lang = L('r', 'u');
	}
		break;
	case L('k', 'k'): // Kazakh
	{
		static const unsigned char stress_amps_tr[8] = { 18, 16, 20, 21, 20, 21, 21, 20 };
		static const short stress_lengths_tr[8] = { 190, 180, 230, 230, 0, 0, 250, 250 };

		tr->letter_bits_offset = OFFSET_CYRILLIC;
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBits(tr, LETTERGP_A, (char *)ru_vowels);
		SetLetterBits(tr, LETTERGP_C, (char *)ru_consonants);
		SetLetterBits(tr, LETTERGP_VOWEL2, (char *)ru_vowels);

		SetupTranslator(tr, stress_lengths_tr, stress_amps_tr);

		tr->langopts.stress_rule = STRESSPOSN_1RU; // stress on the last syllable, before any explicitly unstressed syllable
		tr->langopts.stress_flags = S_NO_AUTO_2 + S_NO_EOC_LENGTHEN; // no automatic secondary stress, don't lengthen at end-of-clause
		tr->langopts.lengthen_tonic = 0;
		tr->langopts.param[LOPT_SUFFIX] = 1;

		tr->langopts.numbers =  NUM_OMIT_1_HUNDRED | NUM_DFRACTION_6;
		tr->langopts.max_initial_consonants = 2;
		SetLengthMods(tr, 3); // all equal
	}
		break;
	case L('k', 'l'): // Greenlandic
	{
		SetupTranslator(tr, stress_lengths_equal, stress_amps_equal);
		tr->langopts.stress_rule = STRESSPOSN_GREENLANDIC;
		tr->langopts.stress_flags = S_NO_AUTO_2;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_SWAP_TENS | NUM_HUNDRED_AND | NUM_OMIT_1_HUNDRED | NUM_ORDINAL_DOT | NUM_1900 | NUM_ROMAN | NUM_ROMAN_CAPITALS | NUM_ROMAN_ORDINAL;
	}
		break;
	case L('k', 'o'): // Korean, TEST
	{
		static const char ko_ivowels[] = { 0x63, 0x64, 0x67, 0x68, 0x6d, 0x72, 0x74, 0x75, 0 }; // y and i vowels
		static const unsigned char ko_voiced[] = { 0x02, 0x05, 0x06, 0xab, 0xaf, 0xb7, 0xbc, 0 }; // voiced consonants, l,m,n,N

		tr->letter_bits_offset = OFFSET_KOREAN;
		tr->langopts.our_alphabet = 0xa700;
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBitsRange(tr, LETTERGP_A, 0x61, 0x75);
		SetLetterBits(tr, LETTERGP_Y, ko_ivowels);
		SetLetterBits(tr, LETTERGP_G, (const char *)ko_voiced);

		tr->langopts.stress_rule = STRESSPOSN_2LLH; // ?? 1st syllable if it is heavy, else 2nd syllable
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		tr->langopts.numbers = NUM_OMIT_1_HUNDRED;
		tr->langopts.numbers2 = NUM2_MYRIADS;
		tr->langopts.break_numbers = BREAK_MYRIADS;
		tr->langopts.max_digits = 20;
	}
		break;
	case L('k', 'u'): // Kurdish
	{
		static const unsigned char stress_amps_ku[8] = { 18, 18, 20, 20, 20, 22, 22, 21 };
		static const short stress_lengths_ku[8] = { 180, 180, 190, 180, 0, 0, 230, 240 };

		SetupTranslator(tr, stress_lengths_ku, stress_amps_ku);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_9;

		tr->langopts.stress_rule = STRESSPOSN_1RU; // stress on the last syllable, before any explicitly unstressed syllable

		tr->langopts.numbers = NUM_HUNDRED_AND | NUM_AND_UNITS | NUM_OMIT_1_HUNDRED | NUM_AND_HUNDRED;
		tr->langopts.max_initial_consonants = 2;
	}
		break;
	case L('k', 'y'): // Kyrgyx
		tr->langopts.numbers = NUM_DEFAULT;
		break;
	case L('l', 'a'): // Latin
	{
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_4; // includes a,e,i,o,u-macron
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_NO_AUTO_2;
		tr->langopts.unstressed_wd1 = 0;
		tr->langopts.unstressed_wd2 = 2;
		tr->langopts.param[LOPT_DIERESES] = 1;
		tr->langopts.numbers = NUM_ROMAN;
		tr->langopts.max_roman = 5000;
	}
		break;
	case L('l', 't'): // Lithuanian
	{
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_4;
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_NO_AUTO_2;
		tr->langopts.unstressed_wd1 = 0;
		tr->langopts.unstressed_wd2 = 2;
		tr->langopts.param[LOPT_DIERESES] = 1;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_DFRACTION_4 | NUM_ORDINAL_DOT;
		tr->langopts.numbers2 = NUM2_THOUSANDS_VAR4;
		tr->langopts.max_roman = 5000;
	}
		break;
	case L('l', 'v'): // latvian
	case L3('l', 't', 'g'): // latgalian
	{
		static const unsigned char stress_amps_lv[8] = { 14, 10, 10, 8, 0, 0, 20, 15 };
		static const short stress_lengths_lv[8] = { 180, 180, 180, 160, 0, 0, 230, 180 };

		SetupTranslator(tr, stress_lengths_lv, stress_amps_lv);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.spelling_stress = true;
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_4;
		tr->langopts.max_digits = 33;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_DFRACTION_4 | NUM_ORDINAL_DOT;
		tr->langopts.stress_flags = S_NO_AUTO_2 | S_FINAL_DIM | S_FINAL_DIM_ONLY | S_EO_CLAUSE1;
	}
		break;
	case L('m', 'k'): // Macedonian
	{
		static const wchar_t vowels_cyrillic[] = {
			// also include 'р' [R]
			0x440, 0x430, 0x435, 0x438, 0x439, 0x43e, 0x443, 0x44b, 0x44d,
			0x44e, 0x44f, 0x450, 0x451, 0x456, 0x457, 0x45d, 0x45e, 0
		};
		static const unsigned char stress_amps_mk[8] = { 17, 17, 20, 20, 20, 22, 22, 21 };
		static const short stress_lengths_mk[8] = { 180, 160, 200, 200, 0, 0, 220, 230 };

		SetupTranslator(tr, stress_lengths_mk, stress_amps_mk);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_5;
		tr->letter_groups[0] = tr->letter_groups[7] = vowels_cyrillic;
		tr->letter_bits_offset = OFFSET_CYRILLIC;

		tr->langopts.stress_rule = STRESSPOSN_3R; // antipenultimate
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_AND_UNITS | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_DFRACTION_2;
		tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_THOUSANDS | NUM2_THOUSANDPLEX_VAR_MILLIARDS | NUM2_THOUSANDS_VAR2;
	}
		break;
	case L('m', 't'): // Maltese
	{
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_3;
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x100; // devoice at end of word
		tr->langopts.stress_rule = STRESSPOSN_2R; // penultimate
		tr->langopts.numbers = NUM_DEFAULT;
	}
		break;
	case L('n', 'l'): // Dutch
	{
		static const short stress_lengths_nl[8] = { 160, 135, 210, 210,  0, 0, 260, 280 };

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.vowel_pause = 0x30; // ??
		tr->langopts.param[LOPT_DIERESES] = 1;
		tr->langopts.param[LOPT_PREFIXES] = 1;
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x100; // devoice at end of word
		SetLetterVowel(tr, 'y');

		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_SWAP_TENS | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_ALLOW_SPACE | NUM_1900 | NUM_ORDINAL_DOT;
		tr->langopts.ordinal_indicator = "e";
		tr->langopts.stress_flags = S_FIRST_PRIMARY;
		memcpy(tr->stress_lengths, stress_lengths_nl, sizeof(tr->stress_lengths));
	}
		break;
	case L('n', 'b'): // Norwegian
	{
		static const short stress_lengths_no[8] = { 160, 140, 200, 200, 0, 0, 220, 230 };

		SetupTranslator(tr, stress_lengths_no, NULL);
		tr->langopts.stress_rule = STRESSPOSN_1L;
		SetLetterVowel(tr, 'y');
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_HUNDRED_AND | NUM_ALLOW_SPACE | NUM_1900 | NUM_ORDINAL_DOT;
	}
		break;
	case L('o', 'm'): // Oromo
	{
		static const unsigned char stress_amps_om[] = { 18, 15, 20, 20, 20, 22, 22, 22 };
		static const short stress_lengths_om[8] = { 200, 200, 200, 200, 0, 0, 200, 200 };

		SetupTranslator(tr, stress_lengths_om, stress_amps_om);
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY | S_FINAL_NO_2 | 0x80000;
		tr->langopts.numbers = NUM_OMIT_1_HUNDRED | NUM_HUNDRED_AND;
		tr->langopts.numbers2 = NUM2_SWAP_THOUSANDS;
	}
		break;
	case L('p', 'l'): // Polish
	{
		static const short stress_lengths_pl[8] = { 160, 190,  175, 175,  0, 0,  200, 210 };
		static const unsigned char stress_amps_pl[8] = { 17, 13, 19, 19, 20, 22, 22, 21 }; // 'diminished' is used to mark a quieter, final unstressed syllable

		SetupTranslator(tr, stress_lengths_pl, stress_amps_pl);

		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_2;
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY; // mark unstressed final syllables as diminished
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x9;
		tr->langopts.max_initial_consonants = 7; // for example: wchrzczony :)
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_DFRACTION_2;
		tr->langopts.numbers2 = NUM2_THOUSANDS_VAR3;
		tr->langopts.param[LOPT_COMBINE_WORDS] = 4 + 0x100; // combine 'nie' (marked with $alt2) with some 1-syllable (and 2-syllable) words (marked with $alt)
		SetLetterVowel(tr, 'y');
	}
		break;
	case L('p', 't'): // Portuguese
	{
		static const short stress_lengths_pt[8] = { 170, 115,  210, 240,  0, 0,  260, 280 };
		static const unsigned char stress_amps_pt[8] = { 16, 11, 19, 21, 20, 22, 22, 21 }; // 'diminished' is used to mark a quieter, final unstressed syllable

		SetupTranslator(tr, stress_lengths_pt, stress_amps_pt);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.stress_rule = STRESSPOSN_1R; // stress on final syllable
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2 | S_INITIAL_2 | S_PRIORITY_STRESS;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_DFRACTION_2 | NUM_HUNDRED_AND | NUM_AND_UNITS | NUM_ROMAN_CAPITALS;
		tr->langopts.numbers2 = NUM2_MULTIPLE_ORDINAL | NUM2_NO_TEEN_ORDINALS | NUM2_ORDINAL_NO_AND;
		tr->langopts.max_roman = 5000;
		SetLetterVowel(tr, 'y');
		ResetLetterBits(tr, 0x2);
		SetLetterBits(tr, 1, "bcdfgjkmnpqstvxz"); // B  hard consonants, excluding h,l,r,w,y
		tr->langopts.param[LOPT_ALT] = 2; // call ApplySpecialAttributes2() if a word has $alt or $alt2
		tr->langopts.accents = 2; // 'capital' after letter name
	}
		break;
	case L('r', 'o'): // Romanian
	{
		static const short stress_lengths_ro[8] = { 170, 170,  180, 180,  0, 0,  240, 260 };
		static const unsigned char stress_amps_ro[8] = { 15, 13, 18, 18, 20, 22, 22, 21 };

		SetupTranslator(tr, stress_lengths_ro, stress_amps_ro);

		tr->langopts.stress_rule = STRESSPOSN_1R;
		tr->langopts.stress_flags = S_FINAL_VOWEL_UNSTRESSED | S_FINAL_DIM_ONLY;

		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_2;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_DFRACTION_3 | NUM_AND_UNITS | NUM_ROMAN;
		tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_ALL;
	}
		break;
	case L('r', 'u'): // Russian
		Translator_Russian(tr);
		break;
	case L('s', 'k'): // Slovak
	case L('c', 's'): // Czech
	{
		static const char sk_voiced[] = "bdgjlmnrvwzaeiouy";

		SetupTranslator(tr, stress_lengths_sk, stress_amps_sk);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_2;

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x3;
		tr->langopts.max_initial_consonants = 5;
		tr->langopts.spelling_stress = true;
		tr->langopts.param[LOPT_COMBINE_WORDS] = 4; // combine some prepositions with the following word

		tr->langopts.numbers = NUM_OMIT_1_HUNDRED | NUM_DFRACTION_2 | NUM_ROMAN;
		tr->langopts.numbers2 = NUM2_THOUSANDS_VAR2;
		tr->langopts.thousands_sep = 0; // no thousands separator
		tr->langopts.decimal_sep = ',';

		if (name2 == L('c', 's'))
			tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_MILLIARDS | NUM2_THOUSANDS_VAR2;

		SetLetterVowel(tr, 'y');
		SetLetterVowel(tr, 'r');
		ResetLetterBits(tr, 0x20);
		SetLetterBits(tr, 5, sk_voiced);
	}
		break;
	case L('s', 'i'): // Sinhala
	{
		SetupTranslator(tr, stress_lengths_ta, stress_amps_ta);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.spelling_stress = true;

		tr->letter_bits_offset = OFFSET_SINHALA;
		memset(tr->letter_bits, 0, sizeof(tr->letter_bits));
		SetLetterBitsRange(tr, LETTERGP_A, 0x05, 0x16); // vowel letters
		SetLetterBitsRange(tr, LETTERGP_A, 0x4a, 0x73); // + vowel signs, and virama

		SetLetterBitsRange(tr, LETTERGP_B, 0x4a, 0x73); // vowel signs, and virama

		SetLetterBitsRange(tr, LETTERGP_C, 0x1a, 0x46); // the main consonant range

		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		tr->langopts.suffix_add_e = tr->letter_bits_offset + 0x4a; // virama
		tr->langopts.numbers =  NUM_OMIT_1_THOUSAND | NUM_SINGLE_STRESS_L | NUM_DFRACTION_7;
		tr->langopts.numbers2 =  NUM2_PERCENT_BEFORE;
		tr->langopts.break_numbers = BREAK_LAKH_HI;
	}
		break;
	case L('s', 'l'): // Slovenian
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_2;
		tr->langopts.stress_rule = STRESSPOSN_2R; // Temporary
		tr->langopts.stress_flags = S_NO_AUTO_2;
		tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x103;
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 0x76; // [v]  don't count this character at start of word
		tr->langopts.param[LOPT_ALT] = 2; // call ApplySpecialAttributes2() if a word has $alt or $alt2
		tr->langopts.param[LOPT_IT_LENGTHEN] = 1; // remove lengthen indicator from unstressed syllables
		tr->letter_bits[(int)'r'] |= 0x80; // add 'r' to letter group 7, vowels for Unpronouncable test
		tr->langopts.numbers =  NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_SWAP_TENS | NUM_OMIT_1_HUNDRED | NUM_DFRACTION_2 | NUM_ORDINAL_DOT | NUM_ROMAN;
		tr->langopts.numbers2 = NUM2_THOUSANDS_VAR4;
		tr->langopts.thousands_sep = ' '; // don't allow dot as thousands separator
		break;
		
	case L3('s', 'm', 'j'): // Lule Saami
	{
		static const unsigned char stress_amps_fi[8] = { 18, 16, 22, 22, 20, 22, 22, 22 };
		static const short stress_lengths_fi[8] = { 150, 180, 200, 200, 0, 0, 210, 250 };

		SetupTranslator(tr, stress_lengths_fi, stress_amps_fi);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags = S_FINAL_DIM_ONLY | S_FINAL_NO_2 | S_2_TO_HEAVY; // move secondary stress from light to a following heavy syllable
		tr->langopts.long_stop = 130;

		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_SWAP_TENS | NUM_OMIT_1_HUNDRED | NUM_DFRACTION_2 | NUM_ORDINAL_DOT;
		SetLetterVowel(tr, 'y');
		tr->langopts.spelling_stress = true;
		tr->langopts.intonation_group = 3; // less intonation, don't raise pitch at comma
	}
		break;
		
	case L('s', 'q'): // Albanian
	{
		static const short stress_lengths_sq[8] = { 150, 150,  180, 180,  0, 0,  300, 300 };
		static const unsigned char stress_amps_sq[8] = { 16, 12, 16, 16, 20, 20, 21, 19 };

		SetupTranslator(tr, stress_lengths_sq, stress_amps_sq);

		tr->langopts.stress_rule = STRESSPOSN_1R;
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2 | S_FINAL_VOWEL_UNSTRESSED;
		SetLetterVowel(tr, 'y');
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_HUNDRED_AND | NUM_AND_UNITS | NUM_DFRACTION_4;
		tr->langopts.accents = 2; // "capital" after letter name
	}
		break;
	case L('s', 'v'): // Swedish
	{
		static const unsigned char stress_amps_sv[] = { 16, 16, 20, 20, 20, 22, 22, 21 };
		static const short stress_lengths_sv[8] = { 160, 135, 220, 220, 0, 0, 250, 280 };
		SetupTranslator(tr, stress_lengths_sv, stress_amps_sv);

		tr->langopts.stress_rule = STRESSPOSN_1L;
		SetLetterVowel(tr, 'y');
		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_ALLOW_SPACE | NUM_1900;
		tr->langopts.accents = 1;
	}
		break;
	case L('s', 'w'): // Swahili
	case L('t', 'n'): // Setswana
	{
		static const short stress_lengths_sw[8] = { 160, 170,  200, 200,    0,   0,  320, 340 };
		static const unsigned char stress_amps_sw[] = { 16, 12, 19, 19, 20, 22, 22, 21 };

		SetupTranslator(tr, stress_lengths_sw, stress_amps_sw);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.vowel_pause = 1;
		tr->langopts.stress_rule = STRESSPOSN_2R;
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2;
		tr->langopts.max_initial_consonants = 4; // for example: mwngi


		tr->langopts.numbers = NUM_AND_UNITS | NUM_HUNDRED_AND | NUM_SINGLE_AND | NUM_OMIT_1_HUNDRED;
	}
		break;
	case L('t', 'a'): // Tamil
	case L('k', 'n'): // Kannada
	case L('m', 'l'): // Malayalam
	case L('t', 'e'): // Telugu
	{
		SetupTranslator(tr, stress_lengths_ta2, stress_amps_ta);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.stress_flags =  S_FINAL_DIM_ONLY | S_FINAL_NO_2; // use 'diminished' for unstressed final syllable
		tr->langopts.spelling_stress = true;
		tr->langopts.break_numbers = BREAK_LAKH_DV;

		if (name2 == L('t', 'a')) {
			SetupTranslator(tr, stress_lengths_ta, NULL);
			tr->letter_bits_offset = OFFSET_TAMIL;
			tr->langopts.numbers =  NUM_OMIT_1_THOUSAND;
			tr->langopts.numbers2 = NUM2_ORDINAL_AND_THOUSANDS;
			tr->langopts.param[LOPT_WORD_MERGE] = 1; // don't break vowels between words
		} else if (name2 == L('m', 'l')) {
			static const short stress_lengths_ml[8] = { 180, 160,  240, 240,  0, 0,  260, 260 };
			SetupTranslator(tr, stress_lengths_ml, stress_amps_equal);
			tr->letter_bits_offset = OFFSET_MALAYALAM;
			tr->langopts.numbers = NUM_OMIT_1_THOUSAND | NUM_OMIT_1_HUNDRED;
			tr->langopts.numbers2 = NUM2_OMIT_1_HUNDRED_ONLY;
			tr->langopts.stress_rule = STRESSPOSN_1SL; // 1st syllable, unless 1st vowel is short and 2nd is long
		} else if (name2 == L('k', 'n')) {
			tr->letter_bits_offset = OFFSET_KANNADA;
			tr->langopts.numbers = NUM_DEFAULT;
		} else if (name2 == L('t', 'e')) {
			tr->letter_bits_offset = OFFSET_TELUGU;
			tr->langopts.numbers = NUM_DEFAULT;
			tr->langopts.numbers2 = NUM2_ORDINAL_DROP_VOWEL;
		}
		SetIndicLetters(tr); // call this after setting OFFSET_
		SetLetterBitsRange(tr, LETTERGP_B, 0x4e, 0x4e); // chillu-virama (unofficial)
	}
		break;
	case L('t', 'r'): // Turkish
	case L('a', 'z'): // Azerbaijan
	{
		static const unsigned char stress_amps_tr[8] = { 18, 16, 20, 21, 20, 21, 21, 20 };
		static const short stress_lengths_tr[8] = { 190, 180, 200, 230, 0, 0, 240, 250 };

		SetupTranslator(tr, stress_lengths_tr, stress_amps_tr);
		tr->encoding = ESPEAKNG_ENCODING_ISO_8859_9;

		tr->langopts.stress_rule = STRESSPOSN_1RU; // stress on the last syllable, before any explicitly unstressed syllable
		tr->langopts.stress_flags = S_NO_AUTO_2; // no automatic secondary stress
		tr->langopts.dotless_i = 1;
		tr->langopts.param[LOPT_SUFFIX] = 1;

		if (name2 == L('a', 'z'))
			tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA  | NUM_ALLOW_SPACE | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_DFRACTION_2;
		else
			tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_DFRACTION_2;
		tr->langopts.max_initial_consonants = 2;
	}
		break;
	case L('t', 't'): // Tatar
	{
		SetCyrillicLetters(tr);
		SetupTranslator(tr, stress_lengths_fr, stress_amps_fr);
		tr->langopts.stress_rule = STRESSPOSN_1R; // stress on final syllable
		tr->langopts.stress_flags = S_NO_AUTO_2; // no automatic secondary stress
		tr->langopts.numbers = NUM_SINGLE_STRESS | NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED | NUM_OMIT_1_THOUSAND | NUM_DFRACTION_4;
	}
		break;
	case L('u', 'k'): // Ukrainian
	{
		Translator_Russian(tr);
	}
		break;
	case L('u', 'r'): // Urdu
	case L('s', 'd'): // Sindhi
	{
		tr->letter_bits_offset = OFFSET_ARABIC;
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		tr->langopts.numbers = NUM_SWAP_TENS;
		tr->langopts.break_numbers = BREAK_LAKH_UR;
	}
		break;
	case L('v', 'i'): // Vietnamese
	{
		static const short stress_lengths_vi[8] = { 150, 150,  180, 180,  210, 230,  230, 240 };
		static const unsigned char stress_amps_vi[] = { 16, 16, 16, 16, 22, 22, 22, 22 };
		static const wchar_t vowels_vi[] = {
			 0x61,   0xe0,   0xe1, 0x1ea3,   0xe3, 0x1ea1, // a
			0x103, 0x1eb1, 0x1eaf, 0x1eb3, 0x1eb5, 0x1eb7, // ă
			 0xe2, 0x1ea7, 0x1ea5, 0x1ea9, 0x1eab, 0x1ead, // â
			 0x65,   0xe8,   0xe9, 0x1ebb, 0x1ebd, 0x1eb9, // e
			 0xea, 0x1ec1, 0x1ebf, 0x1ec3, 0x1ec5, 0x1ec7, // i
			 0x69,   0xec,   0xed, 0x1ec9,  0x129, 0x1ecb, // i
			 0x6f,   0xf2,   0xf3, 0x1ecf,   0xf5, 0x1ecd, // o
			 0xf4, 0x1ed3, 0x1ed1, 0x1ed5, 0x1ed7, 0x1ed9, // ô
			0x1a1, 0x1edd, 0x1edb, 0x1edf, 0x1ee1, 0x1ee3, // ơ
			 0x75,   0xf9,   0xfa, 0x1ee7,  0x169, 0x1ee5, // u
			0x1b0, 0x1eeb, 0x1ee9, 0x1eed, 0x1eef, 0x1ef1, // ư
			 0x79, 0x1ef3,   0xfd, 0x1ef7, 0x1ef9, 0x1ef5, // y
			0
		};

		SetupTranslator(tr, stress_lengths_vi, stress_amps_vi);
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable

		tr->langopts.stress_rule = STRESSPOSN_1L;
		tr->langopts.word_gap = 0x21; // length of a final vowel is less dependent on the next consonant, don't merge consonant with next word
		tr->letter_groups[0] = tr->letter_groups[7] = vowels_vi;
		tr->langopts.tone_language = 1; // Tone language, use  CalcPitches_Tone() rather than CalcPitches()
		tr->langopts.unstressed_wd1 = 2;
		tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_HUNDRED_AND_DIGIT | NUM_DFRACTION_4 | NUM_ZERO_HUNDRED;

	}
		
		break;
	case L3('x', 'e', 'x'): // Xextan
	{
		static const wchar_t xex_punct_within_word[] = { '\'' };
		tr->langopts.numbers = 0; 
		tr->langopts.lowercase_sentence = true;
		tr->punct_within_word = xex_punct_within_word;
}	
		break;
	case L3('s', 'h', 'n'):
		tr->langopts.tone_language = 1; // Tone language, use  CalcPitches_Tone() rather than CalcPitches()
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable
		tr->langopts.numbers = NUM_DEFAULT;
		tr->langopts.break_numbers = BREAK_INDIVIDUAL;
		break;
	case L3('c', 'm', 'n'): // no break, just go to 'zh' case
	case L3('y', 'u', 'e'):
	case L('z','h'):	// zh is used for backwards compatibility. Prefer cmn or yue.
	{
		static const short stress_lengths_zh[8] = { 230, 150, 230, 230, 230, 0, 240, 250 }; // 1=tone5. end-of-sentence, 6=tone 1&4, 7=tone 2&3
		static const unsigned char stress_amps_zh[] = { 22, 16, 22, 22, 22, 22, 22, 22 };

		SetupTranslator(tr, stress_lengths_zh, stress_amps_zh);

		tr->langopts.stress_rule = STRESSPOSN_1R; // stress on final syllable of a "word"
		tr->langopts.stress_flags = S_NO_DIM; // don't automatically set diminished stress (may be set in the intonation module)
		tr->langopts.vowel_pause = 0;
		tr->langopts.tone_language = 1; // Tone language, use  CalcPitches_Tone() rather than CalcPitches()
		tr->langopts.length_mods0 = tr->langopts.length_mods; // don't lengthen vowels in the last syllable
		tr->langopts.tone_numbers = 1; // a number after letters indicates a tone number (eg. pinyin or jyutping)
		tr->langopts.ideographs = 1;
		tr->langopts.our_alphabet = 0x3100;
		tr->langopts.word_gap = 0x21; // length of a final vowel is less dependent on the next consonant, don't merge consonant with next word
		tr->langopts.textmode = true;
		tr->langopts.listx = 1; // compile *_listx after *_list
		if (name2 == L3('y', 'u', 'e')) {
			tr->langopts.numbers = NUM_DEFAULT;
			tr->langopts.numbers2 = NUM2_ZERO_TENS;
			tr->langopts.break_numbers = BREAK_INDIVIDUAL;
		}
	}
		break;
	default:
		tr->langopts.param[LOPT_UNPRONOUNCABLE] = 1; // disable check for unpronouncable words
		break;
	}

	tr->translator_name = name2;

	ProcessLanguageOptions(&tr->langopts);
	return tr;
}

void ProcessLanguageOptions(LANGUAGE_OPTIONS *langopts)
{
	if (langopts->numbers & NUM_DECIMAL_COMMA) {
		// use . and ; for thousands and decimal separators
		langopts->thousands_sep = '.';
		langopts->decimal_sep = ',';
	}
	if (langopts->numbers & NUM_THOUS_SPACE)
		langopts->thousands_sep = 0; // don't allow thousands separator, except space
}

static void Translator_Russian(Translator *tr)
{
	static const unsigned char stress_amps_ru[] = { 16, 16, 18, 18, 20, 24, 24, 22 };
	static const short stress_lengths_ru[8] = { 150, 140, 220, 220, 0, 0, 260, 280 };
	static const char ru_ivowels[] = { 0x15, 0x18, 0x34, 0x37, 0 }; // add "е и є ї" to Y lettergroup (iotated vowels & soft-sign)

	SetupTranslator(tr, stress_lengths_ru, stress_amps_ru);
	SetCyrillicLetters(tr);
	SetLetterBits(tr, LETTERGP_Y, ru_ivowels);

	tr->langopts.param[LOPT_UNPRONOUNCABLE] = 0x432; // [v]  don't count this character at start of word
	tr->langopts.param[LOPT_REGRESSIVE_VOICING] = 0x03;
	tr->langopts.param[LOPT_REDUCE] = 2;
	tr->langopts.stress_rule = STRESSPOSN_SYLCOUNT;
	tr->langopts.stress_flags = S_NO_AUTO_2;

	tr->langopts.numbers = NUM_DECIMAL_COMMA | NUM_OMIT_1_HUNDRED;
	tr->langopts.numbers2 = NUM2_THOUSANDPLEX_VAR_THOUSANDS | NUM2_THOUSANDS_VAR1; // variant numbers before thousands
	tr->langopts.max_digits = 32;
	tr->langopts.max_initial_consonants = 5;
}		
