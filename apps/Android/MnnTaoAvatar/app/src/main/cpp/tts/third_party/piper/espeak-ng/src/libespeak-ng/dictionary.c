/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wctype.h>
#include <wchar.h>
#include <assert.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "common.h"                // for GetFileLength, strncpy0
#include "dictionary.h"
#include "numbers.h"                       // for LookupAccentedLetter, Look...
#include "phoneme.h"                       // for PHONEME_TAB, phVOWEL, phon...
#include "readclause.h"                    // for WordToString2
#include "speech.h"                        // for path_home
#include "compiledict.h"                   // for DecodeRule
#include "synthdata.h"                     // for PhonemeCode, InterpretPhoneme
#include "synthesize.h"                    // for STRESS_IS_PRIMARY, phoneme...
#include "translate.h"                     // for Translator, utf8_in, LANGU...

static int LookupFlags(Translator *tr, const char *word, unsigned int flags_out[2]);
static void DollarRule(char *word[], char *word_start, int consumed, int group_length, char word_buf[N_WORD_BYTES], Translator *tr, int command, int *failed, int *add_points);

typedef struct {
	int points;
	const char *phonemes;
	int end_type;
	char *del_fwd;
} MatchRecord;


int dictionary_skipwords;
char dictionary_name[40];

// accented characters which indicate (in some languages) the start of a separate syllable
static const unsigned short diereses_list[7] = { 0xe4, 0xeb, 0xef, 0xf6, 0xfc, 0xff, 0 };

// convert characters to an approximate 7 bit ascii equivalent
// used for checking for vowels (up to 0x259=schwa)
#define N_REMOVE_ACCENT  0x25e
static const unsigned char remove_accent[N_REMOVE_ACCENT] = {
	'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',  // 0c0
	'd', 'n', 'o', 'o', 'o', 'o', 'o',   0, 'o', 'u', 'u', 'u', 'u', 'y', 't', 's',  // 0d0
	'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',  // 0e0
	'd', 'n', 'o', 'o', 'o', 'o', 'o',   0, 'o', 'u', 'u', 'u', 'u', 'y', 't', 'y',  // 0f0

	'a', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd',  // 100
	'd', 'd', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'g', 'g', 'g', 'g',  // 110
	'g', 'g', 'g', 'g', 'h', 'h', 'h', 'h', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i',  // 120
	'i', 'i', 'i', 'i', 'j', 'j', 'k', 'k', 'k', 'l', 'l', 'l', 'l', 'l', 'l', 'l',  // 130
	'l', 'l', 'l', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'o', 'o', 'o', 'o',  // 140
	'o', 'o', 'o', 'o', 'r', 'r', 'r', 'r', 'r', 'r', 's', 's', 's', 's', 's', 's',  // 150
	's', 's', 't', 't', 't', 't', 't', 't', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u',  // 160
	'u', 'u', 'u', 'u', 'w', 'w', 'y', 'y', 'y', 'z', 'z', 'z', 'z', 'z', 'z', 's',  // 170
	'b', 'b', 'b', 'b',   0,   0, 'o', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'e', 'e',  // 180
	'e', 'f', 'f', 'g', 'g', 'h', 'i', 'i', 'k', 'k', 'l', 'l', 'm', 'n', 'n', 'o',  // 190
	'o', 'o', 'o', 'o', 'p', 'p', 'y',   0,   0, 's', 's', 't', 't', 't', 't', 'u',  // 1a0
	'u', 'u', 'v', 'y', 'y', 'z', 'z', 'z', 'z', 'z', 'z', 'z',   0,   0,   0, 'w',  // 1b0
	't', 't', 't', 'k', 'd', 'd', 'd', 'l', 'l', 'l', 'n', 'n', 'n', 'a', 'a', 'i',  // 1c0
	'i', 'o', 'o', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'e', 'a', 'a',  // 1d0
	'a', 'a', 'a', 'a', 'g', 'g', 'g', 'g', 'k', 'k', 'o', 'o', 'o', 'o', 'z', 'z',  // 1e0
	'j', 'd', 'd', 'd', 'g', 'g', 'w', 'w', 'n', 'n', 'a', 'a', 'a', 'a', 'o', 'o',  // 1f0

	'a', 'a', 'a', 'a', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i', 'o', 'o', 'o', 'o',  // 200
	'r', 'r', 'r', 'r', 'u', 'u', 'u', 'u', 's', 's', 't', 't', 'y', 'y', 'h', 'h',  // 210
	'n', 'd', 'o', 'o', 'z', 'z', 'a', 'a', 'e', 'e', 'o', 'o', 'o', 'o', 'o', 'o',  // 220
	'o', 'o', 'y', 'y', 'l', 'n', 't', 'j', 'd', 'q', 'a', 'c', 'c', 'l', 't', 's',  // 230
	'z',   0,   0, 'b', 'u', 'v', 'e', 'e', 'j', 'j', 'q', 'q', 'r', 'r', 'y', 'y',  // 240
	'a', 'a', 'a', 'b', 'o', 'c', 'd', 'd', 'e', 'e', 'e', 'e', 'e', 'e'
};

static int Reverse4Bytes(int word)
{
	// reverse the order of bytes from little-endian to big-endian
#ifdef ARCH_BIG
	int ix;
	int word2 = 0;

	for (ix = 0; ix <= 24; ix += 8) {
		word2 = word2 << 8;
		word2 |= (word >> ix) & 0xff;
	}
	return word2;
#else
	return word;
#endif
}

static void InitGroups(Translator *tr)
{
	// Called after dictionary 1 is loaded, to set up table of entry points for translation rule chains
	// for single-letters and two-letter combinations

	int ix;
	char *p;
	char *p_name;
	unsigned char c, c2;
	int len;

	tr->n_groups2 = 0;
	for (ix = 0; ix < 256; ix++) {
		tr->groups1[ix] = NULL;
		tr->groups2_count[ix] = 0;
		tr->groups2_start[ix] = 255; // indicates "not set"
	}
	memset(tr->letterGroups, 0, sizeof(tr->letterGroups));
	memset(tr->groups3, 0, sizeof(tr->groups3));

	p = tr->data_dictrules;
	// If there are no rules in the dictionary, compile_dictrules will not
	// write a RULE_GROUP_START (written in the for loop), but will write
	// a RULE_GROUP_END.
	if (*p != RULE_GROUP_END) while (*p != 0) {
		if (*p != RULE_GROUP_START) {
			fprintf(stderr, "Bad rules data in '%s_dict' at 0x%x (%c)\n", dictionary_name, (unsigned int)(p - tr->data_dictrules), *p);
			break;
		}
		p++;

		if (p[0] == RULE_REPLACEMENTS) {
			p = (char *)(((intptr_t)p+4) & ~3); // advance to next word boundary
			tr->langopts.replace_chars = (unsigned char *)p;

			while ( !is_str_totally_null(p, 4) ) {
				p++;
			}

			while (*p != RULE_GROUP_END) p++;
			p++;
			continue;
		}

		if (p[0] == RULE_LETTERGP2) {
			ix = p[1] - 'A';
			if (ix < 0)
				ix += 256;
			p += 2;
			if ((ix >= 0) && (ix < N_LETTER_GROUPS))
				tr->letterGroups[ix] = p;
		} else {
			len = strlen(p);
			p_name = p;
			c = p_name[0];
			c2 = p_name[1];

			p += (len+1);
			if (len == 1)
				tr->groups1[c] = p;
			else if (len == 0)
				tr->groups1[0] = p;
			else if (c == 1) {
				// index by offset from letter base
				tr->groups3[c2 - 1] = p;
			} else {
				if (tr->groups2_start[c] == 255)
					tr->groups2_start[c] = tr->n_groups2;

				tr->groups2_count[c]++;
				tr->groups2[tr->n_groups2] = p;
				tr->groups2_name[tr->n_groups2++] = (c + (c2 << 8));
			}
		}

		// skip over all the rules in this group
		while (*p != RULE_GROUP_END)
			p += (strlen(p) + 1);
		p++;
	}
}

int LoadDictionary(Translator *tr, const char *name, int no_error)
{
	int hash;
	char *p;
	int *pw;
	int length;
	FILE *f;
	int size;
	char fname[sizeof(path_home)+20];

	if (dictionary_name != name)
		strncpy(dictionary_name, name, 40); // currently loaded dictionary name
	if (tr->dictionary_name != name)
		strncpy(tr->dictionary_name, name, 40);

	// Load a pronunciation data file into memory
	// bytes 0-3:  offset to rules data
	// bytes 4-7:  number of hash table entries
	sprintf(fname, "%s%c%s_dict", path_home, PATHSEP, name);
	size = GetFileLength(fname);

	if (tr->data_dictlist != NULL) {
		free(tr->data_dictlist);
		tr->data_dictlist = NULL;
	}

	f = fopen(fname, "rb");
	if ((f == NULL) || (size <= 0)) {
		if (no_error == 0)
			fprintf(stderr, "Can't read dictionary file: '%s'\n", fname);
		if (f != NULL)
			fclose(f);
		return 1;
	}

	if ((tr->data_dictlist = malloc(size)) == NULL) {
		fclose(f);
		return 3;
	}
	size = fread(tr->data_dictlist, 1, size, f);
	fclose(f);

	pw = (int *)(tr->data_dictlist);
	length = Reverse4Bytes(pw[1]);

	if (size <= (N_HASH_DICT + sizeof(int)*2)) {
		fprintf(stderr, "Empty _dict file: '%s\n", fname);
		return 2;
	}

	if ((Reverse4Bytes(pw[0]) != N_HASH_DICT) ||
	    (length <= 0) || (length > 0x8000000)) {
		fprintf(stderr, "Bad data: '%s' (%x length=%x)\n", fname, Reverse4Bytes(pw[0]), length);
		return 2;
	}
	tr->data_dictrules = &(tr->data_dictlist[length]);

	// set up indices into data_dictrules
	InitGroups(tr);

	// set up hash table for data_dictlist
	p = &(tr->data_dictlist[8]);

	for (hash = 0; hash < N_HASH_DICT; hash++) {
		tr->dict_hashtab[hash] = p;
		while ((length = *(uint8_t *)p) != 0)
			p += length;
		p++; // skip over the zero which terminates the list for this hash value
	}

	if ((tr->dict_min_size > 0) && (size < (unsigned int)tr->dict_min_size))
		fprintf(stderr, "Full dictionary is not installed for '%s'\n", name);

	return 0;
}

/* Generate a hash code from the specified string
    This is used to access the dictionary_2 word-lookup dictionary
 */
int HashDictionary(const char *string)
{
	int c;
	int chars = 0;
	int hash = 0;

	while ((c = (*string++ & 0xff)) != 0) {
		hash = hash * 8 + c;
		hash = (hash & 0x3ff) ^ (hash >> 8); // exclusive or
		chars++;
	}

	return (hash+chars) & 0x3ff; // a 10 bit hash code
}

/* Translate a phoneme string from ascii mnemonics to internal phoneme numbers,
   from 'p' up to next blank .
   Returns advanced 'p'
   outptr contains encoded phonemes, unrecognized phoneme stops the encoding
   bad_phoneme must point to char array of length 2 of more
 */
const char *EncodePhonemes(const char *p, char *outptr, int *bad_phoneme)
{
	int ix;
	unsigned char c;
	int count;     // num. of matching characters
	int max;       // highest num. of matching found so far
	int max_ph;    // corresponding phoneme with highest matching
	int consumed;
	unsigned int mnemonic_word;

	if (bad_phoneme != NULL)
		*bad_phoneme = 0;

	// skip initial blanks
	while ((uint8_t)*p < 0x80 && isspace(*p))
		p++;

	while (((c = *p) != 0) && !isspace(c)) {
		consumed = 0;

		switch (c)
		{
		case '|':
			// used to separate phoneme mnemonics if needed, to prevent characters being treated
			// as a multi-letter mnemonic

			if ((c = p[1]) == '|') {
				// treat double || as a word-break symbol, drop through
				// to the default case with c = '|'
			} else {
				p++;
				break;
			}
		default:
			// lookup the phoneme mnemonic, find the phoneme with the highest number of
			// matching characters
			max = -1;
			max_ph = 0;

			for (ix = 1; ix < n_phoneme_tab; ix++) {
				if (phoneme_tab[ix] == NULL)
					continue;
				if (phoneme_tab[ix]->type == phINVALID)
					continue; // this phoneme is not defined for this language

				count = 0;
				mnemonic_word = phoneme_tab[ix]->mnemonic;

				while (((c = p[count]) > ' ') && (count < 4) &&
				       (c == ((mnemonic_word >> (count*8)) & 0xff)))
					count++;

				if ((count > max) &&
				    ((count == 4) || (((mnemonic_word >> (count*8)) & 0xff) == 0))) {
					max = count;
					max_ph = phoneme_tab[ix]->code;
				}
			}

			if (max_ph == 0) {
				// not recognised, report and ignore
				if (bad_phoneme != NULL)
					utf8_in(bad_phoneme, p);
				*outptr++ = 0;
				return p+1;
			}

			if (max <= 0)
				max = 1;
			p += (consumed + max);
			*outptr++ = (char)(max_ph);

			if (max_ph == phonSWITCH) {
				// Switch Language: this phoneme is followed by a text string
				char *p_lang = outptr;
				while (!isspace(c = *p) && (c != 0)) {
					p++;
					*outptr++ = tolower(c);
				}
				*outptr = 0;
				if (c == 0) {
					if (strcmp(p_lang, ESPEAKNG_DEFAULT_VOICE) == 0) {
						*p_lang = 0; // don't need ESPEAKNG_DEFAULT_VOICE, it's assumed by default
						return p;
					}
				} else
					*outptr++ = '|'; // more phonemes follow, terminate language string with separator
			}
			break;
		}
	}
	// terminate the encoded string
	*outptr = 0;
	return p;
}

void DecodePhonemes(const char *inptr, char *outptr)
{
	// Translate from internal phoneme codes into phoneme mnemonics
	unsigned char phcode;
	unsigned char c;
	unsigned int mnem;
	PHONEME_TAB *ph;
	static const char stress_chars[] = "==,,'*  ";

	sprintf(outptr, "* ");
	while ((phcode = *inptr++) > 0) {
		if (phcode == 255)
			continue; // indicates unrecognised phoneme
		if ((ph = phoneme_tab[phcode]) == NULL)
			continue;

		if ((ph->type == phSTRESS) && (ph->std_length <= 4) && (ph->program == 0)) {
			if (ph->std_length > 1)
				*outptr++ = stress_chars[ph->std_length];
		} else {
			mnem = ph->mnemonic;

			while ((c = (mnem & 0xff)) != 0) {
				*outptr++ = c;
				mnem = mnem >> 8;
			}
			if (phcode == phonSWITCH) {
				while (isalpha(*inptr))
					*outptr++ = *inptr++;
			}
		}
	}
	*outptr = 0; // string terminator
}

// using Kirschenbaum to IPA translation, ascii 0x20 to 0x7f
static const unsigned short ipa1[96] = {
	0x20,  0x21,  0x22,  0x2b0, 0x24,  0x25,  0x0e6, 0x2c8, 0x28,  0x29,  0x27e, 0x2b,  0x2cc, 0x2d,  0x2e,  0x2f,
	0x252, 0x31,  0x32,  0x25c, 0x34,  0x35,  0x36,  0x37,  0x275, 0x39,  0x2d0, 0x2b2, 0x3c,  0x3d,  0x3e,  0x294,
	0x259, 0x251, 0x3b2, 0xe7,  0xf0,  0x25b, 0x46,  0x262, 0x127, 0x26a, 0x25f, 0x4b,  0x26b, 0x271, 0x14b, 0x254,
	0x3a6, 0x263, 0x280, 0x283, 0x3b8, 0x28a, 0x28c, 0x153, 0x3c7, 0xf8,  0x292, 0x32a, 0x5c,  0x5d,  0x5e,  0x5f,
	0x60,  0x61,  0x62,  0x63,  0x64,  0x65,  0x66,  0x261, 0x68,  0x69,  0x6a,  0x6b,  0x6c,  0x6d,  0x6e,  0x6f,
	0x70,  0x71,  0x72,  0x73,  0x74,  0x75,  0x76,  0x77,  0x78,  0x79,  0x7a,  0x7b,  0x7c,  0x7d,  0x303, 0x7f
};

#define N_PHON_OUT  500  // realloc increment
static char *phon_out_buf = NULL;   // passes the result of GetTranslatedPhonemeString()
static unsigned int phon_out_size = 0;

char *WritePhMnemonic(char *phon_out, PHONEME_TAB *ph, PHONEME_LIST *plist, int use_ipa, int *flags)
{
	int c;
	int mnem;
	int len;
	bool first;
	int ix = 0;
	char *p;
	PHONEME_DATA phdata;

	if (ph->code == phonEND_WORD) {
		// ignore
		phon_out[0] = 0;
		return phon_out;
	}

	if (ph->code == phonSWITCH) {
		// the tone_ph field contains a phoneme table number
		p = phoneme_tab_list[plist->tone_ph].name;
		sprintf(phon_out, "(%s)", p);
		return phon_out + strlen(phon_out);
	}

	if (use_ipa) {
		// has an ipa name been defined for this phoneme ?
		phdata.ipa_string[0] = 0;

		if (plist == NULL)
			InterpretPhoneme2(ph->code, &phdata);
		else
			InterpretPhoneme(NULL, 0, plist, plist, &phdata, NULL);

		p = phdata.ipa_string;
		if (*p == 0x20) {
			// indicates no name for this phoneme
			*phon_out = 0;
			return phon_out;
		}
		if ((*p != 0) && ((*p & 0xff) < 0x20)) {
			// name starts with a flags byte
			if (flags != NULL)
				*flags = *p;
			p++;
		}

		len = strlen(p);
		if (len > 0) {
			strcpy(phon_out, p);
			phon_out += len;
			*phon_out = 0;
			return phon_out;
		}
	}

	first = true;
	for (mnem = ph->mnemonic; (c = mnem & 0xff) != 0; mnem = mnem >> 8) {
		if (c == '/')
			break; // discard phoneme variant indicator

		if (use_ipa) {
			// convert from ascii to ipa
			if (first && (c == '_'))
				break; // don't show pause phonemes

			if ((c == '#') && (ph->type == phVOWEL))
				break; // # is subscript-h, but only for consonants

			// ignore digits after the first character
			if (!first && IsDigit09(c))
				continue;

			if ((c >= 0x20) && (c < 128))
				c = ipa1[c-0x20];

			ix += utf8_out(c, &phon_out[ix]);
		} else
			phon_out[ix++] = c;
		first = false;
	}

	phon_out = &phon_out[ix];
	*phon_out = 0;
	return phon_out;
}

//// Extension: write phone mnemonic with stress
char *WritePhMnemonicWithStress(char *phon_out, PHONEME_TAB *ph, PHONEME_LIST *plist, int use_ipa, int *flags) {
	if (plist->synthflags & SFLAG_SYLLABLE) {
		unsigned char stress = plist->stresslevel;

		if (stress > 1) {
			int c = 0;

			if (stress > STRESS_IS_PRIORITY) {
				stress = STRESS_IS_PRIORITY;
			}

			if (use_ipa) {
				c = 0x2cc; // ipa, secondary stress

				if (stress > STRESS_IS_SECONDARY) {
					c = 0x02c8; // ipa, primary stress
				}
			} else {
				const char stress_chars[] = "==,,''";

				c = stress_chars[stress];
			}

			if (c != 0) {
				phon_out += utf8_out(c, phon_out);
			}
		}
	}

	return WritePhMnemonic(phon_out, ph, plist, use_ipa, flags);
}
////

const char *GetTranslatedPhonemeString(int phoneme_mode)
{
	/* Called after a clause has been translated into phonemes, in order
	   to display the clause in phoneme mnemonic form.

	   phoneme_mode
	                 bit  1:   use IPA phoneme names
	                 bit  7:   use tie between letters in multi-character phoneme names
	                 bits 8-23 tie or separator character

	 */

	int ix;
	unsigned int len;
	int phon_out_ix = 0;
	int stress;
	int c;
	char *p;
	char *buf;
	int count;
	int flags;
	int use_ipa;
	int use_tie;
	int separate_phonemes;
	char phon_buf[30];
	char phon_buf2[30];
	PHONEME_LIST *plist;

	static const char stress_chars[] = "==,,''";

	if (phon_out_buf == NULL) {
		phon_out_size = N_PHON_OUT;
		if ((phon_out_buf = (char *)malloc(phon_out_size)) == NULL) {
			phon_out_size = 0;
			return "";
		}
	}

	use_ipa = phoneme_mode & espeakPHONEMES_IPA;
	if (phoneme_mode & espeakPHONEMES_TIE) {
		use_tie = phoneme_mode >> 8;
		separate_phonemes = 0;
	} else {
		separate_phonemes = phoneme_mode >> 8;
		use_tie = 0;
	}

	for (ix = 1; ix < (n_phoneme_list-2); ix++) {
		buf = phon_buf;

		plist = &phoneme_list[ix];

		WritePhMnemonic(phon_buf2, plist->ph, plist, use_ipa, &flags);
		if (plist->newword & PHLIST_START_OF_WORD && !(plist->newword & (PHLIST_START_OF_SENTENCE | PHLIST_START_OF_CLAUSE)))
			*buf++ = ' ';

		if ((!plist->newword) || (separate_phonemes == ' ')) {
			if ((separate_phonemes != 0) && (ix > 1)) {
				utf8_in(&c, phon_buf2);
				if ((c < 0x2b0) || (c > 0x36f)) // not if the phoneme starts with a superscript letter
					buf += utf8_out(separate_phonemes, buf);
			}
		}

		if (plist->synthflags & SFLAG_SYLLABLE) {
			if ((stress = plist->stresslevel) > 1) {
				c = 0;
				if (stress > STRESS_IS_PRIORITY) stress = STRESS_IS_PRIORITY;

				if (use_ipa) {
					c = 0x2cc; // ipa, secondary stress
					if (stress > STRESS_IS_SECONDARY)
						c = 0x02c8; // ipa, primary stress
				} else
					c = stress_chars[stress];

				if (c != 0)
					buf += utf8_out(c, buf);
			}
		}

		flags = 0;
		count = 0;
		for (p = phon_buf2; *p != 0;) {
			p += utf8_in(&c, p);
			if (use_tie != 0) {
				// look for non-initial alphabetic character, but not diacritic, superscript etc.
				if ((count > 0) && !(flags & (1 << (count-1))) && ((c < 0x2b0) || (c > 0x36f)) && iswalpha(c))
					buf += utf8_out(use_tie, buf);
			}
			buf += utf8_out(c, buf);
			count++;
		}

		if (plist->ph->code != phonSWITCH) {
			if (plist->synthflags & SFLAG_LENGTHEN)
				buf = WritePhMnemonic(buf, phoneme_tab[phonLENGTHEN], plist, use_ipa, NULL);
			if ((plist->synthflags & SFLAG_SYLLABLE) && (plist->type != phVOWEL)) {
				// syllablic consonant
				buf = WritePhMnemonic(buf, phoneme_tab[phonSYLLABIC], plist, use_ipa, NULL);
			}
			if (plist->tone_ph > 0)
				buf = WritePhMnemonic(buf, phoneme_tab[plist->tone_ph], plist, use_ipa, NULL);
		}

		len = buf - phon_buf;
		if ((phon_out_ix + len) >= phon_out_size) {
			// enlarge the phoneme buffer
			phon_out_size = phon_out_ix + len + N_PHON_OUT;
			char *new_phon_out_buf = (char *)realloc(phon_out_buf, phon_out_size);
			if (new_phon_out_buf == NULL) {
				phon_out_size = 0;
				return "";
			} else
				phon_out_buf = new_phon_out_buf;
		}

		phon_buf[len] = 0;
		strcpy(&phon_out_buf[phon_out_ix], phon_buf);
		phon_out_ix += len;
	}

	if (!phon_out_buf)
		return "";

	phon_out_buf[phon_out_ix] = 0;

	return phon_out_buf;
}

static int LetterGroupNo(char *rule)
{
	/*
	 * Returns number of letter group
	 */
	int groupNo = *rule;
	groupNo = groupNo - 'A'; // subtracting 'A' makes letter_group equal to number in .Lxx definition
	if (groupNo < 0)         // fix sign if necessary
		groupNo += 256;
	return groupNo;
}

static int IsLetterGroup(Translator *tr, char *word, int group, int pre)
{
	/* Match the word against a list of utf-8 strings.
	 * returns length of matching letter group or -1
	 *
	 * How this works:
	 *
	 *       +-+
	 *       |c|<-(tr->letterGroups[group])
	 *       |0|
	 *   *p->|c|<-len+              +-+
	 *       |s|<----+              |a|<-(Actual word to be tested)
	 *       |0|            *word-> |t|<-*w=word-len+1 (for pre-rule)
	 *       |~|                    |a|<-*w=word       (for post-rule)
	 *       |7|                    |s|
	 *       +-+                    +-+
	 *
	 *     7=RULE_GROUP_END
	 *     0=null terminator
	 *     pre==1 — pre-rule
	 *     pre==0 — post-rule
	 */
	char *p; // group counter
	char *w; // word counter
	int len = 0, i;

	p = tr->letterGroups[group];
	if (p == NULL)
		return -1;

	while (*p != RULE_GROUP_END) {
		// If '~' (no character) is allowed in group, return 0.
		if (*p == '~')
			return 0;

		if (pre) {
			len = strlen(p);
			w = word;
			if (*w == 0)
				goto skip;
			for (i = 0; i < len-1; i++)
			{
				w--;
				if (*w == 0)
					// Not found, skip the rest of this group.
					goto skip;
			}
		} else
			w = word;

		//  Check current group
		while ((*p == *w) && (*w != 0)) {
			w++;
			p++;
		}
		if (*p == 0) { // Matched the current group.
			if (pre)
				return len;
			return w - word;
		}

		// No match, so skip the rest of this group.
skip:
		while (*p++ != 0)
			;
	}
	// Not found
	return -1;
}

static int IsLetter(Translator *tr, int letter, int group)
{
	int letter2;

	if (tr->letter_groups[group] != NULL) {
		if (wcschr(tr->letter_groups[group], letter))
			return 1;
		return 0;
	}

	if (group > 7)
		return 0;

	if (tr->letter_bits_offset > 0) {
		if (((letter2 = (letter - tr->letter_bits_offset)) > 0) && (letter2 < 0x100))
			letter = letter2;
		else
			return 0;
	} else if ((letter >= 0xc0) && (letter < N_REMOVE_ACCENT))
		return tr->letter_bits[remove_accent[letter-0xc0]] & (1L << group);

	if ((letter >= 0) && (letter < 0x100))
		return tr->letter_bits[letter] & (1L << group);

	return 0;
}

int IsVowel(Translator *tr, int letter)
{
	return IsLetter(tr, letter, LETTERGP_VOWEL2);
}

int GetVowelStress(Translator *tr, unsigned char *phonemes, signed char *vowel_stress, int *vowel_count, int *stressed_syllable, int control)
{
	// control = 1, set stress to 1 for forced unstressed vowels
	unsigned char phcode;
	PHONEME_TAB *ph;
	unsigned char *ph_out = phonemes;
	int count = 1;
	int max_stress = -1;
	int ix;
	int j;
	int stress = -1;
	int primary_posn = 0;

	vowel_stress[0] = STRESS_IS_UNSTRESSED;
	while (((phcode = *phonemes++) != 0) && (count < (N_WORD_PHONEMES/2)-1)) {
		if ((ph = phoneme_tab[phcode]) == NULL)
			continue;

		if ((ph->type == phSTRESS) && (ph->program == 0)) {
			// stress marker, use this for the following vowel

			if (phcode == phonSTRESS_PREV) {
				// primary stress on preceding vowel
				j = count - 1;
				while ((j > 0) && (*stressed_syllable == 0) && (vowel_stress[j] < STRESS_IS_PRIMARY)) {
					if ((vowel_stress[j] != STRESS_IS_DIMINISHED) && (vowel_stress[j] != STRESS_IS_UNSTRESSED)) {
						// don't promote a phoneme which must be unstressed
						vowel_stress[j] = STRESS_IS_PRIMARY;

						if (max_stress < STRESS_IS_PRIMARY) {
							max_stress = STRESS_IS_PRIMARY;
							primary_posn = j;
						}

						/* reduce any preceding primary stress markers */
						for (ix = 1; ix < j; ix++) {
							if (vowel_stress[ix] == STRESS_IS_PRIMARY)
								vowel_stress[ix] = STRESS_IS_SECONDARY;
						}
						break;
					}
					j--;
				}
			} else {
				if ((ph->std_length < 4) || (*stressed_syllable == 0)) {
					stress = ph->std_length;

					if (stress > max_stress)
						max_stress = stress;
				}
			}
			continue;
		}

		if ((ph->type == phVOWEL) && !(ph->phflags & phNONSYLLABIC)) {
			vowel_stress[count] = (char)stress;
			if ((stress >= STRESS_IS_PRIMARY) && (stress >= max_stress)) {
				primary_posn = count;
				max_stress = stress;
			}

			if ((stress < 0) && (control & 1) && (ph->phflags & phUNSTRESSED))
				vowel_stress[count] = STRESS_IS_UNSTRESSED; // weak vowel, must be unstressed

			count++;
			stress = -1;
		} else if (phcode == phonSYLLABIC) {
			// previous consonant phoneme is syllablic
			vowel_stress[count] = (char)stress;
			if ((stress < 0) && (control & 1))
				vowel_stress[count] = STRESS_IS_UNSTRESSED; // syllabic consonant, usually unstressed
			count++;
		}

		*ph_out++ = phcode;
	}
	vowel_stress[count] = STRESS_IS_UNSTRESSED;
	*ph_out = 0;

	// has the position of the primary stress been specified by $1, $2, etc?
	if (*stressed_syllable > 0) {
		if (*stressed_syllable >= count)
			*stressed_syllable = count-1; // the final syllable

		vowel_stress[*stressed_syllable] = STRESS_IS_PRIMARY;
		max_stress = STRESS_IS_PRIMARY;
		primary_posn = *stressed_syllable;
	}

	if (max_stress == STRESS_IS_PRIORITY) {
		// priority stress, replaces any other primary stress marker
		for (ix = 1; ix < count; ix++) {
			if (vowel_stress[ix] == STRESS_IS_PRIMARY) {
				if (tr->langopts.stress_flags & S_PRIORITY_STRESS)
					vowel_stress[ix] = STRESS_IS_UNSTRESSED;
				else
					vowel_stress[ix] = STRESS_IS_SECONDARY;
			}

			if (vowel_stress[ix] == STRESS_IS_PRIORITY) {
				vowel_stress[ix] = STRESS_IS_PRIMARY;
				primary_posn = ix;
			}
		}
		max_stress = STRESS_IS_PRIMARY;
	}

	*stressed_syllable = primary_posn;
	*vowel_count = count;
	return max_stress;
}

const char stress_phonemes[] = {
	phonSTRESS_D, phonSTRESS_U, phonSTRESS_2, phonSTRESS_3,
	phonSTRESS_P, phonSTRESS_P2, phonSTRESS_TONIC
};

void SetWordStress(Translator *tr, char *output, unsigned int *dictionary_flags, int tonic, int control)
{
	/* Guess stress pattern of word.  This is language specific

	   'output' is used for input and output

	   'dictionary_flags' has bits 0-3   position of stressed vowel (if > 0)
	                                     or unstressed (if == 7) or syllables 1 and 2 (if == 6)
	                          bits 8...  dictionary flags

	   If 'tonic' is set (>= 0), replace highest stress by this value.

	   control:  bit 0   This is an individual symbol, not a word
	            bit 1   Suffix phonemes are still to be added
	 */

	unsigned char phcode;
	unsigned char *p;
	PHONEME_TAB *ph;
	int stress;
	int max_stress;
	int max_stress_input; // any stress specified in the input?
	int vowel_count; // num of vowels + 1
	int ix;
	int v;
	int v_stress;
	int stressed_syllable; // position of stressed syllable
	int max_stress_posn;
	char *max_output;
	int final_ph;
	int final_ph2;
	int mnem;
	int opt_length;
	int stressflags;
	int dflags = 0;
	int first_primary;
	int long_vowel;

	signed char vowel_stress[N_WORD_PHONEMES/2];
	char syllable_weight[N_WORD_PHONEMES/2];
	char vowel_length[N_WORD_PHONEMES/2];
	unsigned char phonetic[N_WORD_PHONEMES];

	static const char consonant_types[16] = { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 };

	memset(syllable_weight, 0, sizeof(syllable_weight));
	memset(vowel_length, 0, sizeof(vowel_length));

	stressflags = tr->langopts.stress_flags;

	if (dictionary_flags != NULL)
		dflags = dictionary_flags[0];

	// copy input string into internal buffer
	for (ix = 0; ix < N_WORD_PHONEMES; ix++) {
		phonetic[ix] = output[ix];
		// check for unknown phoneme codes
		if (phonetic[ix] >= n_phoneme_tab)
			phonetic[ix] = phonSCHWA;
		if (phonetic[ix] == 0)
			break;
	}
	if (ix == 0) return;
	final_ph = phonetic[ix-1];
	final_ph2 = phonetic[(ix > 1) ? ix-2 : ix-1];

	max_output = output + (N_WORD_PHONEMES-3); // check for overrun


	// any stress position marked in the xx_list dictionary ?
	bool unstressed_word = false;
	stressed_syllable = dflags & 0x7;
	if (dflags & 0x8) {
		// this indicates a word without a primary stress
		stressed_syllable = dflags & 0x3;
		unstressed_word = true;
	}

	max_stress = max_stress_input = GetVowelStress(tr, phonetic, vowel_stress, &vowel_count, &stressed_syllable, 1);
	if ((max_stress < 0) && dictionary_flags)
		max_stress = STRESS_IS_DIMINISHED;

	// heavy or light syllables
	ix = 1;
	for (p = phonetic; *p != 0; p++) {
		if ((phoneme_tab[p[0]]->type == phVOWEL) && !(phoneme_tab[p[0]]->phflags & phNONSYLLABIC)) {
			int weight = 0;
			bool lengthened = false;

			if (phoneme_tab[p[1]]->code == phonLENGTHEN)
				lengthened = true;

			if (lengthened || (phoneme_tab[p[0]]->phflags & phLONG)) {
				// long vowel, increase syllable weight
				weight++;
			}
			vowel_length[ix] = weight;

			if (lengthened) p++; // advance over phonLENGTHEN

			if (consonant_types[phoneme_tab[p[1]]->type] && ((phoneme_tab[p[2]]->type != phVOWEL) || (phoneme_tab[p[1]]->phflags & phLONG))) {
				// followed by two consonants, a long consonant, or consonant and end-of-word
				weight++;
			}
			syllable_weight[ix] = weight;
			ix++;
		}
	}

	switch (tr->langopts.stress_rule)
	{
	case STRESSPOSN_2LLH:
		// stress on first syllable, unless it is a light syllable followed by a heavy syllable
		if ((syllable_weight[1] > 0) || (syllable_weight[2] == 0))
			break;
		// fallthrough:
	case STRESSPOSN_2L:
		// stress on second syllable
		if ((stressed_syllable == 0) && (vowel_count > 2)) {
			stressed_syllable = 2;
			if (max_stress == STRESS_IS_DIMINISHED)
				vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
		}
		break;

	case STRESSPOSN_2R:
		// a language with stress on penultimate vowel

		if (stressed_syllable == 0) {
			// no explicit stress - stress the penultimate vowel
			max_stress = STRESS_IS_PRIMARY;

			if (vowel_count > 2) {
				stressed_syllable = vowel_count - 2;

				if (stressflags & S_FINAL_SPANISH) {
					// LANG=Spanish, stress on last vowel if the word ends in a consonant other than 'n' or 's'
					if (phoneme_tab[final_ph]->type != phVOWEL) {
						mnem = phoneme_tab[final_ph]->mnemonic;

						if ((tr->translator_name == L('a', 'n')) || (tr->translator_name == L('c', 'a'))) {
							if (((mnem != 's') && (mnem != 'n')) || phoneme_tab[final_ph2]->type != phVOWEL)
								stressed_syllable = vowel_count - 1; // stress on last syllable
						} else if (tr->translator_name == L('i', 'a')) {
							if ((mnem != 's') || phoneme_tab[final_ph2]->type != phVOWEL)
								stressed_syllable = vowel_count - 1; // stress on last syllable
						} else {
							if ((mnem == 's') && (phoneme_tab[final_ph2]->type == phNASAL)) {
								// -ns  stress remains on penultimate syllable
							} else if (((phoneme_tab[final_ph]->type != phNASAL) && (mnem != 's')) || (phoneme_tab[final_ph2]->type != phVOWEL))
								stressed_syllable = vowel_count - 1;
						}
					}
				}

				if (stressflags & S_FINAL_LONG) {
					// stress on last syllable if it has a long vowel, but previous syllable has a short vowel
					if (vowel_length[vowel_count - 1] > vowel_length[vowel_count - 2])
						stressed_syllable = vowel_count - 1;
				}

				if ((vowel_stress[stressed_syllable] == STRESS_IS_DIMINISHED) || (vowel_stress[stressed_syllable] == STRESS_IS_UNSTRESSED)) {
					// but this vowel is explicitly marked as unstressed
					if (stressed_syllable > 1)
						stressed_syllable--;
					else
						stressed_syllable++;
				}
			} else
				stressed_syllable = 1;

			// only set the stress if it's not already marked explicitly
			if (vowel_stress[stressed_syllable] < 0) {
				// don't stress if next and prev syllables are stressed
				if ((vowel_stress[stressed_syllable-1] < STRESS_IS_PRIMARY) || (vowel_stress[stressed_syllable+1] < STRESS_IS_PRIMARY))
					vowel_stress[stressed_syllable] = max_stress;
			}
		}
		break;
	case STRESSPOSN_1R:
		// stress on last vowel
		if (stressed_syllable == 0) {
			// no explicit stress - stress the final vowel
			stressed_syllable = vowel_count - 1;

			while (stressed_syllable > 0) {
				// find the last vowel which is not unstressed
				if (vowel_stress[stressed_syllable] < STRESS_IS_DIMINISHED) {
					vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
					break;
				} else
					stressed_syllable--;
			}
			max_stress = STRESS_IS_PRIMARY;
		}
		break;
	case  STRESSPOSN_3R: // stress on antipenultimate vowel
		if (stressed_syllable == 0) {
			stressed_syllable = vowel_count - 3;
			if (stressed_syllable < 1)
				stressed_syllable = 1;

			if (max_stress == STRESS_IS_DIMINISHED)
				vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
		}
		break;
	case STRESSPOSN_SYLCOUNT:
		// LANG=Russian
		if (stressed_syllable == 0) {
			// no explicit stress - guess the stress from the number of syllables
			static const char guess_ru[16] =   { 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11 };
			static const char guess_ru_v[16] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10 }; // for final phoneme is a vowel
			static const char guess_ru_t[16] = { 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10 }; // for final phoneme is an unvoiced stop

			stressed_syllable = vowel_count - 3;
			if (vowel_count < 16) {
				if (phoneme_tab[final_ph]->type == phVOWEL)
					stressed_syllable = guess_ru_v[vowel_count];
				else if (phoneme_tab[final_ph]->type == phSTOP)
					stressed_syllable = guess_ru_t[vowel_count];
				else
					stressed_syllable = guess_ru[vowel_count];
			}
			vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
		}
		break;
	case STRESSPOSN_1RH: // LANG=hi stress on the last heaviest syllable
		if (stressed_syllable == 0) {
			int wt;
			int max_weight = -1;

			// find the heaviest syllable, excluding the final syllable
			for (ix = 1; ix < (vowel_count-1); ix++) {
				if (vowel_stress[ix] < STRESS_IS_DIMINISHED) {
					if ((wt = syllable_weight[ix]) >= max_weight) {
						max_weight = wt;
						stressed_syllable = ix;
					}
				}
			}

			if ((syllable_weight[vowel_count-1] == 2) &&  (max_weight < 2)) {
				// the only double=heavy syllable is the final syllable, so stress this
				stressed_syllable = vowel_count-1;
			} else if (max_weight <= 0) {
				// all syllables, exclusing the last, are light. Stress the first syllable
				stressed_syllable = 1;
			}

			vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
		}
		break;
	case STRESSPOSN_1RU : // LANG=tr, the last syllable for any vowel marked explicitly as unstressed
		if (stressed_syllable == 0) {
			stressed_syllable = vowel_count - 1;
			for (ix = 1; ix < vowel_count; ix++) {
				if (vowel_stress[ix] == STRESS_IS_UNSTRESSED) {
					stressed_syllable = ix-1;
					break;
				}
			}
			vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
		}
		break;
	case STRESSPOSN_ALL: // mark all as stressed
		for (ix = 1; ix < vowel_count; ix++) {
			if (vowel_stress[ix] < STRESS_IS_DIMINISHED)
				vowel_stress[ix] = STRESS_IS_PRIMARY;
		}
		break;
	case STRESSPOSN_GREENLANDIC: // LANG=kl (Greenlandic)
		long_vowel = 0;
		for (ix = 1; ix < vowel_count; ix++) {
			if (vowel_stress[ix] == STRESS_IS_PRIMARY)
				vowel_stress[ix] = STRESS_IS_SECONDARY; // change marked stress (consonant clusters) to secondary (except the last)

			if (vowel_length[ix] > 0) {
				long_vowel = ix;
				vowel_stress[ix] = STRESS_IS_SECONDARY; // give secondary stress to all long vowels
			}
		}

		// 'stressed_syllable' gives the last marked stress
		if (stressed_syllable == 0) {
			// no marked stress, choose the last long vowel
			if (long_vowel > 0)
				stressed_syllable = long_vowel;
			else {
				// no long vowels or consonant clusters
				if (vowel_count > 5)
					stressed_syllable = vowel_count - 3; // more than 4 syllables
				else
					stressed_syllable = vowel_count - 1;
			}
		}
		vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
		max_stress = STRESS_IS_PRIMARY;
		break;
	case STRESSPOSN_1SL:  // LANG=ml, 1st unless 1st vowel is short and 2nd is long
		if (stressed_syllable == 0) {
			stressed_syllable = 1;
			if ((vowel_length[1] == 0) && (vowel_count > 2) && (vowel_length[2] > 0))
				stressed_syllable = 2;
			vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
		}
		break;

	case STRESSPOSN_EU: // LANG=eu. If more than 2 syllables: primary stress in second syllable and secondary on last.
		if ((stressed_syllable == 0) && (vowel_count > 2)) {
			for (ix = 1; ix < vowel_count; ix++) {
				vowel_stress[ix] = STRESS_IS_DIMINISHED;
			}
			stressed_syllable = 2;
			if (max_stress == STRESS_IS_DIMINISHED)
				vowel_stress[stressed_syllable] = STRESS_IS_PRIMARY;
			max_stress = STRESS_IS_PRIMARY;
			if (vowel_count > 3) {
				vowel_stress[vowel_count - 1] = STRESS_IS_SECONDARY;
			}
		}
		break;
	}

	if ((stressflags & S_FINAL_VOWEL_UNSTRESSED) && ((control & 2) == 0) && (vowel_count > 2) && (max_stress_input < STRESS_IS_SECONDARY) && (vowel_stress[vowel_count - 1] == STRESS_IS_PRIMARY)) {
		// Don't allow stress on a word-final vowel
		// Only do this if there is no suffix phonemes to be added, and if a stress position was not given explicitly
		if (phoneme_tab[final_ph]->type == phVOWEL) {
			vowel_stress[vowel_count - 1] = STRESS_IS_UNSTRESSED;
			vowel_stress[vowel_count - 2] = STRESS_IS_PRIMARY;
		}
	}

	// now guess the complete stress pattern
	if (max_stress < STRESS_IS_PRIMARY)
		stress = STRESS_IS_PRIMARY; // no primary stress marked, use for 1st syllable
	else
		stress = STRESS_IS_SECONDARY;

	if (unstressed_word == false) {
		if ((stressflags & S_2_SYL_2) && (vowel_count == 3)) {
			// Two syllable word, if one syllable has primary stress, then give the other secondary stress
			if (vowel_stress[1] == STRESS_IS_PRIMARY)
				vowel_stress[2] = STRESS_IS_SECONDARY;
			if (vowel_stress[2] == STRESS_IS_PRIMARY)
				vowel_stress[1] = STRESS_IS_SECONDARY;
		}

		if ((stressflags & S_INITIAL_2) && (vowel_stress[1] < STRESS_IS_DIMINISHED)) {
			// If there is only one syllable before the primary stress, give it a secondary stress
			if ((vowel_count > 3) && (vowel_stress[2] >= STRESS_IS_PRIMARY))
				vowel_stress[1] = STRESS_IS_SECONDARY;
		}
	}

	bool done = false;
	first_primary = 0;
	for (v = 1; v < vowel_count; v++) {
		if (vowel_stress[v] < STRESS_IS_DIMINISHED) {
			if ((stressflags & S_FINAL_NO_2) && (stress < STRESS_IS_PRIMARY) && (v == vowel_count-1)) {
				// flag: don't give secondary stress to final vowel
			} else if ((stressflags & 0x8000) && (done == false)) {
				vowel_stress[v] = (char)stress;
				done = true;
				stress = STRESS_IS_SECONDARY; // use secondary stress for remaining syllables
			} else if ((vowel_stress[v-1] <= STRESS_IS_UNSTRESSED) && ((vowel_stress[v+1] <= STRESS_IS_UNSTRESSED) || ((stress == STRESS_IS_PRIMARY) && (vowel_stress[v+1] <= STRESS_IS_NOT_STRESSED)))) {
				// trochaic: give stress to vowel surrounded by unstressed vowels

				if ((stress == STRESS_IS_SECONDARY) && (stressflags & S_NO_AUTO_2))
					continue; // don't use secondary stress

				// don't put secondary stress on a light syllable if the rest of the word (excluding last syllable) contains a heavy syllable
				if ((v > 1) && (stressflags & S_2_TO_HEAVY) && (syllable_weight[v] == 0)) {
					bool skip = false;
					for (int i = v; i < vowel_count - 1; i++) {
						if (syllable_weight[i] > 0) {
							skip = true;
							break;
						}
					}
					if (skip == true)
						continue;
				}

				if ((v > 1) && (stressflags & S_2_TO_HEAVY) && (syllable_weight[v] == 0) && (syllable_weight[v+1] > 0)) {
					// don't put secondary stress on a light syllable which is followed by a heavy syllable
					continue;
				}

				// should start with secondary stress on the first syllable, or should it count back from
				// the primary stress and put secondary stress on alternate syllables?
				vowel_stress[v] = (char)stress;
				done = true;
				stress = STRESS_IS_SECONDARY; // use secondary stress for remaining syllables
			}
		}

		if (vowel_stress[v] >= STRESS_IS_PRIMARY) {
			if (first_primary == 0)
				first_primary = v;
			else if (stressflags & S_FIRST_PRIMARY) {
				// reduce primary stresses after the first to secondary
				vowel_stress[v] = STRESS_IS_SECONDARY;
			}
		}
	}

	if ((unstressed_word) && (tonic < 0)) {
		if (vowel_count <= 2)
			tonic = tr->langopts.unstressed_wd1; // monosyllable - unstressed
		else
			tonic = tr->langopts.unstressed_wd2; // more than one syllable, used secondary stress as the main stress
	}

	max_stress = STRESS_IS_DIMINISHED;
	max_stress_posn = 0;
	for (v = 1; v < vowel_count; v++) {
		if (vowel_stress[v] >= max_stress) {
			max_stress = vowel_stress[v];
			max_stress_posn = v;
		}
	}

	if (tonic >= 0) {
		// find position of highest stress, and replace it by 'tonic'

		// don't disturb an explicitly set stress by 'unstress-at-end' flag
		if ((tonic > max_stress) || (max_stress <= STRESS_IS_PRIMARY))
			vowel_stress[max_stress_posn] = (char)tonic;
		max_stress = tonic;
	}

	// produce output phoneme string
	p = phonetic;
	v = 1;

	if (!(control & 1) && ((ph = phoneme_tab[*p]) != NULL)) {
		while ((ph->type == phSTRESS) || (*p == phonEND_WORD)) {
			p++;
			ph = phoneme_tab[p[0]];
		}

		if ((tr->langopts.vowel_pause & 0x30) && (ph->type == phVOWEL)) {
			// word starts with a vowel

			if ((tr->langopts.vowel_pause & 0x20) && (vowel_stress[1] >= STRESS_IS_PRIMARY))
				*output++ = phonPAUSE_NOLINK; // not to be replaced by link
			else
				*output++ = phonPAUSE_VSHORT; // break, but no pause
		}
	}

	p = phonetic;
	/* Note: v progression has to strictly follow the vowel_stress production in GetVowelStress */
	while (((phcode = *p++) != 0) && (output < max_output)) {
		if ((ph = phoneme_tab[phcode]) == NULL)
			continue;

		if (ph->type == phPAUSE)
			tr->prev_last_stress = 0;
		else if (((ph->type == phVOWEL) && !(ph->phflags & phNONSYLLABIC)) || (*p == phonSYLLABIC)) {
			// a vowel, or a consonant followed by a syllabic consonant marker

			assert(v <= vowel_count);

			v_stress = vowel_stress[v];
			tr->prev_last_stress = v_stress;

			if (v_stress <= STRESS_IS_UNSTRESSED) {
				if ((v > 1) && (max_stress >= 2) && (stressflags & S_FINAL_DIM) && (v == (vowel_count-1))) {
					// option: mark unstressed final syllable as diminished
					v_stress = STRESS_IS_DIMINISHED;
				} else if ((stressflags & S_NO_DIM) || (v == 1) || (v == (vowel_count-1))) {
					// first or last syllable, or option 'don't set diminished stress'
					v_stress = STRESS_IS_UNSTRESSED;
				} else if ((v == (vowel_count-2)) && (vowel_stress[vowel_count-1] <= STRESS_IS_UNSTRESSED)) {
					// penultimate syllable, followed by an unstressed final syllable
					v_stress = STRESS_IS_UNSTRESSED;
				} else {
					// unstressed syllable within a word
					if ((vowel_stress[v-1] < STRESS_IS_DIMINISHED) || ((stressflags & S_MID_DIM) == 0)) {
						v_stress = STRESS_IS_DIMINISHED;
						vowel_stress[v] = v_stress;
					}
				}
			}

			if ((v_stress == STRESS_IS_DIMINISHED) || (v_stress > STRESS_IS_UNSTRESSED))
				*output++ = stress_phonemes[v_stress]; // mark stress of all vowels except 1 (unstressed)

			if (vowel_stress[v] > max_stress)
				max_stress = vowel_stress[v];

			if ((*p == phonLENGTHEN) && ((opt_length = tr->langopts.param[LOPT_IT_LENGTHEN]) & 1)) {
				// remove lengthen indicator from non-stressed syllables
				bool shorten = false;

				if (opt_length & 0x10) {
					// only allow lengthen indicator on the highest stress syllable in the word
					if (v != max_stress_posn)
						shorten = true;
				} else if (v_stress < STRESS_IS_PRIMARY) {
					// only allow lengthen indicator if stress >= STRESS_IS_PRIMARY.
					shorten = true;
				}

				if (shorten)
					p++;
			}
			v++;
		}

		if (phcode != 1)
			*output++ = phcode;
	}
	*output++ = 0;

	return;
}

void AppendPhonemes(Translator *tr, char *string, int size, const char *ph)
{
	/* Add new phoneme string "ph" to "string"
	    Keeps count of the number of vowel phonemes in the word, and whether these
	   can be stressed syllables.  These values can be used in translation rules
	 */

	const char *p;
	unsigned char c;
	int length;

	length = strlen(ph) + strlen(string);
	if (length >= size)
		return;

	// any stressable vowel ?
	bool unstress_mark = false;
	p = ph;
	while ((c = *p++) != 0) {
		if (c >= n_phoneme_tab) continue;

		if (!phoneme_tab[c]) continue;

		if (phoneme_tab[c]->type == phSTRESS) {
			if (phoneme_tab[c]->std_length < 4)
				unstress_mark = true;
		} else {
			if (phoneme_tab[c]->type == phVOWEL) {
				if (((phoneme_tab[c]->phflags & phUNSTRESSED) == 0) &&
				    (unstress_mark == false)) {
					tr->word_stressed_count++;
				}
				unstress_mark = false;
				tr->word_vowel_count++;
			}
		}
	}

	if (string != NULL)
		strcat(string, ph);
}

static void MatchRule(Translator *tr, char *word[], char *word_start, int group_length, char *rule, MatchRecord *match_out, int word_flags, int dict_flags)
{
	/* Checks a specified word against dictionary rules.
	    Returns with phoneme code string, or NULL if no match found.

	    word (indirect) points to current character group within the input word
	            This is advanced by this procedure as characters are consumed

	    group:  the initial characters used to choose the rules group

	    rule:  address of dictionary rule data for this character group

	    match_out:  returns best points score

	    word_flags:  indicates whether this is a retranslation after a suffix has been removed
	 */

	unsigned char rb;     // current instuction from rule
	unsigned char letter; // current letter from input word, single byte
	int letter_w;         // current letter, wide character
	int last_letter_w;    // last letter, wide character
	int letter_xbytes;    // number of extra bytes of multibyte character (num bytes - 1)

	char *pre_ptr;
	char *post_ptr;       // pointer to first character after group

	char *rule_start;     // start of current match template
	char *p;
	int match_type;       // left, right, or consume
	int syllable_count;
	int vowel;
	int letter_group;
	int lg_pts;
	int n_bytes;
	int add_points;
	int command;

	MatchRecord match;
	MatchRecord best;

	int total_consumed; // letters consumed for best match

	unsigned char condition_num;
	char *common_phonemes; // common to a group of entries
	char *group_chars;
	char word_buf[N_WORD_BYTES];

	group_chars = *word;

	if (rule == NULL) {
		match_out->points = 0;
		(*word)++;
		return;
	}

	total_consumed = 0;
	common_phonemes = NULL;

	best.points = 0;
	best.phonemes = "";
	best.end_type = 0;
	best.del_fwd = NULL;

	// search through dictionary rules
	while (rule[0] != RULE_GROUP_END) {
		bool check_atstart = false;
		int consumed = 0;         // number of letters consumed from input
		int distance_left = -2;
        int distance_right = -6; // used to reduce points for matches further away the current letter
		int failed = 0;
		int unpron_ignore = word_flags & FLAG_UNPRON_TEST;

		match_type = 0;
		letter_w = 0;

		match.points = 1;
		match.end_type = 0;
		match.del_fwd = NULL;

		pre_ptr = *word;
		post_ptr = *word + group_length;

		// work through next rule until end, or until no-match proved
		rule_start = rule;

		while (!failed) {
			rb = *rule++;
			add_points = 0;

			if (rb <= RULE_LINENUM) {
				switch (rb)
				{
				case 0: // no phoneme string for this rule, use previous common rule
					if (common_phonemes != NULL) {
						match.phonemes = common_phonemes;
						while (((rb = *match.phonemes++) != 0) && (rb != RULE_PHONEMES)) {
							if (rb == RULE_CONDITION)
								match.phonemes++; // skip over condition number
							if (rb == RULE_LINENUM)
								match.phonemes += 2; // skip over line number
						}
					} else
						match.phonemes = "";
					rule--; // so we are still pointing at the 0
					failed = 2; // matched OK
					break;
				case RULE_PRE_ATSTART: // pre rule with implied 'start of word'
					check_atstart = true;
					unpron_ignore = 0;
					match_type = RULE_PRE;
					break;
				case RULE_PRE:
					match_type = RULE_PRE;
					if (word_flags & FLAG_UNPRON_TEST) {
						// checking the start of the word for unpronouncable character sequences, only
						// consider rules which explicitly match the start of a word
						// Note: Those rules now use RULE_PRE_ATSTART
						failed = 1;
					}
					break;
				case RULE_POST:
					match_type = RULE_POST;
					break;
				case RULE_PHONEMES:
					match.phonemes = rule;
					failed = 2; // matched OK
					break;
				case RULE_PH_COMMON:
					common_phonemes = rule;
					break;
				case RULE_CONDITION:
					// conditional rule, next byte gives condition number
					condition_num = *rule++;

					if (condition_num >= 32) {
						// allow the rule only if the condition number is NOT set
						if ((tr->dict_condition & (1L << (condition_num-32))) != 0)
							failed = 1;
					} else {
						// allow the rule only if the condition number is set
						if ((tr->dict_condition & (1L << condition_num)) == 0)
							failed = 1;
					}

					if (!failed)
						match.points++; // add one point for a matched conditional rule
					break;
				case RULE_LINENUM:
					rule += 2;
					break;
				}
				continue;
			}

			switch (match_type)
			{
			case 0:
				// match and consume this letter
				letter = *post_ptr++;

				if ((letter == rb) || ((letter == (unsigned char)REPLACED_E) && (rb == 'e'))) {
					if ((letter & 0xc0) != 0x80)
						add_points = 21; // don't add point for non-initial UTF-8 bytes
					consumed++;
				} else
					failed = 1;
				break;
			case RULE_POST:
				// continue moving forwards
				distance_right += 6;
				if (distance_right > 18)
					distance_right = 19;
				last_letter_w = letter_w;
				if (!post_ptr[-1]) {
					// we had already reached the end of text!
					// reading after that does not make sense, that cannot match
					failed = 1;
					break;
				}
				letter_xbytes = utf8_in(&letter_w, post_ptr)-1;
				letter = *post_ptr++;

				switch (rb)
				{
				case RULE_LETTERGP:
					letter_group = LetterGroupNo(rule++);
					if (IsLetter(tr, letter_w, letter_group)) {
						lg_pts = 20;
						if (letter_group == 2)
							lg_pts = 19; // fewer points for C, general consonant
						add_points = (lg_pts-distance_right);
						post_ptr += letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_LETTERGP2: // match against a list of utf-8 strings
					letter_group = LetterGroupNo(rule++);
					if ((n_bytes = IsLetterGroup(tr, post_ptr-1, letter_group, 0)) >= 0) {
						add_points = (20-distance_right);
						// move pointer, if group was found
						post_ptr += (n_bytes-1);
					} else
						failed = 1;
					break;
				case RULE_NOTVOWEL:
					if (IsLetter(tr, letter_w, 0) || ((letter_w == ' ') && (word_flags & FLAG_SUFFIX_VOWEL)))
						failed = 1;
					else {
						add_points = (20-distance_right);
						post_ptr += letter_xbytes;
					}
					break;
				case RULE_DIGIT:
					if (IsDigit(letter_w)) {
						add_points = (20-distance_right);
						post_ptr += letter_xbytes;
					} else if (tr->langopts.tone_numbers) {
						// also match if there is no digit
						add_points = (20-distance_right);
						post_ptr--;
					} else
						failed = 1;
					break;
				case RULE_NONALPHA:
					if (!iswalpha(letter_w)) {
						add_points = (21-distance_right);
						post_ptr += letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_DOUBLE:
					if (letter_w == last_letter_w) {
						add_points = (21-distance_right);
						post_ptr += letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_DOLLAR:
					post_ptr--;
					command = *rule++;
					if (command == DOLLAR_UNPR)
						match.end_type = SUFX_UNPRON; // $unpron
					else if (command == DOLLAR_NOPREFIX) { // $noprefix
						if (word_flags & FLAG_PREFIX_REMOVED)
							failed = 1; // a prefix has been removed
						else
							add_points = 1;
					} else if ((command & 0xf0) == 0x10) {
						// $w_alt
						if (dict_flags & (1 << (BITNUM_FLAG_ALT + (command & 0xf))))
							add_points = 23;
						else
							failed = 1;
					} else if (((command & 0xf0) == 0x20) || (command == DOLLAR_LIST)) {
						DollarRule(word, word_start, consumed, group_length, word_buf, tr, command, &failed, &add_points);
					}

					break;
				case '-':
					if ((letter == '-') || ((letter == ' ') && (word_flags & FLAG_HYPHEN_AFTER)))
						add_points = (22-distance_right); // one point more than match against space
					else
						failed = 1;
					break;
				case RULE_SYLLABLE:
				{
					// more than specified number of vowel letters to the right
					char *p = post_ptr + letter_xbytes;
					int vowel_count = 0;

					syllable_count = 1;
					while (*rule == RULE_SYLLABLE) {
						rule++;
						syllable_count += 1; // number of syllables to match
					}
					vowel = 0;
					while (letter_w != RULE_SPACE && letter_w != 0) {
						if ((vowel == 0) && IsLetter(tr, letter_w, LETTERGP_VOWEL2)) {
							// this is counting vowels which are separated by non-vowel letters
							vowel_count++;
						}
						vowel = IsLetter(tr, letter_w, LETTERGP_VOWEL2);
						p += utf8_in(&letter_w, p);
					}
					if (syllable_count <= vowel_count)
						add_points = (18+syllable_count-distance_right);
					else
						failed = 1;
				}
					break;
				case RULE_NOVOWELS:
				{
					char *p = post_ptr + letter_xbytes;
					while (letter_w != RULE_SPACE && letter_w != 0) {
						if (IsLetter(tr, letter_w, LETTERGP_VOWEL2)) {
							failed = 1;
							break;
						}
						p += utf8_in(&letter_w, p);
					}
					if (!failed)
						add_points = (19-distance_right);
				}
					break;
				case RULE_SKIPCHARS:
				{
					// '(Jxy'  means 'skip characters until xy'
					char *p = post_ptr - 1; // to allow empty jump (without letter between), go one back
					char *p2 = p;		// pointer to the previous character in the word
					int rule_w;		// first wide character of skip rule
					utf8_in(&rule_w, rule);
					int g_bytes = -1;	// bytes of successfully found character group
					while ((letter_w != rule_w) && (letter_w != RULE_SPACE) && (letter_w != 0) && (g_bytes == -1)) {
						if (rule_w == RULE_LETTERGP2)
							g_bytes = IsLetterGroup(tr, p, LetterGroupNo(rule + 1), 0);
						p2 = p;
						p += utf8_in(&letter_w, p);
					}
					if ((letter_w == rule_w) || (g_bytes >= 0))
						post_ptr = p2;
				}
					break;
				case RULE_INC_SCORE:
					post_ptr--;
					add_points = 20; // force an increase in points
					break;
				case RULE_DEC_SCORE:
					post_ptr--;
					add_points = -20; // force an decrease in points
					break;
				case RULE_DEL_FWD:
					// find the next 'e' in the word and replace by 'E'
					for (p = *word + group_length; p < post_ptr; p++) {
						if (*p == 'e') {
							match.del_fwd = p;
							break;
						}
					}
					break;
				case RULE_ENDING:
				{
					int end_type;
					// next 3 bytes are a (non-zero) ending type. 2 bytes of flags + suffix length
					end_type = (rule[0] << 16) + ((rule[1] & 0x7f) << 8) + (rule[2] & 0x7f);

					if ((tr->word_vowel_count == 0) && !(end_type & SUFX_P) && (tr->langopts.param[LOPT_SUFFIX] & 1))
						failed = 1; // don't match a suffix rule if there are no previous syllables (needed for lang=tr).
					else {
						match.end_type = end_type;
						rule += 3;
					}
				}
					break;
				case RULE_NO_SUFFIX:
					if (word_flags & FLAG_SUFFIX_REMOVED)
						failed = 1; // a suffix has been removed
					else {
						post_ptr--;
						add_points = 1;
					}
					break;
				default:
					if (letter == rb) {
						if ((letter & 0xc0) != 0x80) {
							// not for non-initial UTF-8 bytes
							add_points = (21-distance_right);
						}
					} else
						failed = 1;
					break;
				}
				break;
			case RULE_PRE:
				// match backwards from start of current group
				distance_left += 2;
				if (distance_left > 18)
					distance_left = 19;

				if (!*pre_ptr) {
					// we had already reached the beginning of text!
					// reading before this does not make sense, that cannot match
					failed = 1;
					break;
				}
				utf8_in(&last_letter_w, pre_ptr);
				pre_ptr--;
				letter_xbytes = utf8_in2(&letter_w, pre_ptr, 1)-1;
				letter = *pre_ptr;

				switch (rb)
				{
				case RULE_LETTERGP:
					letter_group = LetterGroupNo(rule++);
					if (IsLetter(tr, letter_w, letter_group)) {
						lg_pts = 20;
						if (letter_group == 2)
							lg_pts = 19; // fewer points for C, general consonant
						add_points = (lg_pts-distance_left);
						pre_ptr -= letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_LETTERGP2: // match against a list of utf-8 strings
					letter_group = LetterGroupNo(rule++);
					if ((n_bytes = IsLetterGroup(tr, pre_ptr, letter_group, 1)) >= 0) {
						add_points = (20-distance_right);
						// move pointer, if group was found
						pre_ptr -= (n_bytes-1);
					} else
						failed = 1;
					break;
				case RULE_NOTVOWEL:
					if (!IsLetter(tr, letter_w, 0)) {
						add_points = (20-distance_left);
						pre_ptr -= letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_DOUBLE:
					if (letter_w == last_letter_w) {
						add_points = (21-distance_left);
						pre_ptr -= letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_DIGIT:
					if (IsDigit(letter_w)) {
						add_points = (21-distance_left);
						pre_ptr -= letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_NONALPHA:
					if (!iswalpha(letter_w)) {
						add_points = (21-distance_right);
						pre_ptr -= letter_xbytes;
					} else
						failed = 1;
					break;
				case RULE_DOLLAR:
					pre_ptr++;
					command = *rule++;
					if ((command == DOLLAR_LIST) || ((command & 0xf0) == 0x20)) {
						DollarRule(word, word_start, consumed, group_length, word_buf, tr, command, &failed, &add_points);
					}
					break;
				case RULE_SYLLABLE:
					// more than specified number of vowels to the left
					syllable_count = 1;
					while (*rule == RULE_SYLLABLE) {
						rule++;
						syllable_count++; // number of syllables to match
					}
					if (syllable_count <= tr->word_vowel_count)
						add_points = (18+syllable_count-distance_left);
					else
						failed = 1;
					break;
				case RULE_STRESSED:
					pre_ptr++;
					if (tr->word_stressed_count > 0)
						add_points = 19;
					else
						failed = 1;
					break;
				case RULE_NOVOWELS:
				{
					char *p = pre_ptr - letter_xbytes;
					while (letter_w != RULE_SPACE) {
						if (IsLetter(tr, letter_w, LETTERGP_VOWEL2)) {
							failed = 1;
							break;
						}
						p -= utf8_in2(&letter_w, p-1, 1);
					}
					if (!failed)
						add_points = 3;
				}
					break;
				case RULE_IFVERB:
					pre_ptr++;
					if (tr->expect_verb)
						add_points = 1;
					else
						failed = 1;
					break;
				case RULE_CAPITAL:
					pre_ptr++;
					if (word_flags & FLAG_FIRST_UPPER)
						add_points = 1;
					else
						failed = 1;
					break;
				case '.':
					// dot in pre- section, match on any dot before this point in the word
					for (p = pre_ptr; *p && *p != ' '; p--) {
						if (*p == '.') {
							add_points = 50;
							break;
						}
					}
					if (!*p || *p == ' ')
						failed = 1;
					break;
				case '-':
					if ((letter == '-') || ((letter == ' ') && (word_flags & FLAG_HYPHEN)))
						add_points = (22-distance_right); // one point more than match against space
					else
						failed = 1;
					break;

				case RULE_SKIPCHARS: {
					// 'xyJ)'  means 'skip characters backwards until xy'
					char *p = pre_ptr + 1;	// to allow empty jump (without letter between), go one forward
					char *p2 = p;		// pointer to previous character in word
					int g_bytes = -1;	// bytes of successfully found character group

					while ((*p != *rule) && (*p != RULE_SPACE) && (*p != 0) && (g_bytes == -1)) {
						p2 = p;
						p--;
						if (*rule == RULE_LETTERGP2)
							g_bytes = IsLetterGroup(tr, p2, LetterGroupNo(rule + 1), 1);
					}

					// if succeed, set pre_ptr to next character after 'xy' and remaining
					// 'xy' part is checked as usual in following cycles of PRE rule characters
					if (*p == *rule)
						pre_ptr = p2;
					if (g_bytes >= 0)
						pre_ptr = p2 + 1;

				}
					break;

				default:
					if (letter == rb) {
						if (letter == RULE_SPACE)
							add_points = 4;
						else if ((letter & 0xc0) != 0x80) {
							// not for non-initial UTF-8 bytes
							add_points = (21-distance_left);
						}
					} else
						failed = 1;
					break;
				}
				break;
			}

			if (failed == 0)
				match.points += add_points;
		}

		if ((failed == 2) && (unpron_ignore == 0)) {
			// do we also need to check for 'start of word' ?
			if ((check_atstart == false) || (pre_ptr[-1] == ' ')) {
				if (check_atstart)
					match.points += 4;

				// matched OK, is this better than the last best match ?
				if (match.points >= best.points) {
					memcpy(&best, &match, sizeof(match));
					total_consumed = consumed;
				}

				if ((option_phonemes & espeakPHONEMES_TRACE) && (match.points > 0) && ((word_flags & FLAG_NO_TRACE) == 0)) {
					// show each rule that matches, and it's points score
					int pts;
					char decoded_phonemes[80];
					char output[80];

					pts = match.points;
					if (group_length > 1)
						pts += 35; // to account for an extra letter matching
					DecodePhonemes(match.phonemes, decoded_phonemes);
					fprintf(f_trans, "%3d\t%s [%s]\n", pts, DecodeRule(group_chars, group_length, rule_start, word_flags, output), decoded_phonemes);
				}
			}
		}

		// skip phoneme string to reach start of next template
		while (*rule++ != 0) ;
	}

	// advance input data pointer
	total_consumed += group_length;
	if (total_consumed == 0)
		total_consumed = 1; // always advance over 1st letter

	*word += total_consumed;

	if (best.points == 0)
		best.phonemes = "";
	memcpy(match_out, &best, sizeof(MatchRecord));
}

int TranslateRules(Translator *tr, char *p_start, char *phonemes, int ph_size, char *end_phonemes, int word_flags, unsigned int *dict_flags)
{
	/* Translate a word bounded by space characters
	   Append the result to 'phonemes' and any standard prefix/suffix in 'end_phonemes' */

	unsigned char c, c2;
	unsigned int c12;
	int wc = 0;
	char *p2;           // copy of p for use in double letter chain match
	int found;
	int g;              // group chain number
	int g1;             // first group for this letter
	int letter;
	int any_alpha = 0;
	int ix;
	unsigned int digit_count = 0;
	char *p;
	char word_buf[5];
	const ALPHABET *alphabet;
	int dict_flags0 = 0;
	MatchRecord match1 = { 0 };
	MatchRecord match2 = { 0 };
	char ph_buf[N_PHONEME_BYTES];
	char word_copy[N_WORD_BYTES];
	static const char str_pause[2] = { phonPAUSE_NOLINK, 0 };

	if (tr->data_dictrules == NULL)
		return 0;

	if (dict_flags != NULL)
		dict_flags0 = dict_flags[0];

	for (ix = 0; ix < (N_WORD_BYTES-1);) {
		c = p_start[ix];
		word_copy[ix++] = c;
		if (c == 0)
			break;
	}
	word_copy[ix] = 0;

	if ((option_phonemes & espeakPHONEMES_TRACE) && ((word_flags & FLAG_NO_TRACE) == 0)) {
		char wordbuf[120];
		unsigned int ix;

		for (ix = 0; ((c = p_start[ix]) != ' ') && (c != 0) && (ix < (sizeof(wordbuf)-1)); ix++)
			wordbuf[ix] = c;
		wordbuf[ix] = 0;
		if (word_flags & FLAG_UNPRON_TEST)
			fprintf(f_trans, "Unpronouncable? '%s'\n", wordbuf);
		else
			fprintf(f_trans, "Translate '%s'\n", wordbuf);
	}

	p = p_start;
	tr->word_vowel_count = 0;
	tr->word_stressed_count = 0;

	if (end_phonemes != NULL)
		end_phonemes[0] = 0;

	while (((c = *p) != ' ') && (c != 0)) {
		int wc_bytes = utf8_in(&wc, p);
		if (IsAlpha(wc))
			any_alpha++;

		int n = tr->groups2_count[c];
		if (IsDigit(wc) && ((tr->langopts.tone_numbers == 0) || !any_alpha)) {
			// lookup the number in *_list not *_rules
			char string[8];
			char buf[40];
			string[0] = '_';
			memcpy(&string[1], p, wc_bytes);
			string[1+wc_bytes] = 0;
			Lookup(tr, string, buf);
			if (++digit_count >= 2) {
				strcat(buf, str_pause);
				digit_count = 0;
			}
			AppendPhonemes(tr, phonemes, ph_size, buf);
			p += wc_bytes;
			continue;
		} else {
			digit_count = 0;
			found = 0;

			if (((ix = wc - tr->letter_bits_offset) >= 0) && (ix < 128)) {
				if (tr->groups3[ix] != NULL) {
					MatchRule(tr, &p, p_start, wc_bytes, tr->groups3[ix], &match1, word_flags, dict_flags0);
					found = 1;
				}
			}

			if (!found && (n > 0)) {
				// there are some 2 byte chains for this initial letter
				c2 = p[1];
				c12 = c + (c2 << 8); // 2 characters

				g1 = tr->groups2_start[c];
				for (g = g1; g < (g1+n); g++) {
					if (tr->groups2_name[g] == c12) {
						found = 1;

						p2 = p;
						MatchRule(tr, &p2, p_start, 2, tr->groups2[g], &match2, word_flags, dict_flags0);
						if (match2.points > 0)
							match2.points += 35; // to acount for 2 letters matching

						// now see whether single letter chain gives a better match ?
						MatchRule(tr, &p, p_start, 1, tr->groups1[c], &match1, word_flags, dict_flags0);

						if (match2.points >= match1.points) {
							// use match from the 2-letter group
							memcpy(&match1, &match2, sizeof(MatchRecord));
							p = p2;
						}
					}
				}
			}

			if (!found) {
				// alphabetic, single letter chain
				if (tr->groups1[c] != NULL)
					MatchRule(tr, &p, p_start, 1, tr->groups1[c], &match1, word_flags, dict_flags0);
				else {
					// no group for this letter, use default group
					MatchRule(tr, &p, p_start, 0, tr->groups1[0], &match1, word_flags, dict_flags0);

					if ((match1.points == 0) && ((option_sayas & 0x10) == 0)) {
						n = utf8_in(&letter, p-1)-1;

						if (tr->letter_bits_offset > 0) {
							// not a Latin alphabet, switch to the default Latin alphabet language
							if ((letter <= 0x241) && iswalpha(letter)) {
								sprintf(phonemes, "%cen", phonSWITCH);
								return 0;
							}
						}

						// is it a bracket ?
						if (letter == 0xe000+'(') {
							if (pre_pause < tr->langopts.param[LOPT_BRACKET_PAUSE_ANNOUNCED])
								pre_pause = tr->langopts.param[LOPT_BRACKET_PAUSE_ANNOUNCED]; // a bracket, already spoken by AnnouncePunctuation()
						}
						if (IsBracket(letter)) {
							if (pre_pause < tr->langopts.param[LOPT_BRACKET_PAUSE])
								pre_pause = tr->langopts.param[LOPT_BRACKET_PAUSE];
						}

						// no match, try removing the accent and re-translating the word
						if ((letter >= 0xc0) && (letter < N_REMOVE_ACCENT) && ((ix = remove_accent[letter-0xc0]) != 0)) {
							// within range of the remove_accent table
							if ((p[-2] != ' ') || (p[n] != ' ')) {
								// not the only letter in the word
								p2 = p-1;
								p[-1] = ix;
								while ((p[0] = p[n]) != ' ')  p++;
								while (n-- > 0) *p++ = ' '; // replacement character must be no longer than original

								if (tr->langopts.param[LOPT_DIERESES] && (lookupwchar(diereses_list, letter) > 0)) {
									// vowel with dieresis, replace and continue from this point
									p = p2;
									continue;
								}

								phonemes[0] = 0; // delete any phonemes which have been produced so far
								p = p_start;
								tr->word_vowel_count = 0;
								tr->word_stressed_count = 0;
								continue; // start again at the beginning of the word
							}
						}

						if (((alphabet = AlphabetFromChar(letter)) != NULL)  && (alphabet->offset != tr->letter_bits_offset)) {
							if (tr->langopts.alt_alphabet == alphabet->offset) {
								sprintf(phonemes, "%c%s", phonSWITCH, WordToString2(word_buf, tr->langopts.alt_alphabet_lang));
								return 0;
							}
							if (alphabet->flags & AL_WORDS) {
								// switch to the nominated language for this alphabet
								sprintf(phonemes, "%c%s", phonSWITCH, WordToString2(word_buf, alphabet->language));
								return 0;
							}
						}
					}
				}

				if (match1.points == 0) {
					if ((wc >= 0x300) && (wc <= 0x36f)) {
						// combining accent inside a word, ignore
					} else if (IsAlpha(wc)) {
						if ((any_alpha > 1) || (p[wc_bytes-1] > ' ')) {
							// an unrecognised character in a word, abort and then spell the word
							phonemes[0] = 0;
							if (dict_flags != NULL)
								dict_flags[0] |= FLAG_SPELLWORD;
							break;
						}
					} else {
						LookupLetter(tr, wc, -1, ph_buf, 0);
						if (ph_buf[0]) {
							match1.phonemes = ph_buf;
							match1.points = 1;
						}
					}
					p += (wc_bytes-1);
				} else
					tr->phonemes_repeat_count = 0;
			}
		}

		if (match1.phonemes == NULL)
			match1.phonemes = "";

		if (match1.points > 0) {
			if (word_flags & FLAG_UNPRON_TEST)
				return match1.end_type | 1;

			if ((match1.phonemes[0] == phonSWITCH) && ((word_flags & FLAG_DONT_SWITCH_TRANSLATOR) == 0)) {
				// an instruction to switch language, return immediately so we can re-translate
				strcpy(phonemes, match1.phonemes);
				return 0;
			}

			if ((option_phonemes & espeakPHONEMES_TRACE) && ((word_flags & FLAG_NO_TRACE) == 0))
				fprintf(f_trans, "\n");

			match1.end_type &= ~SUFX_UNPRON;

			if ((match1.end_type != 0) && (end_phonemes != NULL)) {
				// a standard ending has been found, re-translate the word without it
				if ((match1.end_type & SUFX_P) && (word_flags & FLAG_NO_PREFIX)) {
					// ignore the match on a prefix
				} else {
					if ((match1.end_type & SUFX_P) && ((match1.end_type & 0x7f) == 0)) {
						// no prefix length specified
						match1.end_type |= p - p_start;
					}
					strcpy(end_phonemes, match1.phonemes);
					memcpy(p_start, word_copy, strlen(word_copy));
					return match1.end_type;
				}
			}
			if (match1.del_fwd != NULL)
				*match1.del_fwd = REPLACED_E;
			AppendPhonemes(tr, phonemes, ph_size, match1.phonemes);
		}
	}

	memcpy(p_start, word_copy, strlen(word_copy));

	return 0;
}

int TransposeAlphabet(Translator *tr, char *text)
{
	// transpose cyrillic alphabet (for example) into ascii (single byte) character codes
	// return: number of bytes, bit 6: 1=used compression

	int c;
	int offset;
	int min;
	int max;
	const char *map;
	char *p = text;
	char *p2;
	bool all_alpha = true;
	int pairs_start;
	int bufix;
	char buf[N_WORD_BYTES+1];

	offset = tr->transpose_min - 1;
	min = tr->transpose_min;
	max = tr->transpose_max;
	map = tr->transpose_map;

	pairs_start = max - min + 2;

	bufix = 0;
	do {
		p += utf8_in(&c, p);
		if (c != 0) {
			if ((c >= min) && (c <= max)) {
				if (map == NULL)
					buf[bufix++] = c - offset;
				else {
					// get the code from the transpose map
					if (map[c - min] > 0)
						buf[bufix++] = map[c - min];
					else {
						all_alpha = false;
						break;
					}
				}
			} else {
				all_alpha = false;
				break;
			}
		}
	} while ((c != 0) && (bufix < N_WORD_BYTES));
	buf[bufix] = 0;

	if (all_alpha) {
		// compress to 6 bits per character
		int ix;
		int acc = 0;
		int bits = 0;

		p = buf;
		p2 = buf;
		while ((c = *p++) != 0) {
			const short *pairs_list;
			if ((pairs_list = tr->frequent_pairs) != NULL) {
				int c2 = c + (*p << 8);
				for (ix = 0; c2 >= pairs_list[ix]; ix++) {
					if (c2 == pairs_list[ix]) {
						// found an encoding for a 2-character pair
						c = ix + pairs_start; // 2-character codes start after the single letter codes
						p++;
						break;
					}
				}
			}
			acc = (acc << 6) + (c & 0x3f);
			bits += 6;

			if (bits >= 8) {
				bits -= 8;
				*p2++ = (acc >> bits);
			}
		}
		if (bits > 0)
			*p2++ = (acc << (8-bits));
		*p2 = 0;
		ix = p2 - buf;
		memcpy(text, buf, ix);
		return ix | 0x40; // bit 6 indicates compressed characters
	}
	return strlen(text);
}

/* Find an entry in the word_dict file for a specified word.
   Returns NULL if no match, else returns 'word_end'

    word   zero terminated word to match
    word2  pointer to next word(s) in the input text (terminated by space)

    flags:  returns dictionary flags which are associated with a matched word

    end_flags:  indicates whether this is a retranslation after removing a suffix
 */
static const char *LookupDict2(Translator *tr, const char *word, const char *word2,
                               char *phonetic, unsigned int *flags, int end_flags, WORD_TAB *wtab)
{
	char *p;
	char *next;
	int hash;
	int phoneme_len;
	int wlen;
	unsigned char flag;
	unsigned int dictionary_flags;
	unsigned int dictionary_flags2;
	bool condition_failed = false;
	int n_chars;
	int no_phonemes;
	int skipwords;
	int ix;
	int c;
	const char *word_end;
	const char *word1;
	int wflags = 0;
	int lookup_symbol;
	char word_buf[N_WORD_BYTES+1];
	char dict_flags_buf[80];

	if (wtab != NULL)
		wflags = wtab->flags;

	lookup_symbol = flags[1] & FLAG_LOOKUP_SYMBOL;
	word1 = word;
	if (tr->transpose_min > 0) {
		strncpy0(word_buf, word, N_WORD_BYTES);
		wlen = TransposeAlphabet(tr, word_buf); // bit 6 indicates compressed characters
		word = word_buf;
	} else
		wlen = strlen(word);

	hash = HashDictionary(word);
	p = tr->dict_hashtab[hash];

	if (p == NULL) {
		if (flags != NULL)
			*flags = 0;
		return 0;
	}

	// Find the first entry in the list for this hash value which matches.
	// This corresponds to the last matching entry in the *_list file.

	while (*p != 0) {
		next = p + (p[0] & 0xff);

		if (((p[1] & 0x7f) != wlen) || (memcmp(word, &p[2], wlen & 0x3f) != 0)) {
			// bit 6 of wlen indicates whether the word has been compressed; so we need to match on this also.
			p = next;
			continue;
		}

		// found matching entry. Decode the phonetic string
		word_end = word2;

		dictionary_flags = 0;
		dictionary_flags2 = 0;
		no_phonemes = p[1] & 0x80;

		p += ((p[1] & 0x3f) + 2);

		if (no_phonemes) {
			phonetic[0] = 0;
			phoneme_len = 0;
		} else {
			phoneme_len = strlen(p);
			assert(phoneme_len < N_PHONEME_BYTES);
			strcpy(phonetic, p);
			p += (phoneme_len + 1);
		}

		while (p < next) {
			// examine the flags which follow the phoneme string

			flag = *p++;
			if (flag >= 100) {
				// conditional rule
				if (flag >= 132) {
					// fail if this condition is set
					if ((tr->dict_condition & (1 << (flag-132))) != 0)
						condition_failed = true;
				} else {
					// allow only if this condition is set
					if ((tr->dict_condition & (1 << (flag-100))) == 0)
						condition_failed = true;
				}
			} else if (flag > 80) {
				// flags 81 to 90  match more than one word
				// This comes after the other flags
				n_chars = next - p;
				skipwords = flag - 80;

				// don't use the contraction if any of the words are emphasized
				//  or has an embedded command, such as MARK
				if (wtab != NULL) {
					for (ix = 0; ix <= skipwords && wtab[ix].length; ix++) {
						if (wtab[ix].flags & FLAG_EMPHASIZED2)
							condition_failed = true;

					}
				}

				if (strncmp(word2, p, n_chars) != 0)
					condition_failed = true;

				if (condition_failed) {
					p = next;
					break;
				}

				dictionary_flags |= FLAG_SKIPWORDS;
				dictionary_skipwords = skipwords;
				p = next;
				word_end = word2 + n_chars;
			} else if (flag > 64) {
				// stressed syllable information, put in bits 0-3
				dictionary_flags = (dictionary_flags & ~0xf) | (flag & 0xf);
				if ((flag & 0xc) == 0xc)
					dictionary_flags |= FLAG_STRESS_END;
			} else if (flag >= 32)
				dictionary_flags2 |= (1L << (flag-32));
			else
				dictionary_flags |= (1L << flag);
		}

		if (condition_failed) {
			condition_failed = false;
			continue;
		}

		if ((end_flags & FLAG_SUFX) == 0) {
			// no suffix has been removed
			if (dictionary_flags2 & FLAG_STEM)
				continue; // this word must have a suffix
		}

		if ((end_flags & SUFX_P) && (dictionary_flags2 & (FLAG_ONLY | FLAG_ONLY_S)))
			continue; // $only or $onlys, don't match if a prefix has been removed

		if (end_flags & FLAG_SUFX) {
			// a suffix was removed from the word
			if (dictionary_flags2 & FLAG_ONLY)
				continue; // no match if any suffix

			if ((dictionary_flags2 & FLAG_ONLY_S) && ((end_flags & FLAG_SUFX_S) == 0)) {
				// only a 's' suffix allowed, but the suffix wasn't 's'
				continue;
			}
		}

		if (dictionary_flags2 & FLAG_CAPITAL) {
			if (!(wflags & FLAG_FIRST_UPPER))
				continue;
		}
		if (dictionary_flags2 & FLAG_ALLCAPS) {
			if (!(wflags & FLAG_ALL_UPPER))
				continue;
		}
		if (dictionary_flags & FLAG_NEEDS_DOT) {
			if (!(wflags & FLAG_HAS_DOT))
				continue;
		}

		if ((dictionary_flags2 & FLAG_ATEND) && (word_end < translator->clause_end) && (lookup_symbol == 0)) {
			// only use this pronunciation if it's the last word of the clause, or called from Lookup()
			continue;
		}

		if ((dictionary_flags2 & FLAG_ATSTART) && !(wflags & FLAG_FIRST_WORD)) {
			// only use this pronunciation if it's the first word of a clause
			continue;
		}

		if ((dictionary_flags2 & FLAG_SENTENCE) && !(translator->clause_terminator & CLAUSE_TYPE_SENTENCE)) {
			// only if this clause is a sentence , i.e. terminator is {. ? !} not {, : :}
			continue;
		}

		if (dictionary_flags2 & FLAG_VERB) {
			// this is a verb-form pronunciation

			if (tr->expect_verb || (tr->expect_verb_s && (end_flags & FLAG_SUFX_S))) {
				// OK, we are expecting a verb
				if ((tr->translator_name == L('e', 'n')) && (tr->prev_dict_flags[0] & FLAG_ALT7_TRANS) && (end_flags & FLAG_SUFX_S)) {
					// lang=en, don't use verb form after 'to' if the word has 's' suffix
					continue;
				}
			} else {
				// don't use the 'verb' pronunciation unless we are expecting a verb
				continue;
			}
		}
		if (dictionary_flags2 & FLAG_PAST) {
			if (!tr->expect_past) {
				// don't use the 'past' pronunciation unless we are expecting past tense
				continue;
			}
		}
		if (dictionary_flags2 & FLAG_NOUN) {
			if ((!tr->expect_noun) || (end_flags & SUFX_V)) {
				// don't use the 'noun' pronunciation unless we are expecting a noun
				continue;
			}
		}
		if (dictionary_flags2 & FLAG_NATIVE) {
			if (tr != translator)
				continue; // don't use if we've switched translators
		}
		if (dictionary_flags & FLAG_ALT2_TRANS) {
			// language specific
			if ((tr->translator_name == L('h', 'u')) && !(tr->prev_dict_flags[0] & FLAG_ALT_TRANS))
				continue;
		}

		if (flags != NULL) {
			flags[0] = dictionary_flags | FLAG_FOUND_ATTRIBUTES;
			flags[1] = dictionary_flags2;
		}

		if (phoneme_len == 0) {
			if (option_phonemes & espeakPHONEMES_TRACE) {
				print_dictionary_flags(flags, dict_flags_buf, sizeof(dict_flags_buf));
				fprintf(f_trans, "Flags:  %s  %s\n", word1, dict_flags_buf);
			}
			return 0; // no phoneme translation found here, only flags. So use rules
		}

		if (flags != NULL)
			flags[0] |= FLAG_FOUND; // this flag indicates word was found in dictionary

		if (option_phonemes & espeakPHONEMES_TRACE) {
			char ph_decoded[N_WORD_PHONEMES];
			bool textmode;

			DecodePhonemes(phonetic, ph_decoded);

			if ((dictionary_flags & FLAG_TEXTMODE) == 0)
				textmode = false;
			else
				textmode = true;

			if (textmode == translator->langopts.textmode) {
				// only show this line if the word translates to phonemes, not replacement text
				if ((dictionary_flags & FLAG_SKIPWORDS) && (wtab != NULL)) {
					// matched more than one word
					// (check for wtab prevents showing RULE_SPELLING byte when speaking individual letters)
					memcpy(word_buf, word2, word_end-word2);
					word_buf[word_end-word2-1] = 0;
					fprintf(f_trans, "Found: '%s %s\n", word1, word_buf);
				} else
					fprintf(f_trans, "Found: '%s", word1);
				print_dictionary_flags(flags, dict_flags_buf, sizeof(dict_flags_buf));
				fprintf(f_trans, "' [%s]  %s\n", ph_decoded, dict_flags_buf);
			}
		}

		ix = utf8_in(&c, word);
		if (flags != NULL && (word[ix] == 0) && !IsAlpha(c))
			flags[0] |= FLAG_MAX3;
		return word_end;

	}
	return 0;
}


    static int utf8_nbytes(const char *buf)
{
	// Returns the number of bytes for the first UTF-8 character in buf

	unsigned char c = (unsigned char)buf[0];
	if (c < 0x80)
		return 1;
	if (c < 0xe0)
		return 2;
	if (c < 0xf0)
		return 3;
	return 4;
}

/* Lookup a specified word in the word dictionary.
   Returns phonetic data in 'phonetic' and bits in 'flags'

   end_flags:  indicates if a suffix has been removed
 */
int LookupDictList(Translator *tr, char **wordptr, char *ph_out, unsigned int *flags, int end_flags, WORD_TAB *wtab)
{
	int length;
	const char *found;
	const char *word1;
	const char *word2;
	unsigned char c;
	int nbytes;
	int len;
	char word[N_WORD_BYTES];
	static char word_replacement[N_WORD_BYTES];

	MAKE_MEM_UNDEFINED(&word_replacement, sizeof(word_replacement));

	length = 0;
	word2 = word1 = *wordptr;

	while ((word2[nbytes = utf8_nbytes(word2)] == ' ') && (word2[nbytes+1] == '.')) {
		// look for an abbreviation of the form a.b.c
		// try removing the spaces between the dots and looking for a match
		if (length + 1 > sizeof(word)) {
			/* Too long abbreviation, leave as it is */
			length = 0;
			break;
		}
		memcpy(&word[length], word2, nbytes);
		length += nbytes;
		word[length++] = '.';
		word2 += nbytes+3;
	}
	if (length > 0) {
		// found an abbreviation containing dots
		nbytes = 0;
		while (((c = word2[nbytes]) != 0) && (c != ' '))
			nbytes++;
		if (length + nbytes + 1 <= sizeof(word)) {
			memcpy(&word[length], word2, nbytes);
			word[length+nbytes] = 0;
			found =  LookupDict2(tr, word, word2, ph_out, flags, end_flags, wtab);
			if (found) {
				// set the skip words flag
				flags[0] |= FLAG_SKIPWORDS;
				dictionary_skipwords = length;
				return 1;
			}
		}
	}

	for (length = 0; length < (N_WORD_BYTES-1); length++) {
		if (((c = *word1++) == 0) || (c == ' '))
			break;

		if ((c == '.') && (length > 0) && (IsDigit09(word[length-1])))
			break; // needed for lang=hu, eg. "december 2.-ig"

		word[length] = c;
	}
	word[length] = 0;

	found = LookupDict2(tr, word, word1, ph_out, flags, end_flags, wtab);

	if (flags[0] & FLAG_MAX3) {
		if (strcmp(ph_out, tr->phonemes_repeat) == 0) {
			tr->phonemes_repeat_count++;
			if (tr->phonemes_repeat_count > 3)
				ph_out[0] = 0;
		} else {
			strncpy0(tr->phonemes_repeat, ph_out, sizeof(tr->phonemes_repeat));
			tr->phonemes_repeat_count = 1;
		}
	} else
		tr->phonemes_repeat_count = 0;

	if ((found == 0) && (flags[1] & FLAG_ACCENT)) {
		int letter;
		word2 = word;
		if (*word2 == '_') word2++;
		len = utf8_in(&letter, word2);
		LookupAccentedLetter(tr, letter, ph_out);
		found = word2 + len;
	}

	if (found == 0 && length >= 2) {
		ph_out[0] = 0;

		// try modifications to find a recognised word

		if ((end_flags & FLAG_SUFX_E_ADDED) && (word[length-1] == 'e')) {
			// try removing an 'e' which has been added by RemoveEnding
			word[length-1] = 0;
			found = LookupDict2(tr, word, word1, ph_out, flags, end_flags, wtab);
		} else if ((end_flags & SUFX_D) && (word[length-1] == word[length-2])) {
			// try removing a double letter
			word[length-1] = 0;
			found = LookupDict2(tr, word, word1, ph_out, flags, end_flags, wtab);
		}
	}

	if (found) {
		// if textmode is the default, then words which have phonemes are marked.
		if (tr->langopts.textmode)
			*flags ^= FLAG_TEXTMODE;

		if (*flags & FLAG_TEXTMODE) {
			// the word translates to replacement text, not to phonemes

			if (end_flags & FLAG_ALLOW_TEXTMODE) {
				// only use replacement text if this is the original word, not if a prefix or suffix has been removed
				word_replacement[0] = 0;
				word_replacement[1] = ' ';
				sprintf(&word_replacement[2], "%s ", ph_out); // replacement word, preceded by zerochar and space

				word1 = *wordptr;
				*wordptr = &word_replacement[2];

				if (option_phonemes & espeakPHONEMES_TRACE) {
					len = found - word1;
					memcpy(word, word1, len); // include multiple matching words
					word[len] = 0;
					fprintf(f_trans, "Replace: %s  %s\n", word, *wordptr);
				}
			}

			ph_out[0] = 0;
			return 0;
		}

		return 1;
	}

	ph_out[0] = 0;
	return 0;
}

extern char word_phonemes[N_WORD_PHONEMES]; // a word translated into phoneme codes

int Lookup(Translator *tr, const char *word, char *ph_out)
{
	// Look up in *_list, returns dictionary flags[0] and phonemes

	int flags0;
	unsigned int flags[2];
	char *word1 = (char *)word;

	flags[0] = 0;
	flags[1] = FLAG_LOOKUP_SYMBOL;
	if ((flags0 = LookupDictList(tr, &word1, ph_out, flags, FLAG_ALLOW_TEXTMODE, NULL)) != 0)
		flags0 = flags[0];

	if (flags[0] & FLAG_TEXTMODE) {
		int say_as = option_sayas;
		option_sayas = 0; // don't speak replacement word as letter names
		// NOTE: TranslateRoman checks text[-2] and IsLetterGroup looks
		// for a heading \0, so pad the start of text to prevent
		// it reading data on the stack.
		char text[80];

		text[0] = 0;
		text[1] = ' ';
		text[2] = ' ';
		strncpy0(text+3, word1, sizeof(text)-3);
		flags0 = TranslateWord(tr, text+3, NULL, NULL);
		strcpy(ph_out, word_phonemes);
		option_sayas = say_as;
	}
	return flags0;
}

static int LookupFlags(Translator *tr, const char *word, unsigned int flags_out[2])
{
	char buf[100];
	static unsigned int flags[2];
	char *word1 = (char *)word;

	flags[0] = flags[1] = 0;
	LookupDictList(tr, &word1, buf, flags, 0, NULL);
	flags_out[0] = flags[0];
	flags_out[1] = flags[1];
	return flags[0];
}

int RemoveEnding(Translator *tr, char *word, int end_type, char *word_copy)
{
	/* Removes a standard suffix from a word, once it has been indicated by the dictionary rules.
	   end_type: bits 0-6  number of letters
	             bits 8-14  suffix flags

	    word_copy: make a copy of the original word
	    This routine is language specific.  In English it deals with reversing y->i and e-dropping
	    that were done when the suffix was added to the original word.
	 */

	int i;
	char *word_end;
	int len_ending;
	int end_flags;
	char ending[50] = {0};

	// these lists are language specific, but are only relevant if the 'e' suffix flag is used
	static const char * const add_e_exceptions[] = {
		"ion", NULL
	};

	static const char * const add_e_additions[] = {
		"c", "rs", "ir", "ur", "ath", "ns", "u",
		"spong", // sponge
		"rang", // strange
		"larg", // large
		NULL
	};

	for (word_end = word; *word_end != ' '; word_end++) {
		// replace discarded 'e's
		if (*word_end == REPLACED_E)
			*word_end = 'e';
	}
	i = word_end - word;
	if (i >= N_WORD_BYTES) i = N_WORD_BYTES-1;

	if (word_copy != NULL) {
		memcpy(word_copy, word, i);
		word_copy[i] = 0;
	}

	// look for multibyte characters to increase the number of bytes to remove
	for (len_ending = i = (end_type & 0x3f); i > 0; i--) { // num.of characters of the suffix
		word_end--;
		while (word_end >= word && (*word_end & 0xc0) == 0x80) {
			word_end--; // for multibyte characters
			len_ending++;
		}
	}

	// remove bytes from the end of the word and replace them by spaces
	for (i = 0; (i < len_ending) && (i < (int)sizeof(ending)-1); i++) {
		ending[i] = word_end[i];
		word_end[i] = ' ';
	}
	ending[i] = 0;
	word_end--; // now pointing at last character of stem

	end_flags = (end_type & 0xfff0) | FLAG_SUFX;

	/* add an 'e' to the stem if appropriate,
	    if  stem ends in vowel+consonant
	    or  stem ends in 'c'  (add 'e' to soften it) */

	if (end_type & SUFX_I) {
		if (word_end[0] == 'i')
			word_end[0] = 'y';
	}

	if (end_type & SUFX_E) {
		if (tr->translator_name == L('n', 'l')) {
			if (((word_end[0] & 0x80) == 0) && ((word_end[-1] & 0x80) == 0) && IsVowel(tr, word_end[-1]) && IsLetter(tr, word_end[0], LETTERGP_C) && !IsVowel(tr, word_end[-2])) {
				// double the vowel before the (ascii) final consonant
				word_end[1] = word_end[0];
				word_end[0] = word_end[-1];
				word_end[2] = ' ';
			}
		} else if (tr->translator_name == L('e', 'n')) {
			// add 'e' to end of stem
			if (IsLetter(tr, word_end[-1], LETTERGP_VOWEL2) && IsLetter(tr, word_end[0], 1)) {
				// vowel(incl.'y') + hard.consonant

				const char *p;
				for (i = 0; (p = add_e_exceptions[i]) != NULL; i++) {
					int len = strlen(p);
					if (word_end + 1-len >= word && memcmp(p, &word_end[1-len], len) == 0)
						break;
				}
				if (p == NULL)
					end_flags |= FLAG_SUFX_E_ADDED; // no exception found
			} else {
				const char *p;
				for (i = 0; (p = add_e_additions[i]) != NULL; i++) {
					int len = strlen(p);
					if (word_end + 1-len >= word && memcmp(p, &word_end[1-len], len) == 0) {
						end_flags |= FLAG_SUFX_E_ADDED;
						break;
					}
				}
			}
		} else if (tr->langopts.suffix_add_e != 0)
			end_flags |= FLAG_SUFX_E_ADDED;

		if (end_flags & FLAG_SUFX_E_ADDED) {
			utf8_out(tr->langopts.suffix_add_e, &word_end[1]);

			if (option_phonemes & espeakPHONEMES_TRACE)
				fprintf(f_trans, "add e\n");
		}
	}

	if ((end_type & SUFX_V) && (tr->expect_verb == 0))
		tr->expect_verb = 1; // this suffix indicates the verb pronunciation


	if ((strcmp(ending, "s") == 0) || (strcmp(ending, "es") == 0))
		end_flags |= FLAG_SUFX_S;

	if (ending[0] == '\'')
		end_flags &= ~FLAG_SUFX; // don't consider 's as an added suffix

	return end_flags;
}

static void DollarRule(char *word[], char *word_start, int consumed, int group_length, char word_buf[N_WORD_BYTES], Translator *tr, int command, int *failed, int *add_points) {
	// $list or $p_alt
	// make a copy of the word up to the post-match characters
	int ix = *word - word_start + consumed + group_length + 1;

	if (ix+2 > N_WORD_BYTES) {
		*failed = 1;
		return;
	}

	memcpy(word_buf, word_start-1, ix);
	word_buf[ix] = ' ';
	word_buf[ix+1] = 0;
	unsigned int flags[2];
	LookupFlags(tr, &word_buf[1], flags);

	if ((command == DOLLAR_LIST) && (flags[0] & FLAG_FOUND) && !(flags[1] & FLAG_ONLY))
		*add_points = 23;
	else if (flags[0] & (1 << (BITNUM_FLAG_ALT + (command & 0xf))))
		*add_points = 23;
	else
		*failed = 1;
}
