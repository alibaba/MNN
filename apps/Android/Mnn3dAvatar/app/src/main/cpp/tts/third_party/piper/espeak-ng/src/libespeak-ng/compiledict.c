/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2017 Reece H. Dunn
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
 * along with this program; if not, write see:
 *             <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wctype.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "common.h"               // for strncpy0
#include "compiledict.h"
#include "dictionary.h"           // for EncodePhonemes, HashDicti...
#include "error.h"                // for create_file_error_context
#include "mnemonics.h"               // for LookupMnemName, MNEM_TAB
#include "phoneme.h"              // for PHONEME_TAB_LIST, phonSWITCH, phone...
#include "speech.h"		// for path_home
#include "synthesize.h"           // for Write4Bytes

static const MNEM_TAB mnem_rules[] = {
	{ "unpr",     DOLLAR_UNPR },
	{ "noprefix", DOLLAR_NOPREFIX },  // rule fails if a prefix has been removed
	{ "list",     DOLLAR_LIST },    // a pronunciation is given in the *_list file

	{ "w_alt1", 0x11 },
	{ "w_alt2", 0x12 },
	{ "w_alt3", 0x13 },
	{ "w_alt4", 0x14 },
	{ "w_alt5", 0x15 },
	{ "w_alt6", 0x16 },
	{ "w_alt",  0x11 }, // note: put longer names before their sub-strings

	{ "p_alt1", 0x21 },
	{ "p_alt2", 0x22 },
	{ "p_alt3", 0x23 },
	{ "p_alt4", 0x24 },
	{ "p_alt5", 0x25 },
	{ "p_alt6", 0x26 },
	{ "p_alt",  0x21 },

	{ NULL, -1 }
};

static const MNEM_TAB mnem_flags[] = {
	// these in the first group put a value in bits0-3 of dictionary_flags
	{ "$1",   0x41 }, // stress on 1st syllable
	{ "$2",   0x42 }, // stress on 2nd syllable
	{ "$3",   0x43 },
	{ "$4",   0x44 },
	{ "$5",   0x45 },
	{ "$6",   0x46 },
	{ "$7",   0x47 },
	{ "$u",   0x48 }, // reduce to unstressed
	{ "$u1",  0x49 },
	{ "$u2",  0x4a },
	{ "$u3",  0x4b },
	{ "$u+",  0x4c }, // reduce to unstressed, but stress at end of clause
	{ "$u1+", 0x4d },
	{ "$u2+", 0x4e },
	{ "$u3+", 0x4f },

	// these set the corresponding numbered bit if dictionary_flags
	{ "$pause",          8 }, // ensure pause before this word
	{ "$strend",         9 }, // full stress if at end of clause
	{ "$strend2",       10 }, // full stress if at end of clause, or only followed by unstressed
	{ "$unstressend",   11 }, // reduce stress at end of clause
	{ "$accent_before", 12 }, // used with accent names, say this accent name before the letter name
	{ "$abbrev",        13 }, // use this pronuciation rather than split into letters

	// language specific
	{ "$double",        14 }, // IT double the initial consonant of next word
	{ "$alt",           15 }, // use alternative pronunciation
	{ "$alt1",          15 }, // synonym for $alt
	{ "$alt2",          16 },
	{ "$alt3",          17 },
	{ "$alt4",          18 },
	{ "$alt5",          19 },
	{ "$alt6",          20 },
	{ "$alt7",          21 },

	{ "$combine",       23 }, // Combine with the next word

	{ "$dot",           24 }, // ignore '.' after this word (abbreviation)
	{ "$hasdot",        25 }, // use this pronunciation if there is a dot after the word

	{ "$max3",          27 }, // limit to 3 repetitions
	{ "$brk",           28 }, // a shorter $pause
	{ "$text",          29 }, // word translates to replcement text, not phonemes

	// flags in dictionary word 2
	{ "$verbf",      0x20 }, // verb follows
	{ "$verbsf",     0x21 }, // verb follows, allow -s suffix
	{ "$nounf",      0x22 }, // noun follows
	{ "$pastf",      0x23 }, // past tense follows
	{ "$verb",       0x24 }, // use this pronunciation when its a verb
	{ "$noun",       0x25 }, // use this pronunciation when its a noun
	{ "$past",       0x26 }, // use this pronunciation when its past tense
	{ "$verbextend", 0x28 }, // extend influence of 'verb follows'
	{ "$capital",    0x29 }, // use this pronunciation if initial letter is upper case
	{ "$allcaps",    0x2a }, // use this pronunciation if initial letter is upper case
	{ "$accent",     0x2b }, // character name is base-character name + accent name
	{ "$sentence",   0x2d }, // only if this clause is a sentence (i.e. terminator is {. ? !} not {, ; :}
	{ "$only",       0x2e }, // only match on this word without suffix
	{ "$onlys",      0x2f }, // only match with none, or with 's' suffix
	{ "$stem",       0x30 }, // must have a suffix
	{ "$atend",      0x31 }, // use this pronunciation if at end of clause
	{ "$atstart",    0x32 }, // use this pronunciation at start of clause
	{ "$native",     0x33 }, // not if we've switched translators

	// doesn't set dictionary_flags
	{ "$?",           100 }, // conditional rule, followed by byte giving the condition number

	{ "$textmode",    200 },
	{ "$phonememode", 201 },

	{ NULL, -1 }
};

#define LEN_GROUP_NAME  12

typedef struct {
	char name[LEN_GROUP_NAME+1];
	void *start;
	size_t length;
	int group3_ix;
	int index;
} RGROUP;

typedef enum
{
	LINE_PARSER_WORD = 0,
	LINE_PARSER_END_OF_WORD = 1,
	LINE_PARSER_MULTIPLE_WORDS = 2,
	LINE_PARSER_END_OF_WORDS = 3,
	LINE_PARSER_PRONUNCIATION = 4,
	LINE_PARSER_END_OF_PRONUNCIATION = 5,
} LINE_PARSER_STATES;

typedef struct {
	FILE *f_log;

	char word_phonemes[N_WORD_PHONEMES];    // a word translated into phoneme codes

	int linenum;
	int error_count;
	bool text_mode;
	int debug_flag;
	int error_need_dictionary;

	// A hash chain is a linked-list of hash chain entry objects:
	//     struct hash_chain_entry {
	//         hash_chain_entry *next_entry;
	//         // dict_line output from compile_line:
	//         uint8_t length;
	//         char contents[length];
	//     };
	char *hash_chains[N_HASH_DICT];

	char letterGroupsDefined[N_LETTER_GROUPS];

	char rule_cond[80];
	char rule_pre[80];
	char rule_post[80];
	char rule_match[80];
	char rule_phonemes[80];
	char group_name[LEN_GROUP_NAME+1];
	int group3_ix;
} CompileContext;

static void clean_context(CompileContext *ctx) {
	for (int i = 0; i < N_HASH_DICT; i++) {
		char *p;
		while ((p = ctx->hash_chains[i])) {
			memcpy(&p, ctx->hash_chains[i], sizeof(char*));
			free(ctx->hash_chains[i]);
			ctx->hash_chains[i] = p;
		}
	}
	free(ctx);
}

void print_dictionary_flags(unsigned int *flags, char *buf, int buf_len)
{
	int stress;
	int ix;
	const char *name;
	int len;
	int total = 0;

	buf[0] = 0;
	if ((stress = flags[0] & 0xf) != 0) {
		sprintf(buf, "%s", LookupMnemName(mnem_flags, stress + 0x40));
		total = strlen(buf);
		buf += total;
	}

	for (ix = 8; ix < 64; ix++) {
		if (((ix < 30) && (flags[0] & (1 << ix))) || ((ix >= 0x20) && (flags[1] & (1 << (ix-0x20))))) {
			name = LookupMnemName(mnem_flags, ix);
			len = strlen(name) + 1;
			total += len;
			if (total >= buf_len)
				continue;
			sprintf(buf, " %s", name);
			buf += len;
		}
	}
}

char *DecodeRule(const char *group_chars, int group_length, char *rule, int control, char *output)
{
	// Convert compiled match template to ascii

	unsigned char rb;
	unsigned char c;
	char *p;
	char *p_end;
	int ix;
	int match_type;
	bool finished = false;
	int value;
	int linenum = 0;
	int flags;
	int suffix_char;
	int condition_num = 0;
	bool at_start = false;
	const char *name;
	char buf[200];
	char buf_pre[200];
	char suffix[20];

	static const char symbols[] = {
		' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
		'&', '%', '+', '#', 'S', 'D', 'Z', 'A', 'L', '!',
		' ', '@', '?', 'J', 'N', 'K', 'V', '?', 'T', 'X',
		'?', 'W'
	};

	static const char symbols_lg[] = { 'A', 'B', 'C', 'H', 'F', 'G', 'Y' };

	match_type = 0;
	buf_pre[0] = 0;

	for (ix = 0; ix < group_length; ix++)
		buf[ix] = group_chars[ix];
	buf[ix] = 0;

	p = &buf[strlen(buf)];
	while (!finished) {
		rb = *rule++;

		if (rb <= RULE_LINENUM) {
			switch (rb)
			{
			case 0:
			case RULE_PHONEMES:
				finished = true;
				break;
			case RULE_PRE_ATSTART:
				at_start = true;
				// fallthrough:
			case RULE_PRE:
				match_type = RULE_PRE;
				*p = 0;
				p = buf_pre;
				break;
			case RULE_POST:
				match_type = RULE_POST;
				*p = 0;
				strcat(buf, " (");
				p = &buf[strlen(buf)];
				break;
			case RULE_PH_COMMON:
				break;
			case RULE_CONDITION:
				// conditional rule, next byte gives condition number
				condition_num = *rule++;
				break;
			case RULE_LINENUM:
				value = (rule[1] & 0xff) - 1;
				linenum = (rule[0] & 0xff) - 1 + (value * 255);
				rule += 2;
				break;
			}
			continue;
		}

		if (rb == RULE_DOLLAR) {
			value = *rule++ & 0xff;
			if ((value != 0x01) || (control & FLAG_UNPRON_TEST)) {
				// TODO write the string backwards if in RULE_PRE
				p[0] = '$';
				name = LookupMnemName(mnem_rules, value);
				strcpy(&p[1], name);
				p += (strlen(name)+1);
			}
			c = ' ';
		} else if (rb == RULE_ENDING) {
			static const char flag_chars[] = "eipvdfq tba ";
			flags = ((rule[0] & 0x7f)<< 8) + (rule[1] & 0x7f);
			suffix_char = 'S';
			if (flags & (SUFX_P >> 8))
				suffix_char = 'P';
			sprintf(suffix, "%c%d", suffix_char, rule[2] & 0x7f);
			rule += 3;
			for (ix = 0; ix < 9; ix++) {
				if (flags & 1)
					sprintf(&suffix[strlen(suffix)], "%c", flag_chars[ix]);
				flags = (flags >> 1);
			}
			strcpy(p, suffix);
			p += strlen(suffix);
			c = ' ';
		} else if (rb == RULE_LETTERGP)
			c = symbols_lg[*rule++ - 'A'];
		else if (rb == RULE_LETTERGP2) {
			value = *rule++ - 'A';
			if (value < 0)
				value += 256;
			p[0] = 'L';
			p[1] = (value / 10) + '0';
			c = (value % 10) + '0';

			if (match_type == RULE_PRE) {
				p[0] = c;
				c = 'L';
			}
			p += 2;
		} else if (rb <= RULE_LAST_RULE)
			c = symbols[rb];
		else if (rb == RULE_SPACE)
			c = '_';
		else
			c = rb;
		*p++ = c;
	}
	*p = 0;

	p = output;
	p_end = p + sizeof(output) - 1;

	if (linenum > 0) {
		sprintf(p, "%5d:\t", linenum);
		p += 7;
	}
	if (condition_num > 0) {
		sprintf(p, "?%d ", condition_num);
		p = &p[strlen(p)];
	}
	if (((ix = strlen(buf_pre)) > 0) || at_start) {
		if (at_start)
			*p++ = '_';
		while ((--ix >= 0) && (p < p_end-3))
			*p++ = buf_pre[ix];
		*p++ = ')';
		*p++ = ' ';
	}
	*p = 0;

	buf[p_end - p] = 0; // prevent overflow in output[]
	strcat(p, buf);
	ix = strlen(output);
	while (ix < 8)
		output[ix++] = ' ';
	output[ix] = 0;
	return output;
}

static int compile_line(CompileContext *ctx, char *linebuf, char *dict_line, int n_dict_line, int *hash)
{
	// Compile a line in the language_list file
	unsigned char c;
	char *p;
	char *word;
	char *phonetic;
	char *phonetic_end;
	unsigned int ix;
	LINE_PARSER_STATES step;
	unsigned int n_flag_codes = 0;
	int flagnum;
	int flag_offset;
	int length;
	int multiple_words = 0;
	bool multiple_numeric_hyphen = false;
	char *multiple_string = NULL;
	char *multiple_string_end = NULL;

	int len_word;
	int len_phonetic;
	bool text_not_phonemes = false; // this word specifies replacement text, not phonemes
	unsigned int wc;
	bool all_upper_case;

	char *mnemptr;
	unsigned char flag_codes[100];
	char encoded_ph[200];
	char bad_phoneme_str[4];
	int bad_phoneme;
	static const char nullstring[] = { 0 };

	phonetic = word = (char*)nullstring;

	p = linebuf;

	step = LINE_PARSER_WORD;

	c = *p;
	while (c != '\n' && c != '\0') {
		c = *p;

		if ((c == '?') && (step == 0)) {
			// conditional rule, allow only if the numbered condition is set for the voice
			flag_offset = 100;

			p++;
			if (*p == '!') {
				// allow only if the numbered condition is NOT set
				flag_offset = 132;
				p++;
			}

			ix = 0;
			if (IsDigit09(*p)) {
				ix += (*p-'0');
				p++;
			}
			if (IsDigit09(*p)) {
				ix = ix*10 + (*p-'0');
				p++;
			}
			flag_codes[n_flag_codes++] = ix + flag_offset;
			c = *p;
		}

		if ((c == '$') && isalnum(p[1])) {
			// read keyword parameter
			mnemptr = p;
			while (!isspace2(c = *p)) p++;
			*p = 0;

			flagnum = LookupMnem(mnem_flags, mnemptr);
			if (flagnum > 0) {
				if (flagnum == 200)
					ctx->text_mode = true;
				else if (flagnum == 201)
					ctx->text_mode = false;
				else if (flagnum == BITNUM_FLAG_TEXTMODE)
					text_not_phonemes = true;
				else
					flag_codes[n_flag_codes++] = flagnum;
			} else {
				fprintf(ctx->f_log, "%5d: Unknown keyword: %s\n", ctx->linenum, mnemptr);
				ctx->error_count++;
			}
		}

		if ((c == '/') && (p[1] == '/') && (multiple_words == 0))
			c = '\n'; // "//" treat comment as end of line

		switch (step)
		{
		case LINE_PARSER_WORD:
			if (c == '(') {
				multiple_words = 1;
				word = p+1;
				step = LINE_PARSER_END_OF_WORD;
			} else if (!isspace2(c)) {
				word = p;
				step = LINE_PARSER_END_OF_WORD;
			}
			break;
		case LINE_PARSER_END_OF_WORD:
			if ((c == '-') && multiple_words) {
				if (IsDigit09(word[0]))
					multiple_numeric_hyphen = true;
				flag_codes[n_flag_codes++] = BITNUM_FLAG_HYPHENATED;
				c = ' ';
			}
			if (isspace2(c)) {
				p[0] = 0; // terminate english word

				if (multiple_words) {
					multiple_string = multiple_string_end = p+1;
					step = LINE_PARSER_MULTIPLE_WORDS;
				} else
					step = LINE_PARSER_END_OF_WORDS;
			} else if (c == ')') {
				if (multiple_words) {
					p[0] = 0;
					multiple_words = 0;
					step = LINE_PARSER_END_OF_WORDS;
				} else if (word[0] != '_') {
					fprintf(ctx->f_log, "%5d: Missing '('\n", ctx->linenum);
					ctx->error_count++;
					step = LINE_PARSER_END_OF_WORDS;
				}
			}
			break;
		case LINE_PARSER_MULTIPLE_WORDS:
			if (isspace2(c))
				multiple_words++;
			else if (c == ')') {
				p[0] = ' '; // terminate extra string
				multiple_string_end = p+1;
				step = LINE_PARSER_END_OF_WORDS;
			}
			break;
		case LINE_PARSER_END_OF_WORDS:
			if (!isspace2(c)) {
				phonetic = p;
				step = LINE_PARSER_PRONUNCIATION;
			}
			break;
		case LINE_PARSER_PRONUNCIATION:
			if (isspace2(c)) {
				phonetic_end = p;
				p[0] = 0; // terminate phonetic
				step = LINE_PARSER_END_OF_PRONUNCIATION;
			}
			break;
		case LINE_PARSER_END_OF_PRONUNCIATION:
			if (!isspace2(c)) {
				*phonetic_end = ' ';
				step = LINE_PARSER_PRONUNCIATION;
			}
			break;
		}
		p++;
	}

	if (word[0] == 0)
		return 0; // blank line

	if (ctx->text_mode)
		text_not_phonemes = true;

	if (text_not_phonemes) {
		if (word[0] == '_') {
			// This is a special word, used by eSpeak.  Translate this into phonemes now
			strcat(phonetic, " "); // need a space to indicate word-boundary

			// PROBLEM  vowel reductions are not applied to the translated phonemes
			// condition rules are not applied
			TranslateWord(translator, phonetic, NULL, NULL);
			text_not_phonemes = false;
			strncpy0(encoded_ph, ctx->word_phonemes, N_WORD_BYTES-4);

			if ((ctx->word_phonemes[0] == 0) && (ctx->error_need_dictionary < 3)) {
				// the dictionary was not loaded, we need a second attempt
				ctx->error_need_dictionary++;
				fprintf(ctx->f_log, "%5d: Need to compile dictionary again\n", ctx->linenum);
			}
		} else
			// this is replacement text, so don't encode as phonemes. Restrict the length of the replacement word
			strncpy0(encoded_ph, phonetic, N_WORD_BYTES-4);
	} else {
		EncodePhonemes(phonetic, encoded_ph, &bad_phoneme);
		if (strchr(encoded_ph, phonSWITCH) != 0)
			flag_codes[n_flag_codes++] = BITNUM_FLAG_ONLY_S;  // don't match on suffixes (except 's') when switching languages

		// check for errors in the phonemes codes
		if (bad_phoneme != 0) {
			// unrecognised phoneme, report error
			bad_phoneme_str[utf8_out(bad_phoneme, bad_phoneme_str)] = 0;
			fprintf(ctx->f_log, "%5d: Bad phoneme [%s] (U+%x) in: %s  %s\n", ctx->linenum, bad_phoneme_str, bad_phoneme, word, phonetic);
			ctx->error_count++;
		}
	}

	if (text_not_phonemes != translator->langopts.textmode)
		flag_codes[n_flag_codes++] = BITNUM_FLAG_TEXTMODE;

	if (sscanf(word, "U+%x", &wc) == 1) {
		// Character code
		ix = utf8_out(wc, word);
		word[ix] = 0;
	} else if (word[0] != '_') {
		// convert to lower case, and note if the word is all-capitals
		int c2;

		all_upper_case = true;
		for (p = word;;) {
			// this assumes that the lower case char is the same length as the upper case char
			// OK, except for Turkish "I", but use towlower() rather than towlower2()
			ix = utf8_in(&c2, p);
			if (c2 == 0)
				break;
			if (iswupper(c2))
				utf8_out(towlower2(c2, translator), p);
			else
				all_upper_case = false;
			p += ix;
		}
		if (all_upper_case)
			flag_codes[n_flag_codes++] = BITNUM_FLAG_ALLCAPS;
	}

	len_word = strlen(word);

	if (translator->transpose_min > 0)
		len_word = TransposeAlphabet(translator, word);

	*hash = HashDictionary(word);
	len_phonetic = strlen(encoded_ph);

	dict_line[1] = len_word; // bit 6 indicates whether the word has been compressed
	len_word &= 0x3f;

	memcpy(&dict_line[2], word, len_word);

	if (len_phonetic == 0) {
		// no phonemes specified. set bit 7
		dict_line[1] |= 0x80;
		length = len_word + 2;
	} else {
		length = len_word + len_phonetic + 3;
		if (length < n_dict_line) {
			strcpy(&dict_line[(len_word)+2], encoded_ph);
		} else {
			fprintf(ctx->f_log, "%5d: Dictionary line length would overflow the data buffer: %d\n", ctx->linenum, length);
			ctx->error_count++;
			// no phonemes specified. set bit 7
			dict_line[1] |= 0x80;
			length = len_word + 2;
		}
	}

	for (ix = 0; ix < n_flag_codes; ix++)
		dict_line[ix+length] = flag_codes[ix];
	length += n_flag_codes;

	if ((multiple_string != NULL) && (multiple_words > 0)) {
		if (multiple_words > 10) {
			fprintf(ctx->f_log, "%5d: Two many parts in a multi-word entry: %d\n", ctx->linenum, multiple_words);
			ctx->error_count++;
		} else {
			dict_line[length++] = 80 + multiple_words;
			ix = multiple_string_end - multiple_string;
			if (multiple_numeric_hyphen)
				dict_line[length++] = ' ';   // ???
			memcpy(&dict_line[length], multiple_string, ix);
			length += ix;
		}
	}
	*((uint8_t *)dict_line) = (uint8_t)length;

	return length;
}

static void compile_dictlist_start(CompileContext *ctx)
{
	// initialise dictionary list
	int ix;
	char *p;
	char *p2;

	for (ix = 0; ix < N_HASH_DICT; ix++) {
		p = ctx->hash_chains[ix];
		while (p != NULL) {
			memcpy(&p2, p, sizeof(char *));
			free(p);
			p = p2;
		}
		ctx->hash_chains[ix] = NULL;
	}
}

static void compile_dictlist_end(CompileContext *ctx, FILE *f_out)
{
	// Write out the compiled dictionary list
	int hash;
	int length;
	char *p;

	for (hash = 0; hash < N_HASH_DICT; hash++) {
		p = ctx->hash_chains[hash];

		while (p != NULL) {
			length = *(uint8_t *)(p+sizeof(char *));
			fwrite(p+sizeof(char *), length, 1, f_out);
			memcpy(&p, p, sizeof(char *));
		}
		fputc(0, f_out);
	}
}

static int compile_dictlist_file(CompileContext *ctx, const char *path, const char *filename)
{
	int length;
	int hash;
	char *p;
	int count = 0;
	FILE *f_in;
	char buf[200];
	char fname[sizeof(path_home)+45];
	char dict_line[256]; // length is uint8_t, so an entry can't take up more than 256 bytes

	ctx->text_mode = false;

	// try with and without '.txt' extension
	sprintf(fname, "%s%s.txt", path, filename);
	if ((f_in = fopen(fname, "r")) == NULL) {
		sprintf(fname, "%s%s", path, filename);
		if ((f_in = fopen(fname, "r")) == NULL)
			return -1;
	}

	if (ctx->f_log != NULL)
		fprintf(ctx->f_log, "Compiling: '%s'\n", fname);

	ctx->linenum = 0;

	while (fgets(buf, sizeof(buf), f_in) != NULL) {
		ctx->linenum++;

		length = compile_line(ctx, buf, dict_line, sizeof(dict_line), &hash);
		if (length == 0)  continue; // blank line

		p = (char *)malloc(length+sizeof(char *));
		if (p == NULL) {
			if (ctx->f_log != NULL) {
				fprintf(ctx->f_log, "Can't allocate memory\n");
				ctx->error_count++;
			}
			break;
		}

		memcpy(p, &ctx->hash_chains[hash], sizeof(char *));
		ctx->hash_chains[hash] = p;
		// NOTE: dict_line[0] is the entry length (0-255)
		memcpy(p+sizeof(char *), dict_line, length);
		count++;
	}

	if (ctx->f_log != NULL)
		fprintf(ctx->f_log, "\t%d entries\n", count);
	fclose(f_in);
	return 0;
}

#define N_RULES 3000 // max rules for each group

static int isHexDigit(int c)
{
	if ((c >= '0') && (c <= '9'))
		return c - '0';
	if ((c >= 'a') && (c <= 'f'))
		return c - 'a' + 10;
	if ((c >= 'A') && (c <= 'F'))
		return c - 'A' + 10;
	return -1;
}

static void copy_rule_string(CompileContext *ctx, char *string, int *state_out)
{
	// state 0: conditional, 1=pre, 2=match, 3=post, 4=phonemes
	char * const outbuf[5] = { ctx->rule_cond, ctx->rule_pre, ctx->rule_match, ctx->rule_post, ctx->rule_phonemes };
	static const int next_state[5] = { 2, 2, 4, 4, 4 };
	char *output;
	char *p;
	int ix;
	int len;
	char c;
	int c2, c3;
	int sxflags;
	int value;
	bool literal;
	bool hexdigit_input = false;
	int state = *state_out;
	const MNEM_TAB *mr;

	if (string[0] == 0) return;

	output = outbuf[state];
	if (state == 4) {
		// append to any previous phoneme string, i.e. allow spaces in the phoneme string
		len = strlen(ctx->rule_phonemes);
		if (len > 0)
			ctx->rule_phonemes[len++] = ' ';
		output = &ctx->rule_phonemes[len];
	}
	sxflags = 0x808000; // to ensure non-zero bytes

	for (p = string, ix = 0;;) {
		literal = false;
		c = *p++;
		if ((c == '0') && (p[0] == 'x') && (isHexDigit(p[1]) >= 0) && (isHexDigit(p[2]) >= 0)) {
			hexdigit_input = true;
			c = p[1];
			p += 2;
		}
		if (c == '\\') {
			c = *p++; // treat next character literally
			if ((c >= '0') && (c <= '3') && (p[0] >= '0') && (p[0] <= '7') && (p[1] >= '0') && (p[1] <= '7')) {
				// character code given by 3 digit octal value;
				c = (c-'0')*64 + (p[0]-'0')*8 + (p[1]-'0');
				p += 2;
			}
			literal = true;
		}
		if (hexdigit_input) {
			if (((c2 = isHexDigit(c)) >= 0) && ((c3 = isHexDigit(p[0])) >= 0)) {
				c = c2 * 16 + c3;
				literal = true;
				p++;
			} else
				hexdigit_input = false;
		}
		if ((state == 1) || (state == 3)) {
			// replace special characters (note: 'E' is reserved for a replaced silent 'e')
			if (literal == false) {
				static const char lettergp_letters[9] = { LETTERGP_A, LETTERGP_B, LETTERGP_C, 0, 0, LETTERGP_F, LETTERGP_G, LETTERGP_H, LETTERGP_Y };
				switch (c)
				{
				case '_':
					c = RULE_SPACE;
					break;

				case 'Y':
					c = 'I';
					// fallthrough:
				case 'A': // vowel
				case 'B':
				case 'C':
				case 'H':
				case 'F':
				case 'G':
					if (state == 1) {
						// pre-rule, put the number before the RULE_LETTERGP;
						output[ix++] = lettergp_letters[c-'A'] + 'A';
						c = RULE_LETTERGP;
					} else {
						output[ix++] = RULE_LETTERGP;
						c = lettergp_letters[c-'A'] + 'A';
					}
					break;
				case 'D':
					c = RULE_DIGIT;
					break;
				case 'K':
					c = RULE_NOTVOWEL;
					break;
				case 'N':
					c = RULE_NO_SUFFIX;
					break;
				case 'V':
					c = RULE_IFVERB;
					break;
				case 'Z':
					c = RULE_NONALPHA;
					break;
				case '+':
					c = RULE_INC_SCORE;
					break;
				case '<': // Can't use - as opposite for + because it is used literally as part of word
					c = RULE_DEC_SCORE;
					break;
				case '@':
					c = RULE_SYLLABLE;
					break;
				case '&':
					c = RULE_STRESSED;
					break;
				case '%':
					c = RULE_DOUBLE;
					break;
				case '#':
					c = RULE_DEL_FWD;
					break;
				case '!':
					c = RULE_CAPITAL;
					break;
				case 'T':
					output[ix++] = RULE_DOLLAR;
					c = 0x11;
					break;
				case 'W':
					c = RULE_SPELLING;
					break;
				case 'X':
					c = RULE_NOVOWELS;
					break;
				case 'J':
					c = RULE_SKIPCHARS;
					break;
				case 'L':
					// expect two digits
					c = *p++ - '0';
					value = *p++ - '0';
					c = c * 10 + value;
					if ((value < 0) || (value > 9)) {
						c = 0;
						fprintf(ctx->f_log, "%5d: Expected 2 digits after 'L'\n", ctx->linenum);
						ctx->error_count++;
					} else if ((c <= 0) || (c >= N_LETTER_GROUPS) || (ctx->letterGroupsDefined[(int)c] == 0)) {
						fprintf(ctx->f_log, "%5d: Letter group L%.2d not defined\n", ctx->linenum, c);
						ctx->error_count++;
					}
					c += 'A';
					if (state == 1) {
						// pre-rule, put the group number before the RULE_LETTERGP command
						output[ix++] = c;
						c = RULE_LETTERGP2;
					} else
						output[ix++] = RULE_LETTERGP2;
					break;
				case '$':
					value = 0;
					mr = mnem_rules;
					while (mr->mnem != NULL) {
						len = strlen(mr->mnem);
						if (strncmp(p, mr->mnem, len) == 0) {
							value = mr->value;
							p += len;
							break;
						}
						mr++;
					}

					if (state == 1) {
						// pre-rule, put the number before the RULE_DOLLAR
						output[ix++] = value;
						c = RULE_DOLLAR;
					} else {
						output[ix++] = RULE_DOLLAR;
						c = value;
					}

					if (value == 0) {
						fprintf(ctx->f_log, "%5d: $ command not recognized\n", ctx->linenum);
						ctx->error_count++;
					}
					break;
				case 'P': // Prefix
					sxflags |= SUFX_P;
					// fallthrough
				case 'S': // Suffix
					output[ix++] = RULE_ENDING;
					value = 0;
					while (!isspace2(c = *p++) && (c != 0)) {
						switch (c)
						{
						case 'e':
							sxflags |= SUFX_E;
							break;
						case 'i':
							sxflags |= SUFX_I;
							break;
						case 'p': // obsolete, replaced by 'P' above
							sxflags |= SUFX_P;
							break;
						case 'v':
							sxflags |= SUFX_V;
							break;
						case 'd':
							sxflags |= SUFX_D;
							break;
						case 'f':
							sxflags |= SUFX_F;
							break;
						case 'q':
							sxflags |= SUFX_Q;
							break;
						case 't':
							sxflags |= SUFX_T;
							break;
						case 'b':
							sxflags |= SUFX_B;
							break;
						case 'a':
							sxflags |= SUFX_A;
							break;
						case 'm':
							sxflags |= SUFX_M;
							break;
						default:
							if (IsDigit09(c))
								value = (value*10) + (c - '0');
							break;
						}
					}
					p--;
					output[ix++] = sxflags >> 16;
					output[ix++] = sxflags >> 8;
					c = value | 0x80;
					break;
				}
			}
		}
		output[ix++] = c;
		if (c == 0) break;
	}

	*state_out = next_state[state];
}

static char *compile_rule(CompileContext *ctx, char *input)
{
	int ix;
	unsigned char c;
	int wc;
	char *p;
	char *prule;
	int len;
	int len_name;
	int start;
	int state = 2;
	bool finish = false;
	char buf[80];
	char output[150];
	int bad_phoneme;
	char bad_phoneme_str[4];

	buf[0] = 0;
	ctx->rule_cond[0] = 0;
	ctx->rule_pre[0] = 0;
	ctx->rule_post[0] = 0;
	ctx->rule_match[0] = 0;
	ctx->rule_phonemes[0] = 0;

	p = buf;

	for (ix = 0; finish == false; ix++) {
		switch (c = input[ix])
		{
		case ')': // end of prefix section
			*p = 0;
			state = 1;
			copy_rule_string(ctx, buf, &state);
			p = buf;
			break;
		case '(': // start of suffix section
			*p = 0;
			state = 2;
			copy_rule_string(ctx, buf, &state);
			state = 3;
			p = buf;
			if (input[ix+1] == ' ') {
				fprintf(ctx->f_log, "%5d: Syntax error. Space after (, or negative score for previous rule\n", ctx->linenum);
				ctx->error_count++;
			}
			break;
		case '\n': // end of line
		case '\r':
		case 0:    // end of line
			*p = 0;
			copy_rule_string(ctx, buf, &state);
			finish = true;
			break;
		case '\t': // end of section section
		case ' ':
			*p = 0;
			copy_rule_string(ctx, buf, &state);
			p = buf;
			break;
		case '?':
			if (state == 2)
				state = 0;
			else
				*p++ = c;
			break;
		default:
			*p++ = c;
			break;
		}
	}

	if (strcmp(ctx->rule_match, "$group") == 0)
		strcpy(ctx->rule_match, ctx->group_name);

	if (ctx->rule_match[0] == 0) {
		if (ctx->rule_post[0] != 0) {
			fprintf(ctx->f_log, "%5d: Syntax error\n", ctx->linenum);
			ctx->error_count++;
		}
		return NULL;
	}

	EncodePhonemes(ctx->rule_phonemes, buf, &bad_phoneme);
	if (bad_phoneme != 0) {
		bad_phoneme_str[utf8_out(bad_phoneme, bad_phoneme_str)] = 0;
		fprintf(ctx->f_log, "%5d: Bad phoneme [%s] (U+%x) in: %s\n", ctx->linenum, bad_phoneme_str, bad_phoneme, input);
		ctx->error_count++;
	}
	strcpy(output, buf);
	len = strlen(buf)+1;

	len_name = strlen(ctx->group_name);
	if ((len_name > 0) && (memcmp(ctx->rule_match, ctx->group_name, len_name) != 0)) {
		utf8_in(&wc, ctx->rule_match);
		if ((ctx->group_name[0] == '9') && IsDigit(wc)) {
			// numeric group, rule_match starts with a digit, so OK
		} else {
			fprintf(ctx->f_log, "%5d: Wrong initial letters '%s' for group '%s'\n", ctx->linenum, ctx->rule_match, ctx->group_name);
			ctx->error_count++;
		}
	}
	strcpy(&output[len], ctx->rule_match);
	len += strlen(ctx->rule_match);

	if (ctx->debug_flag) {
		output[len] = RULE_LINENUM;
		output[len+1] = (ctx->linenum % 255) + 1;
		output[len+2] = (ctx->linenum / 255) + 1;
		len += 3;
	}

	if (ctx->rule_cond[0] != 0) {
		if (ctx->rule_cond[0] == '!') {
			// allow the rule only if the condition number is NOT set for the voice
			ix = atoi(&ctx->rule_cond[1]) + 32;
		} else {
			// allow the rule only if the condition number is set for the voice
			ix = atoi(ctx->rule_cond);
		}

		if ((ix > 0) && (ix < 255)) {
			output[len++] = RULE_CONDITION;
			output[len++] = ix;
		} else {
			fprintf(ctx->f_log, "%5d: bad condition number ?%d\n", ctx->linenum, ix);
			ctx->error_count++;
		}
	}
	if (ctx->rule_pre[0] != 0) {
		start = 0;
		if (ctx->rule_pre[0] == RULE_SPACE) {
			// omit '_' at the beginning of the pre-string and imply it by using RULE_PRE_ATSTART
			c = RULE_PRE_ATSTART;
			start = 1;
		} else
			c = RULE_PRE;
		output[len++] = c;

		// output PRE string in reverse order
		for (ix = strlen(ctx->rule_pre)-1; ix >= start; ix--)
			output[len++] = ctx->rule_pre[ix];
	}

	if (ctx->rule_post[0] != 0) {
		sprintf(&output[len], "%c%s", RULE_POST, ctx->rule_post);
		len += (strlen(ctx->rule_post)+1);
	}
	output[len++] = 0;
	if ((prule = (char *)malloc(len)) != NULL)
		memcpy(prule, output, len);
	return prule;
}

static int __cdecl string_sorter(char **a, char **b)
{
	char *pa, *pb;
	int ix;

	if ((ix = strcmp(pa = *a, pb = *b)) != 0)
		return ix;
	pa += (strlen(pa)+1);
	pb += (strlen(pb)+1);
	return strcmp(pa, pb);
}

static int __cdecl rgroup_sorter(RGROUP *a, RGROUP *b)
{
	// Sort long names before short names
	int ix;
	ix = strlen(b->name) - strlen(a->name);
	if (ix != 0) return ix;
	ix = strcmp(a->name, b->name);
	if (ix != 0) return ix;
	return a->index - b->index;
}

static void* output_rule_group(int n_rules, char **rules, char *name, size_t *outsize)
{
	int ix;
	int len1;
	int len2;
	int len_name;
	char *p;
	char *p2, *p3;
	const char *common;
	char *outptr = NULL;
	size_t outpos, outlen = 0;

	short nextchar_count[256];
	memset(nextchar_count, 0, sizeof(nextchar_count));

	len_name = strlen(name);

	// sort the rules in this group by their phoneme string
	common = "";
	qsort((void *)rules, n_rules, sizeof(char *), (int(__cdecl *)(const void *, const void *))string_sorter);

	if (strcmp(name, "9") == 0)
		len_name = 0; //  don't remove characters from numeric match strings

	for (ix = 0; ix < n_rules; ix++) {
		p = rules[ix];
		len1 = strlen(p) + 1; // phoneme string
		p3 = &p[len1];
		p2 = p3 + len_name; // remove group name from start of match string
		len2 = strlen(p2);

		nextchar_count[(unsigned char)(p2[0])]++; // the next byte after the group name

		outpos = outlen;
		if ((common[0] != 0) && (strcmp(p, common) == 0)) {
			outlen += len2 + 1;
			outptr = realloc(outptr, outlen);
			memmove(outptr + outpos, p2, len2);
			outptr[outlen-1] = 0;
		} else {
			if ((ix < n_rules-1) && (strcmp(p, rules[ix+1]) == 0)) {
				outlen ++;
				outptr = realloc(outptr, outlen);
				common = rules[ix]; // phoneme string is same as next, set as common
				outptr[outpos++] = RULE_PH_COMMON;
			}

			outlen += len2 + 1 + len1;
			outptr = realloc(outptr, outlen);
			memmove(outptr + outpos, p2, len2);
			outpos += len2;
			outptr[outpos++] = RULE_PHONEMES;
			memmove(outptr + outpos, p, len1);
		}
	}
	if (outsize) *outsize = outlen;
	return outptr;
}

static int compile_lettergroup(CompileContext *ctx, char *input, FILE *f_out)
{
	char *p;
	char *p_start;
	int group;
	int ix;
	int n_items;
	int length;
	int max_length = 0;

	#define N_LETTERGP_ITEMS 200
	char *items[N_LETTERGP_ITEMS];
	char item_length[N_LETTERGP_ITEMS];

	p = input;
	if (!IsDigit09(p[0]) || !IsDigit09(p[1])) {
		fprintf(ctx->f_log, "%5d: Expected 2 digits after '.L'\n", ctx->linenum);
		ctx->error_count++;
		return 1;
	}

	group = atoi(&p[0]);
	if (group >= N_LETTER_GROUPS) {
		fprintf(ctx->f_log, "%5d: lettergroup out of range (01-%.2d)\n", ctx->linenum, N_LETTER_GROUPS-1);
		ctx->error_count++;
		return 1;
	}

	while (!isspace2(*p)) p++;

	fputc(RULE_GROUP_START, f_out);
	fputc(RULE_LETTERGP2, f_out);
	fputc(group + 'A', f_out);
	if (ctx->letterGroupsDefined[group] != 0) {
		fprintf(ctx->f_log, "%5d: lettergroup L%.2d is already defined\n", ctx->linenum, group);
		ctx->error_count++;
	}
	ctx->letterGroupsDefined[group] = 1;

	n_items = 0;
	while (n_items < N_LETTERGP_ITEMS) {
		while (isspace2(*p)) p++;
		if (*p == 0)
			break;

		items[n_items] = p_start = p;
		while ((*p & 0xff) > ' ') {
			if (*p == '_') *p = ' '; // allow '_' for word break
			p++;
		}
		*p++ = 0;
		length = p - p_start;
		if (length > max_length)
			max_length = length;
		item_length[n_items++] = length;
	}

	// write out the items, longest first
	while (max_length > 1) {
		for (ix = 0; ix < n_items; ix++) {
			if (item_length[ix] == max_length)
				fwrite(items[ix], 1, max_length, f_out);
		}
		max_length--;
	}

	fputc(RULE_GROUP_END, f_out);

	return 0;
}

static void free_rules(char **rules, int n_rules)
{
	for (int i = 0; i < n_rules; ++i) {
		free(*rules);
		*rules++ = NULL;
	}
}

static espeak_ng_STATUS compile_dictrules(CompileContext *ctx, FILE *f_in, FILE *f_out)
{
	char *prule;
	unsigned char *p;
	int ix;
	int c;
	int gp;
	int n_rules = 0;
	int count = 0;
	int different;
	int wc;
	int err_n_rules = 0;
	const char *prev_rgroup_name;
	unsigned int char_code;
	int compile_mode = 0;
	char *buf;
	char buf1[500];
	char *rules[N_RULES];

	int n_rgroups = 0;
	int n_groups3 = 0;
	RGROUP rgroup[N_RULE_GROUP2];

	ctx->linenum = 0;
	ctx->group_name[0] = 0;

	for (;;) {
		ctx->linenum++;
		buf = fgets(buf1, sizeof(buf1), f_in);
		if (buf != NULL) {
			if ((p = (unsigned char *)strstr(buf, "//")) != NULL)
				*p = 0;

			if (buf[0] == '\r') buf++; // ignore extra \r in \r\n
		}

		if ((buf == NULL) || (buf[0] == '.')) {
			// next .group or end of file, write out the previous group

			if (n_rules > 0) {
				strcpy(rgroup[n_rgroups].name, ctx->group_name);
				rgroup[n_rgroups].group3_ix = ctx->group3_ix;
				rgroup[n_rgroups].start = output_rule_group(n_rules, rules, ctx->group_name, &rgroup[n_rgroups].length);
				rgroup[n_rgroups].index = n_rgroups;
				n_rgroups++;

				count += n_rules;
				free_rules(rules, n_rules);
			}
			n_rules = 0;
			err_n_rules = 0;

			if (compile_mode == 2) {
				// end of the character replacements section
				fwrite(&n_rules, 1, 4, f_out); // write a zero word to terminate the replacemenmt list
				fputc(RULE_GROUP_END, f_out);
				compile_mode = 0;
			}

			if (buf == NULL) break; // end of file

			if (memcmp(buf, ".L", 2) == 0) {
				compile_lettergroup(ctx, &buf[2], f_out);
				continue;
			}

			if (memcmp(buf, ".replace", 8) == 0) {
				compile_mode = 2;
				fputc(RULE_GROUP_START, f_out);
				fputc(RULE_REPLACEMENTS, f_out);

				// advance to next word boundary
				while ((ftell(f_out) & 3) != 0)
					fputc(0, f_out);
			}

			if (memcmp(buf, ".group", 6) == 0) {
				compile_mode = 1;

				p = (unsigned char *)&buf[6];
				while ((p[0] == ' ') || (p[0] == '\t')) p++; // Note: Windows isspace(0xe1) gives TRUE !
				ix = 0;
				while ((*p > ' ') && (ix < LEN_GROUP_NAME))
					ctx->group_name[ix++] = *p++;
				ctx->group_name[ix] = 0;
				ctx->group3_ix = 0;

				if (sscanf(ctx->group_name, "0x%x", &char_code) == 1) {
					// group character is given as a character code (max 16 bits)
					p = (unsigned char *)ctx->group_name;

					if (char_code > 0x100)
						*p++ = (char_code >> 8);
					*p++ = char_code;
					*p = 0;
				} else {
					if (translator->letter_bits_offset > 0) {
						utf8_in(&wc, ctx->group_name);
						if (((ix = (wc - translator->letter_bits_offset)) >= 0) && (ix < 128))
							ctx->group3_ix = ix+1; // not zero
					}
				}

				if ((ctx->group3_ix == 0) && (strlen(ctx->group_name) > 2)) {
					if (utf8_in(&c, ctx->group_name) < 2) {
						fprintf(ctx->f_log, "%5d: Group name longer than 2 bytes (UTF8)", ctx->linenum);
						ctx->error_count++;
					}

					ctx->group_name[2] = 0;
				}
			}

			continue;
		}

		switch (compile_mode)
		{
		case 1: //  .group
			prule = compile_rule(ctx, buf);
			if (prule != NULL) {
				if (n_rules < N_RULES)
					rules[n_rules++] = prule;
				else {
					if (err_n_rules == 0) {
						fprintf(stderr, "\nExceeded limit of rules (%d) in group '%s'\n", N_RULES, ctx->group_name);
						ctx->error_count++;
						err_n_rules = 1;
					}
				}

			}
			break;
		case 2: //  .replace
			p = (unsigned char *)buf;

			while (isspace2(*p)) p++;
			if ((unsigned char)(*p) > 0x20) {
				while ((unsigned char)(*p) > 0x20) { // not space or zero-byte
					fputc(*p, f_out);
					p++;
				}
				fputc(0, f_out);

				while (isspace2(*p)) p++;
				while ((unsigned char)(*p) > 0x20) {
					fputc(*p, f_out);
					p++;
				}
				fputc(0, f_out);
			}
			break;
		}
	}

	qsort((void *)rgroup, n_rgroups, sizeof(rgroup[0]), (int(__cdecl *)(const void *, const void *))rgroup_sorter);

	prev_rgroup_name = "\n";

	for (gp = 0; gp < n_rgroups; gp++) {
		if ((different = strcmp(rgroup[gp].name, prev_rgroup_name)) != 0) {
			// not the same as the previous group
			if (gp > 0)
				fputc(RULE_GROUP_END, f_out);
			fputc(RULE_GROUP_START, f_out);

			if (rgroup[gp].group3_ix != 0) {
				n_groups3++;
				fputc(1, f_out);
				fputc(rgroup[gp].group3_ix, f_out);
			} else
				fprintf(f_out, "%s", prev_rgroup_name = rgroup[gp].name);
			fputc(0, f_out);
		}

		fwrite(rgroup[gp].start, rgroup[gp].length, 1, f_out);
	}
	fputc(RULE_GROUP_END, f_out);
	fputc(0, f_out);

	fprintf(ctx->f_log, "\t%d rules, %d groups (%d)\n\n", count, n_rgroups, n_groups3);
	free_rules(rules, n_rules);
	for (gp = 0; gp < n_rgroups; gp++) {
		free(rgroup[gp].start);
	}
	return ENS_OK;
}

#pragma GCC visibility push(default)
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_CompileDictionary(const char *dsource, const char *dict_name, FILE *log, int flags, espeak_ng_ERROR_CONTEXT *context)
{
	if (!log) log = stderr;
	if (!dict_name) dict_name = dictionary_name;

	// fname:  space to write the filename in case of error
	// flags: bit 0:  include source line number information, for debug purposes.

	FILE *f_in;
	FILE *f_out;
	int offset_rules = 0;
	int value;
	char fname_in[sizeof(path_home)+45];
	char fname_out[sizeof(path_home)+15];
	char path[sizeof(path_home)+40];       // path_dsource+20

	CompileContext *ctx = calloc(1, sizeof(CompileContext));

	ctx->error_count = 0;
	ctx->error_need_dictionary = 0;
	memset(ctx->letterGroupsDefined, 0, sizeof(ctx->letterGroupsDefined));

	ctx->debug_flag = flags & 1;

	if (dsource == NULL)
		dsource = "";

	ctx->f_log = log;
	if (ctx->f_log == NULL)
		ctx->f_log = stderr;

	// try with and without '.txt' extension
	sprintf(path, "%s%s_", dsource, dict_name);
	sprintf(fname_in, "%srules.txt", path);
	if ((f_in = fopen(fname_in, "r")) == NULL) {
		sprintf(fname_in, "%srules", path);
		if ((f_in = fopen(fname_in, "r")) == NULL) {
			clean_context(ctx);
			return create_file_error_context(context, errno, fname_in);
		}
	}

	sprintf(fname_out, "%s%c%s_dict", path_home, PATHSEP, dict_name);
	if ((f_out = fopen(fname_out, "wb+")) == NULL) {
		int error = errno;
		fclose(f_in);
		clean_context(ctx);
		return create_file_error_context(context, error, fname_out);
	}

	value = N_HASH_DICT;
	Write4Bytes(f_out, value);
	Write4Bytes(f_out, offset_rules);

	compile_dictlist_start(ctx);

	fprintf(ctx->f_log, "Using phonemetable: '%s'\n", phoneme_tab_list[phoneme_tab_number].name);
	compile_dictlist_file(ctx, path, "roots");
	if (translator->langopts.listx) {
		compile_dictlist_file(ctx, path, "list");
		compile_dictlist_file(ctx, path, "listx");
	} else {
		compile_dictlist_file(ctx, path, "listx");
		compile_dictlist_file(ctx, path, "list");
	}
	compile_dictlist_file(ctx, path, "emoji");
	compile_dictlist_file(ctx, path, "extra");

	compile_dictlist_end(ctx, f_out);
	offset_rules = ftell(f_out);

	fprintf(ctx->f_log, "Compiling: '%s'\n", fname_in);

	espeak_ng_STATUS status = compile_dictrules(ctx, f_in, f_out);
	fclose(f_in);

	fseek(f_out, 4, SEEK_SET);
	Write4Bytes(f_out, offset_rules);
	fclose(f_out);
	fflush(ctx->f_log);

	if (status != ENS_OK) {
		clean_context(ctx);
		return status;
	}

	LoadDictionary(translator, dict_name, 0);

	status = ctx->error_count > 0 ? ENS_COMPILE_ERROR : ENS_OK;
	clean_context(ctx);
	return status;
}
#pragma GCC visibility pop
