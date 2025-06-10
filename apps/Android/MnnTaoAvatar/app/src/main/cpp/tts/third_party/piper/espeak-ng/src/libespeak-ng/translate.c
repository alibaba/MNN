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
 * along with this program; if not, see: <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "translate.h"
#include "common.h"
#include "dictionary.h"           // for TranslateRules, LookupDictList, Cha...
#include "phoneme.h"              // for phonSWITCH, PHONEME_TAB, phonPAUSE_...
#include "phonemelist.h"          // for MakePhonemeList
#include "readclause.h"           // for towlower2, Eof, ReadClause, is_str_...
#include "synthdata.h"            // for SelectPhonemeTable, LookupPhonemeTable
#include "synthesize.h"           // for PHONEME_LIST2, N_PHONEME_LIST, PHON...
#include "ucd/ucd.h"              // for ucd_toupper
#include "voice.h"                // for voice, voice_t
#include "speech.h"               // for MAKE_MEM_UNDEFINED
#include "translateword.h"

static int CalcWordLength(int source_index, int charix_top, short int *charix, WORD_TAB *words, int word_count);
static void CombineFlag(Translator *tr, WORD_TAB *wtab, char *word, int *flags, unsigned char *p, char *word_phonemes);
static void SwitchLanguage(char *word, char *word_phonemes);

Translator *translator = NULL; // the main translator
Translator *translator2 = NULL; // secondary translator for certain words
static char translator2_language[20] = { 0 };
Translator *translator3 = NULL; // tertiary translator for certain words
static char translator3_language[20] = { 0 };

FILE *f_trans = NULL; // phoneme output text
int option_tone_flags = 0; // bit 8=emphasize allcaps, bit 9=emphasize penultimate stress
int option_phonemes = 0;
int option_phoneme_events = 0;
int option_endpause = 0; // suppress pause after end of text
int option_capitals = 0;
int option_punctuation = 0;
int option_sayas = 0;
static int option_sayas2 = 0; // used in translate_clause()
static int option_emphasis = 0; // 0=normal, 1=normal, 2=weak, 3=moderate, 4=strong
int option_ssml = 0;
int option_phoneme_input = 0; // allow [[phonemes]] in input
int option_wordgap = 0;

static int count_sayas_digits;
int skip_sentences;
int skip_words;
int skip_characters;
char skip_marker[N_MARKER_LENGTH];
bool skipping_text; // waiting until word count, sentence count, or named marker is reached
int end_character_position;
int count_sentences;
static int count_words;
int clause_start_char;
int clause_start_word;
static bool new_sentence;
static int word_emphasis = 0; // set if emphasis level 3 or 4
static int embedded_flag = 0; // there are embedded commands to be applied to the next phoneme, used in TranslateWord2()

static int max_clause_pause = 0;
static bool any_stressed_words;
int pre_pause;
static ALPHABET *current_alphabet;

char word_phonemes[N_WORD_PHONEMES]; // a word translated into phoneme codes
int n_ph_list2;
PHONEME_LIST2 ph_list2[N_PHONEME_LIST]; // first stage of text->phonemes

wchar_t option_punctlist[N_PUNCTLIST] = { 0 };

// these are overridden by defaults set in the "speak" file
int option_linelength = 0;

#define N_EMBEDDED_LIST  250
static int embedded_ix;
static int embedded_read;
unsigned int embedded_list[N_EMBEDDED_LIST];

// the source text of a single clause (UTF8 bytes)
static char source[N_TR_SOURCE+40]; // extra space for embedded command & voice change info at end

int n_replace_phonemes;
REPLACE_PHONEMES replace_phonemes[N_REPLACE_PHONEMES];

// other characters which break a word, but don't produce a pause
static const unsigned short breaks[] = { '_', 0 };

void DeleteTranslator(Translator *tr)
{
	if (!tr) return;

	if (tr->data_dictlist != NULL)
		free(tr->data_dictlist);
	free(tr);
}

int lookupwchar(const unsigned short *list, int c)
{
	// Is the character c in the list ?
	int ix;

	for (ix = 0; list[ix] != 0; ix++) {
		if (list[ix] == c)
			return ix+1;
	}
	return 0;
}

char *strchr_w(const char *s, int c)
{
	// return NULL for any non-ascii character
	if (c >= 0x80)
		return NULL;
	return strchr((char *)s, c); // (char *) is needed for Borland compiler
}

int TranslateWord(Translator *tr, char *word_start, WORD_TAB *wtab, char *word_out)
{
	char words_phonemes[N_WORD_PHONEMES]; // a word translated into phoneme codes
	char *phonemes = words_phonemes;


	int flags = TranslateWord3(tr, word_start, wtab, word_out, &any_stressed_words, current_alphabet, word_phonemes, sizeof(word_phonemes));
	if (flags & FLAG_TEXTMODE && word_out) {
		// Ensure that start of word rules match with the replaced text,
		// so that emoji and other characters are pronounced correctly.
		char word[N_WORD_BYTES+1];
		word[0] = 0;
		word[1] = ' ';
		strcpy(word+2, word_out);
		word_out = word+2;

			bool first_word = true;
			int available = N_WORD_PHONEMES;
		while (*word_out && available > 1) {
			int c;
			utf8_in(&c, word_out);
			if (iswupper(c)) {
				wtab->flags |= FLAG_FIRST_UPPER;
				utf8_out(tolower(c), word_out);
			} else {
				wtab->flags &= ~FLAG_FIRST_UPPER;
			}

			// dictionary_skipwords is a global variable and TranslateWord3 will reset it to 0 at the beginning.
			// However, dictionary_skipwords value is still needed outside this scope.
			// So we backup and restore it at the end of this scope.
			int skipwords = dictionary_skipwords;
			TranslateWord3(tr, word_out, wtab, NULL, &any_stressed_words, current_alphabet, word_phonemes, sizeof(word_phonemes));

			int n;
			if (first_word) {
				n = snprintf(phonemes, available, "%s", word_phonemes);
				first_word = false;
			} else {
				n = snprintf(phonemes, available, "%c%s", phonEND_WORD, word_phonemes);
			}

			available -= n;
			phonemes += n;

			// skip to the next word in a multi-word replacement. Always skip at least one word.
			for (dictionary_skipwords++; dictionary_skipwords > 0; dictionary_skipwords--) {
				while (!isspace(*word_out)) ++word_out;
				while (isspace(*word_out))  ++word_out;
			}
			dictionary_skipwords = skipwords;
		}

		// If the list file contains a text replacement to another
		// entry in the list file, e.g.:
		//     ripost     riposte $text
		//     riposte    rI#p0st
		// calling it from a prefix or suffix rule such as 'riposted'
		// causes word_out[0] to be NULL, as TranslateWord3 has the
		// information needed to perform the mapping. In this case,
		// no phonemes have been written in this loop and the phonemes
		// have been calculated, so don't override them.
		if (phonemes != words_phonemes) {
			snprintf(word_phonemes, sizeof(word_phonemes), "%s", words_phonemes);
		}
	}
	return flags;
}

static void SetPlist2(PHONEME_LIST2 *p, unsigned char phcode)
{
	p->phcode = phcode;
	p->stresslevel = 0;
	p->tone_ph = 0;
	p->synthflags = embedded_flag;
	p->sourceix = 0;
	embedded_flag = 0;
}

static int CountSyllables(unsigned char *phonemes)
{
	int count = 0;
	int phon;
	while ((phon = *phonemes++) != 0) {
		if (phoneme_tab[phon]->type == phVOWEL)
			count++;
	}
	return count;
}

static void Word_EmbeddedCmd(void)
{
	// Process embedded commands for emphasis, sayas, and break
	int embedded_cmd;
	do {
		embedded_cmd = embedded_list[embedded_read++];
		int value = embedded_cmd >> 8;

		switch (embedded_cmd & 0x1f)
		{
		case EMBED_Y:
			option_sayas = value;
			break;

		case EMBED_F:
			option_emphasis = value;
			break;

		case EMBED_B:
			// break command
			if (value == 0)
				pre_pause = 0; // break=none
			else
				pre_pause += value;
			break;
		}
	} while (((embedded_cmd & 0x80) == 0) && (embedded_read < embedded_ix));
}

static int SetAlternateTranslator(const char *new_language, Translator **translator, char translator_language[20])
{
	// Set alternate translator to a second language
	int new_phoneme_tab;

	if ((new_phoneme_tab = SelectPhonemeTableName(new_language)) >= 0) {
		if ((*translator != NULL) && (strcmp(new_language, translator_language) != 0)) {
			// we already have an alternative translator, but not for the required language, delete it
			DeleteTranslator(*translator);
			*translator = NULL;
		}

		if (*translator == NULL) {
			*translator = SelectTranslator(new_language);
			strcpy(translator_language, new_language);

			if (LoadDictionary(*translator, (*translator)->dictionary_name, 0) != 0) {
				SelectPhonemeTable(voice->phoneme_tab_ix); // revert to original phoneme table
				new_phoneme_tab = -1;
				translator_language[0] = 0;
			}
			(*translator)->phoneme_tab_ix = new_phoneme_tab;
		}
	}
	if (*translator != NULL)
		(*translator)->phonemes_repeat[0] = 0;
	return new_phoneme_tab;
}

int SetTranslator2(const char *new_language)
{
	return SetAlternateTranslator(new_language, &translator2, translator2_language);
}

int SetTranslator3(const char *new_language)
{
	return SetAlternateTranslator(new_language, &translator3, translator3_language);
}

static int TranslateWord2(Translator *tr, char *word, WORD_TAB *wtab, int pre_pause)
{
	int flags = 0;
	int stress;
	int next_stress;
	int next_tone = 0;
	unsigned char *p;
	int srcix;
	int found_dict_flag;
	unsigned char ph_code;
	PHONEME_LIST2 *plist2;
	PHONEME_TAB *ph;
	int max_stress;
	int max_stress_ix = 0;
	int prev_vowel = -1;
	int pitch_raised = 0;
	int switch_phonemes = -1;
	bool first_phoneme = true;
	int source_ix;
	int len;
	int bad_phoneme;
	int word_flags;
	char word_copy[N_WORD_BYTES+1];
	char word_replaced[N_WORD_BYTES+1];
	char old_dictionary_name[40];

	len = wtab->length;
	if (len > 31) len = 31;
	source_ix = (wtab->sourceix & 0x7ff) | (len << 11); // bits 0-10 sourceix, bits 11-15 word length

	word_flags = wtab[0].flags;
	if (word_flags & FLAG_EMBEDDED) {
		wtab[0].flags &= ~FLAG_EMBEDDED; // clear it in case we call TranslateWord2() again for the same word
		embedded_flag = SFLAG_EMBEDDED;

		Word_EmbeddedCmd();
	}

	if (n_ph_list2 >= N_PHONEME_LIST-2) {
		// No room, can't translate anything
		return 0;
	}

	if ((word[0] == 0) || (word_flags & FLAG_DELETE_WORD)) {
		// nothing to translate.  Add a dummy phoneme to carry any embedded commands
		if (embedded_flag) {
			SetPlist2(&ph_list2[n_ph_list2], phonEND_WORD);
			ph_list2[n_ph_list2].wordstress = 0;
			n_ph_list2++;
			embedded_flag = 0;
		}
		word_phonemes[0] = 0;
		return 0;
	}

	if (n_ph_list2 >= N_PHONEME_LIST-7-2) {
		// We may require up to 7 phonemes, plus the 2 phonemes from the caller, can't translate safely
		return 0;
	}

	// after a $pause word attribute, ignore a $pause attribute on the next two words
	if (tr->prepause_timeout > 0)
		tr->prepause_timeout--;

	if ((option_sayas & 0xf0) == 0x10) {
		if (!(word_flags & FLAG_FIRST_WORD)) {
			// SAYAS_CHARS, SAYAS_GLYPHS, or SAYAS_SINGLECHARS.  Pause between each word.
			pre_pause += 4;
		}
	}

	if (word_flags & FLAG_FIRST_UPPER) {
		if ((option_capitals > 2) && (embedded_ix < N_EMBEDDED_LIST-6)) {
			// indicate capital letter by raising pitch
			if (embedded_flag)
				embedded_list[embedded_ix-1] &= ~0x80; // already embedded command before this word, remove terminator
			if ((pitch_raised = option_capitals) == 3)
				pitch_raised = 20; // default pitch raise for capitals
			embedded_list[embedded_ix++] = EMBED_P+0x40+0x80 + (pitch_raised << 8); // raise pitch
			embedded_flag = SFLAG_EMBEDDED;
		}
	}

	p = (unsigned char *)word_phonemes;
	if (word_flags & FLAG_PHONEMES) {
		// The input is in phoneme mnemonics, not language text

		if (memcmp(word, "_^_", 3) == 0) {
			SwitchLanguage(word, word_phonemes);
		} else {
			EncodePhonemes(word, word_phonemes, &bad_phoneme);
		}

		flags = FLAG_FOUND;
	} else {
		int c2;
		int ix = 0;
		int word_copy_len;
		while (((c2 = word_copy[ix] = word[ix]) != ' ') && (c2 != 0) && (ix < N_WORD_BYTES)) ix++;
		word_copy_len = ix;

		word_replaced[2] = 0;
		flags = TranslateWord(translator, word, wtab, &word_replaced[2]);

		if (flags & FLAG_SPELLWORD) {
			// re-translate the word as individual letters, separated by spaces
			memcpy(word, word_copy, word_copy_len);
			return flags;
		}

		if ((flags & FLAG_COMBINE) && !(wtab[1].flags & FLAG_PHONEMES)) {
			CombineFlag(tr, wtab, word, &flags, p, word_phonemes);
		}

		if (p[0] == phonSWITCH) {
			int switch_attempt;
			strcpy(old_dictionary_name, dictionary_name);
			for (switch_attempt = 0; switch_attempt < 2; switch_attempt++) {
				// this word uses a different language
				memcpy(word, word_copy, word_copy_len);

				const char *new_language;
				new_language = (char *)(&p[1]);
				if (new_language[0] == 0)
					new_language = ESPEAKNG_DEFAULT_VOICE;

				switch_phonemes = SetTranslator2(new_language);

				if (switch_phonemes >= 0) {
					// re-translate the word using the new translator
					wtab[0].flags |= FLAG_TRANSLATOR2;
					if (word_replaced[2] != 0) {
						word_replaced[0] = 0; // byte before the start of the word
						word_replaced[1] = ' ';
						flags = TranslateWord(translator2, &word_replaced[1], wtab, NULL);
					} else
						flags = TranslateWord(translator2, word, wtab, &word_replaced[2]);
				}

				if (p[0] != phonSWITCH)
					break;
			}

			if (p[0] == phonSWITCH)
				return FLAG_SPELLWORD;

			if (switch_phonemes < 0) {
				// language code is not recognised or 2nd translator won't translate it
				p[0] = phonSCHWA; // just say something
				p[1] = phonSCHWA;
				p[2] = 0;
			}

			if (switch_phonemes == -1) {
				strcpy(dictionary_name, old_dictionary_name);
				SelectPhonemeTable(voice->phoneme_tab_ix);

				// leave switch_phonemes set, but use the original phoneme table number.
				// This will suppress LOPT_REGRESSIVE_VOICING
				switch_phonemes = voice->phoneme_tab_ix; // original phoneme table
			}
		}

		if (!(word_flags & FLAG_HYPHEN)) {
			if (flags & FLAG_PAUSE1) {
				if (pre_pause < 1)
					pre_pause = 1;
			}
			if ((flags & FLAG_PREPAUSE) && !(word_flags & (FLAG_LAST_WORD | FLAG_FIRST_WORD)) && !(wtab[-1].flags & FLAG_FIRST_WORD) && (tr->prepause_timeout == 0)) {
				// the word is marked in the dictionary list with $pause
				if (pre_pause < 4) pre_pause = 4;
				tr->prepause_timeout = 3;
			}
		}

		if ((option_emphasis >= 3) && (pre_pause < 1))
			pre_pause = 1;
	}

	stress = 0;
	next_stress = 1;
	srcix = 0;
	max_stress = -1;

	found_dict_flag = 0;
	if ((flags & FLAG_FOUND) && !(flags & FLAG_TEXTMODE))
		found_dict_flag = SFLAG_DICTIONARY;

	// Each iteration may require up to 1 phoneme
	// and after this loop we may require up to 7 phonemes
	// and our caller requires 2 phonemes
	while ((pre_pause > 0) && (n_ph_list2 < N_PHONEME_LIST-7-2)) {
		// add pause phonemes here. Either because of punctuation (brackets or quotes) in the
		// text, or because the word is marked in the dictionary lookup as a conjunction
		if (pre_pause > 1) {
			SetPlist2(&ph_list2[n_ph_list2++], phonPAUSE);
			pre_pause -= 2;
		} else {
			SetPlist2(&ph_list2[n_ph_list2++], phonPAUSE_NOLINK);
			pre_pause--;
		}
		tr->end_stressed_vowel = 0; // forget about the previous word
		tr->prev_dict_flags[0] = 0;
		tr->prev_dict_flags[1] = 0;
	}
	plist2 = &ph_list2[n_ph_list2];
	// From here we may require up to 4+1+3 phonemes

	// This may require up to 4 phonemes
	if ((option_capitals == 1) && (word_flags & FLAG_FIRST_UPPER)) {
		SetPlist2(&ph_list2[n_ph_list2++], phonPAUSE_SHORT);
		SetPlist2(&ph_list2[n_ph_list2++], phonCAPITAL);
		if ((word_flags & FLAG_ALL_UPPER) && IsAlpha(word[1])) {
			// word > 1 letter and all capitals
			SetPlist2(&ph_list2[n_ph_list2++], phonPAUSE_SHORT);
			SetPlist2(&ph_list2[n_ph_list2++], phonCAPITAL);
		}
	}

	// This may require up to 1 phoneme
	if (switch_phonemes >= 0) {
		if ((p[0] == phonPAUSE) && (p[1] == phonSWITCH)) {
			// the new word starts with a phoneme table switch, so there's no need to switch before it.
			if (ph_list2[n_ph_list2-1].phcode == phonSWITCH) {
				// previous phoneme is also a phonSWITCH, delete it
				n_ph_list2--;
			}
		} else {
			// this word uses a different phoneme table
			if (ph_list2[n_ph_list2-1].phcode == phonSWITCH) {
				// previous phoneme is also a phonSWITCH, just change its phoneme table number
				n_ph_list2--;
			} else
				SetPlist2(&ph_list2[n_ph_list2], phonSWITCH);
			ph_list2[n_ph_list2++].tone_ph = switch_phonemes; // temporary phoneme table number
		}
	}

	// remove initial pause from a word if it follows a hyphen
	if ((word_flags & FLAG_HYPHEN) && (phoneme_tab[*p]->type == phPAUSE))
		p++;

	if ((p[0] == 0) && (embedded_flag)) {
		// no phonemes.  Insert a very short pause to carry an embedded command
		p[0] = phonPAUSE_VSHORT;
		p[1] = 0;
	}

	// Each iteration may require up to 1 phoneme
	// and after this loop we may require up to 3 phonemes
	// and our caller requires 2 phonemes
	while (((ph_code = *p++) != 0) && (n_ph_list2 < N_PHONEME_LIST-3-2)) {
		if (ph_code == 255)
			continue; // unknown phoneme

		// Add the phonemes to the first stage phoneme list (ph_list2)
		ph = phoneme_tab[ph_code];
		if (ph == NULL) {
			printf("Invalid phoneme code %d\n", ph_code);
			continue;
		}

		if (ph_code == phonSWITCH) {
			ph_list2[n_ph_list2].phcode = ph_code;
			ph_list2[n_ph_list2].stresslevel = 0;
			ph_list2[n_ph_list2].sourceix = 0;
			ph_list2[n_ph_list2].synthflags = 0;
			ph_list2[n_ph_list2++].tone_ph = *p;
			SelectPhonemeTable(*p);
			p++;
		} else if (ph->type == phSTRESS) {
			// don't add stress phonemes codes to the list, but give their stress
			// value to the next vowel phoneme
			// std_length is used to hold stress number or (if >10) a tone number for a tone language
			if (ph->program == 0)
				next_stress = ph->std_length;
			else {
				// for tone languages, the tone number for a syllable follows the vowel
				if (prev_vowel >= 0)
					ph_list2[prev_vowel].tone_ph = ph_code;
				else
					next_tone = ph_code; // no previous vowel, apply to the next vowel
			}
		} else if (ph_code == phonSYLLABIC) {
			// mark the previous phoneme as a syllabic consonant
			prev_vowel = n_ph_list2-1;
			ph_list2[prev_vowel].synthflags |= SFLAG_SYLLABLE;
			ph_list2[prev_vowel].stresslevel = next_stress;
		} else if (ph_code == phonLENGTHEN)
			ph_list2[n_ph_list2-1].synthflags |= SFLAG_LENGTHEN;
		else if (ph_code == phonEND_WORD) {
			// a || symbol in a phoneme string was used to indicate a word boundary
			// Don't add this phoneme to the list, but make sure the next phoneme has
			// a newword indication
			srcix = source_ix+1;
		} else if (ph_code == phonX1) {
			// a language specific action
				flags |= FLAG_DOUBLING;
		} else {
			ph_list2[n_ph_list2].phcode = ph_code;
			ph_list2[n_ph_list2].tone_ph = 0;
			ph_list2[n_ph_list2].synthflags = embedded_flag | found_dict_flag;
			embedded_flag = 0;
			ph_list2[n_ph_list2].sourceix = srcix;
			srcix = 0;

			if (ph->type == phVOWEL) {
				stress = next_stress;
				next_stress = 1; // default is 'unstressed'

				if (stress >= 4)
					any_stressed_words = true;

				if ((prev_vowel >= 0) && (n_ph_list2-1) != prev_vowel)
					ph_list2[n_ph_list2-1].stresslevel = stress; // set stress for previous consonant

				ph_list2[n_ph_list2].synthflags |= SFLAG_SYLLABLE;
				prev_vowel = n_ph_list2;

				if (stress > max_stress) {
					max_stress = stress;
					max_stress_ix = n_ph_list2;
				}
				if (next_tone != 0) {
					ph_list2[n_ph_list2].tone_ph = next_tone;
					next_tone = 0;
				}
			} else {
				if (first_phoneme && tr->prev_dict_flags[0] & FLAG_DOUBLING) {
						// double the initial consonant if the previous word is marked with a flag
					ph_list2[n_ph_list2].synthflags |= SFLAG_LENGTHEN;
				}
			}

			ph_list2[n_ph_list2].stresslevel = stress;
			n_ph_list2++;
			first_phoneme = false;
		}
	}
	// From here, we may require up to 3 phonemes

	// This may require up to 1 phoneme
	if (word_flags & FLAG_COMMA_AFTER)
		SetPlist2(&ph_list2[n_ph_list2++], phonPAUSE_CLAUSE);

	// don't set new-word if there is a hyphen before it
	if ((word_flags & FLAG_HYPHEN) == 0)
		plist2->sourceix = source_ix;

	tr->end_stressed_vowel = 0;
	if ((stress >= 4) && (phoneme_tab[ph_list2[n_ph_list2-1].phcode]->type == phVOWEL))
		tr->end_stressed_vowel = 1; // word ends with a stressed vowel

	// This may require up to 1 phoneme
	if (switch_phonemes >= 0) {
		// this word uses a different phoneme table, now switch back
		strcpy(dictionary_name, old_dictionary_name);
		SelectPhonemeTable(voice->phoneme_tab_ix);
		SetPlist2(&ph_list2[n_ph_list2], phonSWITCH);
		ph_list2[n_ph_list2++].tone_ph = voice->phoneme_tab_ix; // original phoneme table number
	}


	// This may require up to 1 phoneme
	if (pitch_raised > 0) {
		embedded_list[embedded_ix++] = EMBED_P+0x60+0x80 + (pitch_raised << 8); // lower pitch
		SetPlist2(&ph_list2[n_ph_list2], phonPAUSE_SHORT);
		ph_list2[n_ph_list2++].synthflags = SFLAG_EMBEDDED;
	}

	if (flags & FLAG_STRESS_END2) {
		// this's word's stress could be increased later
		ph_list2[max_stress_ix].synthflags |= SFLAG_PROMOTE_STRESS;
	}

	tr->prev_dict_flags[0] = flags;
	return flags;
}

static int EmbeddedCommand(unsigned int *source_index_out)
{
	// An embedded command to change the pitch, volume, etc.
	// returns number of commands added to embedded_list

	// pitch,speed,amplitude,expression,reverb,tone,voice,sayas
	const char *commands = "PSARHTIVYMUBF";
	int value = -1;
	int sign = 0;
	unsigned char c;
	char *p;
	int cmd;
	int source_index = *source_index_out;

	c = source[source_index];
	if (c == '+') {
		sign = 0x40;
		source_index++;
	} else if (c == '-') {
		sign = 0x60;
		source_index++;
	}

	if (IsDigit09(source[source_index])) {
		value = atoi(&source[source_index]);
		while (IsDigit09(source[source_index]))
			source_index++;
	}

	c = source[source_index++];
	if (embedded_ix >= (N_EMBEDDED_LIST - 2))
		return 0; // list is full

	if ((p = strchr_w(commands, c)) == NULL)
		return 0;
	cmd = (p - commands)+1;
	if (value == -1) {
		value = embedded_default[cmd];
		sign = 0;
	}

	if (cmd == EMBED_Y) {
		option_sayas2 = value;
		count_sayas_digits = 0;
	}
	if (cmd == EMBED_F) {
		if (value >= 3)
			word_emphasis = FLAG_EMPHASIZED;
		else
			word_emphasis = 0;
	}

	embedded_list[embedded_ix++] = cmd + sign + (value << 8);
	*source_index_out = source_index;
	return 1;
}

static const char *FindReplacementChars(Translator *tr, const char **pfrom, unsigned int c, const char *next, int *ignore_next_n)
{
	const char *from = *pfrom;
	while ( !is_str_totally_null(from, 4) ) {
		unsigned int fc = 0; // from character
		unsigned int nc = c; // next character
		const char *match_next = next;

		*pfrom = from;

		from += utf8_in((int *)&fc, from);
		if (nc == fc) {
			if (*from == 0) return from + 1;

			bool matched = true;
			int nmatched = 0;
			while (*from != 0) {
				from += utf8_in((int *)&fc, from);

				match_next += utf8_in((int *)&nc, match_next);
				nc = towlower2(nc, tr);

				if (nc != fc)
					matched = false;
				else
					nmatched++;
			}

			if (matched) {
				*ignore_next_n = nmatched;
				return from + 1;
			}
		}

		// replacement 'from' string (skip the remaining part, if any)
		while (*from != '\0') from++;
		from++;

		// replacement 'to' string
		while (*from != '\0') from++;
		from++;
	}
	return NULL;
}

// handle .replace rule in xx_rules file
static int SubstituteChar(Translator *tr, unsigned int c, unsigned int next_in, const char *next, int *insert, int *wordflags)
{
	unsigned int new_c, c2 = ' ', c_lower;
	int upper_case = 0;

	static int ignore_next_n = 0;
	if (ignore_next_n > 0) {
		ignore_next_n--;
		return 8;
	}

	if (c == 0) return 0;

	const char *from = (const char *)tr->langopts.replace_chars;
	if (from == NULL)
		return c;

	// there is a list of character codes to be substituted with alternative codes

	if (iswupper(c_lower = c)) {
		c_lower = towlower2(c, tr);
		upper_case = 1;
	}

	const char *to = FindReplacementChars(tr, &from, c_lower, next, &ignore_next_n);
	if (to == NULL)
		return c; // no substitution

	if (option_phonemes & espeakPHONEMES_TRACE)
		fprintf(f_trans, "Replace: %s > %s\n", from, to);

	to += utf8_in((int *)&new_c, to);
	if (*to != 0) {
		// there is a second character to be inserted
		// don't convert the case of the second character unless the next letter is also upper case
		to += utf8_in((int *)&c2, to);
		if (upper_case && iswupper(next_in))
			c2 = ucd_toupper(c2);
		*insert = c2;
	}

	if (upper_case)
		new_c = ucd_toupper(new_c);

	*wordflags |= FLAG_CHAR_REPLACED;
	return new_c;
}

static int TranslateChar(Translator *tr, char *ptr, int prev_in, unsigned int c, unsigned int next_in, int *insert, int *wordflags)
{
	// To allow language specific examination and replacement of characters

	int code;
	int next2;

	static const unsigned char hangul_compatibility[0x34] = {
		0,  0x00, 0x01, 0xaa, 0x02, 0xac, 0xad, 0x03,
		0x04, 0x05, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb4,
		0xb6, 0x06, 0x07, 0x08, 0xb9, 0x09, 0x0a, 0xbc,
		0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x61,
		0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
		0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71,
		0x72, 0x73, 0x74, 0x75
	};

	// check for Korean Hangul letters
	if (((code = c - 0xac00) >= 0) && (c <= 0xd7af)) {
		// break a syllable hangul into 2 or 3 individual jamo
		int initial = (code/28)/21;
		int medial = (code/28) % 21;
		int final = code % 28;

		if (initial == 11) {
			// null initial
			c = medial + 0x1161;
			if (final > 0)
				*insert = final + 0x11a7;
		} else {
			// extract the initial and insert the remainder with a null initial
			c = initial + 0x1100;
			*insert = (11*28*21) + (medial*28) + final + 0xac00;
		}
		return c;
	} else if (((code = c - 0x3130) >= 0) && (code < 0x34)) {
		// Hangul compatibility jamo
		return hangul_compatibility[code] + 0x1100;
	}

	switch (tr->translator_name)
	{
	case L('a', 'f'):
	case L('n', 'l'):
		// look for 'n  and replace by a special character (unicode: schwa)

		if ((c == '\'') && !iswalpha(prev_in)) {
			utf8_in(&next2, &ptr[1]);

			if (IsSpace(next2)) {
				if ((next_in == 'n') && (tr->translator_name == L('a', 'f'))) {
					// n preceded by either apostrophe or U2019 "right single quotation mark"
					ptr[0] = ' '; // delete the n
					return 0x0259; // replace  '  by  unicode schwa character
				}
				if ((next_in == 'n') || (next_in == 't')) {
					// Dutch, [@n] and [@t]
					return 0x0259; // replace  '  by  unicode schwa character
				}
			}
		}
		break;
	}
	// handle .replace rule in xx_rules file
	return SubstituteChar(tr, c, next_in, ptr, insert, wordflags);
}

static const char *const UCase_ga[] = { "bp", "bhf", "dt", "gc", "hA", "mb", "nd", "ng", "ts", "tA", "nA", NULL };

static int UpperCaseInWord(Translator *tr, char *word, int c)
{
	if (tr->translator_name == L('g', 'a')) {
		int ix;
		const char *p;

		for (ix = 0;; ix++) {
			int len;
			if ((p = UCase_ga[ix]) == NULL)
				break;

			len = strlen(p);
			if ((word[-len] == ' ') && (memcmp(&word[-len+1], p, len-1) == 0)) {
				if ((c == p[len-1]) || ((p[len-1] == 'A') && IsVowel(tr, c)))
					return 1;
			}
		}
	}
	return 0;
}

// Same as TranslateClause except we also get the clause terminator used (full stop, comma, etc.).
// Used by espeak_TextToPhonemesWithTerminator.
void TranslateClauseWithTerminator(Translator *tr, int *tone_out, char **voice_change, int *terminator_out)
{
	int ix;
	int c;
	int cc = 0;
	unsigned int source_index = 0;
	int source_index_word = 0;
	int prev_in;
	int prev_out = ' ';
	int prev_in_save = 0;
	int next_in;
	int next_in_nbytes;
	int char_inserted = 0;
	int clause_pause;
	int pre_pause_add = 0;
	int all_upper_case = FLAG_ALL_UPPER;
	int alpha_count = 0;
	bool finished = false;
	bool single_quoted = false;
	bool phoneme_mode = false;
	int dict_flags = 0; // returned from dictionary lookup
	int word_flags; // set here
	int next_word_flags;
	bool new_sentence2;
	int embedded_count = 0;
	int letter_count = 0;
	bool space_inserted = false;
	bool syllable_marked = false;
	bool decimal_sep_count = false;
	char *word;
	char *p;
	int j, k;
	int n_digits;
	int charix_top = 0;

	short charix[N_TR_SOURCE+4];
	WORD_TAB words[N_CLAUSE_WORDS];
	static char voice_change_name[40];
	int word_count = 0; // index into words

	char sbuf[N_TR_SOURCE];

	int terminator;
	int tone;

	if (tr == NULL)
		return;

	MAKE_MEM_UNDEFINED(&voice_change_name, sizeof(voice_change_name));

	embedded_ix = 0;
	embedded_read = 0;
	pre_pause = 0;
	any_stressed_words = false;

	if ((clause_start_char = count_characters) < 0)
		clause_start_char = 0;
	clause_start_word = count_words + 1;

	for (ix = 0; ix < N_TR_SOURCE; ix++)
		charix[ix] = 0;
	MAKE_MEM_UNDEFINED(&source, sizeof(source));
	terminator = ReadClause(tr, source, charix, &charix_top, N_TR_SOURCE, &tone, voice_change_name);

	if (terminator_out != NULL) {
		*terminator_out = terminator;
	}

	if (tone_out != NULL) {
		if (tone == 0)
			*tone_out = (terminator & CLAUSE_INTONATION_TYPE) >> 12; // tone type not overridden in ReadClause, use default
		else
			*tone_out = tone; // override tone type
	}

	charix[charix_top+1] = 0;
	charix[charix_top+2] = 0x7fff;
	charix[charix_top+3] = 0;

	clause_pause = (terminator & CLAUSE_PAUSE) * 10; // mS
	if (terminator & CLAUSE_PAUSE_LONG)
		clause_pause = clause_pause * 32; // pause value is *320mS not *10mS

	for (p = source; *p != 0; p++) {
		if (!isspace2(*p))
			break;
	}
	if (*p == 0) {
		// No characters except spaces. This is not a sentence.
		// Don't add this pause, just make up the previous pause to this value;
		clause_pause -= max_clause_pause;
		if (clause_pause < 0)
			clause_pause = 0;

		if (new_sentence)
			terminator |= CLAUSE_TYPE_SENTENCE; // carry forward an end-of-sentence indicator
		max_clause_pause += clause_pause;
		new_sentence2 = false;
	} else {
		max_clause_pause = clause_pause;
		new_sentence2 = new_sentence;
	}
	tr->clause_terminator = terminator;

	if (new_sentence2) {
		count_sentences++;
		if (skip_sentences > 0) {
			skip_sentences--;
			if (skip_sentences == 0)
				skipping_text = false;
		}
	}

	MAKE_MEM_UNDEFINED(&ph_list2, sizeof(ph_list2));
	memset(&ph_list2[0], 0, sizeof(ph_list2[0]));
	ph_list2[0].phcode = phonPAUSE_SHORT;

	n_ph_list2 = 1;
	tr->prev_last_stress = 0;
	tr->prepause_timeout = 0;
	tr->expect_verb = 0;
	tr->expect_noun = 0;
	tr->expect_past = 0;
	tr->expect_verb_s = 0;
	tr->phonemes_repeat_count = 0;
	tr->end_stressed_vowel = 0;
	tr->prev_dict_flags[0] = 0;
	tr->prev_dict_flags[1] = 0;

	word_count = 0;
	word_flags = 0;
	next_word_flags = 0;

	sbuf[0] = 0;
	sbuf[1] = ' ';
	sbuf[2] = ' ';
	ix = 3;
	prev_in = ' ';

	words[0].start = ix;
	words[0].flags = 0;

	words[0].length = CalcWordLength(source_index, charix_top, charix, words, 0);

	int prev_out2;
	while (!finished && (ix < (int)sizeof(sbuf) - 1)) {
		prev_out2 = prev_out;
		utf8_in2(&prev_out, &sbuf[ix-1], 1);

		if (tr->langopts.tone_numbers && IsDigit09(prev_out) && IsAlpha(prev_out2)) {
			// tone numbers can be part of a word, consider them as alphabetic
			prev_out = 'a';
		}

		if (prev_in_save != 0) {
			prev_in = prev_in_save;
			prev_in_save = 0;
		} else if (source_index > 0)
			utf8_in2(&prev_in, &source[source_index-1], 1);

		unsigned int prev_source_index = source_index;

		if (char_inserted) {
			c = char_inserted;
			char_inserted = 0;
		} else {
			source_index += utf8_in(&cc, &source[source_index]);
			c = cc;
		}

		if (c == 0) {
			finished = true;
			c = ' ';
			next_in = ' ';
			next_in_nbytes = 0;
		}
		else
			next_in_nbytes = utf8_in(&next_in, &source[source_index]);

		if (c == CTRL_EMBEDDED) {
			// start of embedded command in the text
			int srcix = source_index-1;

			if (prev_in != ' ') {
				c = ' ';
				prev_in_save = c;
				source_index--;
			} else {
				embedded_count += EmbeddedCommand(&source_index);
				prev_in_save = prev_in;
				// replace the embedded command by spaces
				memset(&source[srcix], ' ', source_index-srcix);
				source_index = srcix;
				continue;
			}
		}

		if ((option_sayas2 == SAYAS_KEY) && (c != ' ')) {
			if ((prev_in == ' ') && (next_in == ' '))
				option_sayas2 = SAYAS_SINGLE_CHARS; // single character, speak its name
			c = towlower2(c, tr);
		}


		if (phoneme_mode) {
			all_upper_case = FLAG_PHONEMES;

			if ((c == ']') && (next_in == ']')) {
				phoneme_mode = false;
				source_index++;
				c = ' ';
			}
		} else if ((option_sayas2 & 0xf0) == SAYAS_DIGITS) {
			if (iswdigit(c)) {
				count_sayas_digits++;
				if (count_sayas_digits > (option_sayas2 & 0xf)) {
					// break after the specified number of digits
					c = ' ';
					space_inserted = true;
					count_sayas_digits = 0;
				}
			} else {
				count_sayas_digits = 0;
				if (iswdigit(prev_out)) {
					c = ' ';
					space_inserted = true;
				}
			}
		} else if ((option_sayas2 & 0x10) == 0) {
			// speak as words

			if ((c == 0x92) || (c == 0xb4) || (c == 0x2019) || (c == 0x2032))
				c = '\''; // 'microsoft' quote or sexed closing single quote, or prime - possibly used as apostrophe

			if (((c == 0x2018) || (c == '?')) && IsAlpha(prev_out) && IsAlpha(next_in)) {
				// ? between two letters may be a smart-quote replaced by ?
				c = '\'';
			}

			if (c == CHAR_EMPHASIS) {
				// this character is a marker that the previous word is the focus of the clause
				c = ' ';
				word_flags |= FLAG_FOCUS;
			}

			if (c == CHAR_COMMA_BREAK) {
				c = ' ';
				word_flags |= FLAG_COMMA_AFTER;
			}
			// language specific character translations
			c = TranslateChar(tr, &source[source_index], prev_in, c, next_in, &char_inserted, &word_flags);
			if (c == 8)
				continue; // ignore this character

			if (char_inserted)
				next_in = char_inserted;

			// allow certain punctuation within a word (usually only apostrophe)
			if (!IsAlpha(c) && !IsSpace(c) && (wcschr(tr->punct_within_word, c) == 0)) {
				if (IsAlpha(prev_out)) {
					if (tr->langopts.tone_numbers && IsDigit09(c) && !IsDigit09(next_in)) {
						// allow a tone number as part of the word
					} else {
						c = ' '; // ensure we have an end-of-word terminator
						space_inserted = true;
					}
				}
			}

			if (iswdigit(prev_out)) {
				if (!iswdigit(c) && (c != '.') && (c != ',') && (c != ' ')) {
					c = ' '; // terminate digit string with a space
					space_inserted = true;
				}
			} else { // Prev output is not digit
				if (prev_in == ',') {
					// Workaround for several consecutive commas —
					// replace current character with space
					if (c == ',')
						c = ' ';
				} else {
					decimal_sep_count = false;
				}
			}

			if (c == '[') {
				if ((next_in == '\002') || ((next_in == '[') && option_phoneme_input)) {
					//  "[\002" is used internally to start phoneme mode
					phoneme_mode = true;
					source_index++;
					continue;
				}
			}

			if (IsAlpha(c)) {
				alpha_count++;
				if (!IsAlpha(prev_out) || (tr->langopts.ideographs && ((c > 0x3040) || (prev_out > 0x3040)))) {
					if (wcschr(tr->punct_within_word, prev_out) == 0)
						letter_count = 0; // don't reset count for an apostrophy within a word

					if ((prev_out != ' ') && (wcschr(tr->punct_within_word, prev_out) == 0)) {
						// start of word, insert space if not one there already
						c = ' ';
						space_inserted = true;

						if (!IsBracket(prev_out)) // ?? perhaps only set FLAG_NOSPACE for . - /  (hyphenated words, URLs, etc)
							next_word_flags |= FLAG_NOSPACE;
					} else {
						if (iswupper(c))
							word_flags |= FLAG_FIRST_UPPER;

						if ((prev_out == ' ') && iswdigit(sbuf[ix-2]) && !iswdigit(prev_in)) {
							// word, following a number, but with a space between
							// Add an extra space, to distinguish "2 a" from "2a"
							sbuf[ix++] = ' ';
							words[word_count].start++;
						}
					}
				}

				if (c != ' ') {
					letter_count++;

					if (tr->letter_bits_offset > 0) {
						if (((c < 0x250) && (prev_out >= tr->letter_bits_offset)) ||
						    ((c >= tr->letter_bits_offset) && (letter_count > 1) && (prev_out < 0x250))) {
							// Don't mix native and Latin characters in the same word
							// Break into separate words
							if (IsAlpha(prev_out)) {
								c = ' ';
								space_inserted = true;
								word_flags |= FLAG_HYPHEN_AFTER;
								next_word_flags |= FLAG_HYPHEN;
							}
						}
					}
				}

				if (iswupper(c)) {
					c = towlower2(c, tr);

					if (tr->langopts.param[LOPT_CAPS_IN_WORD]) {
						if (syllable_marked == false) {
							char_inserted = c;
							c = 0x2c8; // stress marker
							syllable_marked = true;
						}
					} else {
						if (iswlower(prev_in)) {
							// lower case followed by upper case, possibly CamelCase
							if ((prev_out != ' ') && UpperCaseInWord(tr, &sbuf[ix], c) == 0)
							{ // start a new word
								c = ' ';
								space_inserted = true;
								prev_in_save = c;
							}
						} else if ((c != ' ') && iswupper(prev_in) && iswlower(next_in)) {
							int next2_in;
							utf8_in(&next2_in, &source[source_index + next_in_nbytes]);

							if ((tr->translator_name == L('n', 'l')) && (letter_count == 2) && (c == 'j') && (prev_in == 'I')) {
								// Dutch words may capitalise initial IJ, don't split
							} else if ((prev_out != ' ') && IsAlpha(next2_in)) {
								// changing from upper to lower case, start new word at the last uppercase, if 3 or more letters
								c = ' ';
								space_inserted = true;
								prev_in_save = c;
								next_word_flags |= FLAG_NOSPACE;
							}
						}
					}
				} else {
					if ((all_upper_case) && (letter_count > 2)) {
						// Flag as plural only English
						if (tr->translator_name == L('e', 'n') && (c == 's') && (next_in == ' ')) {
							c = ' ';
							all_upper_case |= FLAG_HAS_PLURAL;

							if (sbuf[ix-1] == '\'')
								sbuf[ix-1] = ' ';
						} else
							all_upper_case = 0; // current word contains lower case letters, not "'s"
					} else
						all_upper_case = 0;
				}
			} else if (c == '-') {
				if (!IsSpace(prev_in) && IsAlpha(next_in)) {
					if (prev_out != ' ') {
						// previous 'word' not yet ended (not alpha or numeric), start new word now.
						c = ' ';
						space_inserted = true;
					} else {
						// '-' between two letters is a hyphen, treat as a space
						word_flags |= FLAG_HYPHEN;
						if (word_count > 0)
							words[word_count-1].flags |= FLAG_HYPHEN_AFTER;
						c = ' ';
					}
				} else if ((prev_in == ' ') && (next_in == ' ')) {
					// ' - ' dash between two spaces, treat as pause
					c = ' ';
					pre_pause_add = 4;
				} else if (next_in == '-') {
					// double hyphen, treat as pause
					source_index++;
					c = ' ';
					pre_pause_add = 4;
				} else if ((prev_out == ' ') && IsAlpha(prev_out2) && !IsAlpha(prev_in)) {
					// insert extra space between a word + space + hyphen, to distinguish 'a -2' from 'a-2'
					sbuf[ix++] = ' ';
					words[word_count].start++;
				}
			} else if (c == '.') {
				if (prev_out == '.') {
					// multiple dots, separate by spaces. Note >3 dots has been replaced by elipsis
					c = ' ';
					space_inserted = true;
				} else if ((word_count > 0) && !(words[word_count-1].flags & FLAG_NOSPACE) && IsAlpha(prev_in)) {
					// dot after a word, with space following, probably an abbreviation
					words[word_count-1].flags |= FLAG_HAS_DOT;

					if (IsSpace(next_in) || (next_in == '-'))
						c = ' '; // remove the dot if it's followed by a space or hyphen, so that it's not pronounced
				}
			} else if (c == '\'') {
				if (((prev_in == '.' && next_in == 's') || iswalnum(prev_in)) && IsAlpha(next_in)) {
					// between two letters, or in an abbreviation (eg. u.s.a.'s). Consider the apostrophe as part of the word
					single_quoted = false;
				} else if ((tr->langopts.param[LOPT_APOSTROPHE] & 1) && IsAlpha(next_in))
					single_quoted = false; // apostrophe at start of word is part of the word
				else if ((tr->langopts.param[LOPT_APOSTROPHE] & 2) && IsAlpha(prev_in))
					single_quoted = false; // apostrophe at end of word is part of the word
				else if ((wcschr(tr->char_plus_apostrophe, prev_in) != 0) && (prev_out2 == ' ')) {
					// consider single character plus apostrophe as a word
					single_quoted = false;
					if (next_in == ' ')
						source_index++; // skip following space
				} else {
					if ((prev_out == 's') && (single_quoted == false)) {
						// looks like apostrophe after an 's'
						c = ' ';
					} else {
						if (IsSpace(prev_out))
							single_quoted = true;
						else
							single_quoted = false;

						pre_pause_add = 4; // single quote
						c = ' ';
					}
				}
			} else if (lookupwchar(breaks, c) != 0)
				c = ' '; // various characters to treat as space
			else if (iswdigit(c)) {
				if (tr->langopts.tone_numbers && IsAlpha(prev_out) && !IsDigit(next_in)) {
				} else if ((prev_out != ' ') && !iswdigit(prev_out)) {
					if ((prev_out != tr->langopts.decimal_sep) || ((decimal_sep_count == true) && (tr->langopts.decimal_sep == ','))) {
						c = ' ';
						space_inserted = true;
					} else
						decimal_sep_count = true;
				} else if ((prev_out == ' ') && IsAlpha(prev_out2) && !IsAlpha(prev_in)) {
					// insert extra space between a word and a number, to distinguish 'a 2' from 'a2'
					sbuf[ix++] = ' ';
					words[word_count].start++;
				}
			}
		}

		if (IsSpace(c)) {
			if (prev_out == ' ') {
				word_flags |= FLAG_MULTIPLE_SPACES;
				continue; // multiple spaces
			}

			if ((cc == 0x09) || (cc == 0x0a))
				next_word_flags |= FLAG_MULTIPLE_SPACES; // tab or newline, not a simple space

			if (space_inserted) {
				// count the number of characters since the start of the word
				j = 0;
				k = source_index - 1;
				while ((k >= source_index_word) && (charix[k] != 0)) {
					if (charix[k] > 0) // don't count initial bytes of multi-byte character
						j++;
					k--;
				}
				words[word_count].length = j;
			}

			source_index_word = source_index;

			// end of 'word'
			sbuf[ix++] = ' ';

			if ((word_count < N_CLAUSE_WORDS-1) && (ix > words[word_count].start)) {
				if (embedded_count > 0) {
					// there are embedded commands before this word
					embedded_list[embedded_ix-1] |= 0x80; // terminate list of commands for this word
					words[word_count].flags |= FLAG_EMBEDDED;
					embedded_count = 0;
				}
				if (alpha_count == 0) {
					all_upper_case &= ~FLAG_ALL_UPPER;
				}
				words[word_count].pre_pause = pre_pause;
				words[word_count].flags |= (all_upper_case | word_flags | word_emphasis);

				if (pre_pause > 0) {
					// insert an extra space before the word, to prevent influence from previous word across the pause
					for (j = ix; j > words[word_count].start; j--)
						sbuf[j] = sbuf[j-1];
					sbuf[j] = ' ';
					words[word_count].start++;
					ix++;
				}

				word_count++;
				words[word_count].start = ix;
				words[word_count].flags = 0;

				words[word_count].length = CalcWordLength(source_index, charix_top, charix, words, word_count);

				word_flags = next_word_flags;
				next_word_flags = 0;
				pre_pause = 0;
				all_upper_case = FLAG_ALL_UPPER;
				alpha_count = 0;
				syllable_marked = false;
			}

			if (space_inserted) {
				source_index = prev_source_index; // rewind to the previous character
				char_inserted = 0;
				space_inserted = false;
			}
		} else {
			if ((ix < (N_TR_SOURCE - 4)))
				ix += utf8_out(c, &sbuf[ix]);
		}
		if (pre_pause_add > pre_pause)
			pre_pause = pre_pause_add;
		pre_pause_add = 0;
	}

	if ((word_count == 0) && (embedded_count > 0)) {
		// add a null 'word' to carry the embedded command flag
		embedded_list[embedded_ix-1] |= 0x80;
		words[word_count].flags |= FLAG_EMBEDDED;
		word_count = 1;
	}

	tr->clause_end = &sbuf[ix-1];
	sbuf[ix] = 0;
	words[0].pre_pause = 0; // don't add extra pause at beginning of clause
	words[word_count].pre_pause = 8;
	if (word_count > 0) {
		ix = word_count-1;
		while ((ix > 0) && (IsBracket(sbuf[words[ix].start])))
			ix--; // the last word is a bracket, mark the previous word as last
		words[ix].flags |= FLAG_LAST_WORD;

		// FLAG_NOSPACE check to avoid recognizing  .mr  -mr
		if ((terminator & CLAUSE_DOT_AFTER_LAST_WORD) && !(words[word_count-1].flags & FLAG_NOSPACE))
			words[word_count-1].flags |= FLAG_HAS_DOT;
	}
	words[0].flags |= FLAG_FIRST_WORD;

	// Each TranslateWord2 may require up to 7 phonemes
	// and after this loop we require 2 phonemes
	for (ix = 0; ix < word_count && (n_ph_list2 < N_PHONEME_LIST-7-2); ix++) {
		int nx;
		int c_temp;
		char *pn;
		char *pw;
		char number_buf[150];
		WORD_TAB num_wtab[N_CLAUSE_WORDS]; // copy of 'words', when splitting numbers into parts

		// start speaking at a specified word position in the text?
		count_words++;
		if (skip_words > 0) {
			skip_words--;
			if (skip_words == 0)
				skipping_text = false;
		}
		if (skipping_text)
			continue;

		current_alphabet = NULL;

		// digits should have been converted to Latin alphabet ('0' to '9')
		word = pw = &sbuf[words[ix].start];

		if (iswdigit(word[0]) && (tr->langopts.break_numbers != BREAK_THOUSANDS)) {
			// Languages with 100000 numbers.  Remove thousands separators so that we can insert them again later
			pn = number_buf;
			while (pn < &number_buf[sizeof(number_buf)-20]) {
				if (iswdigit(*pw))
					*pn++ = *pw++;
				else if ((*pw == tr->langopts.thousands_sep) && (pw[1] == ' ')
				           && iswdigit(pw[2]) && (pw[3] != ' ') && (pw[4] != ' ')) { // don't allow only 1 or 2 digits in the final part
					pw += 2;
					ix++; // skip "word"
				} else {
					nx = pw - word;
					memset(word, ' ', nx);
					nx = pn - number_buf;
					memcpy(word, number_buf, nx);
					break;
				}
			}
			pw = word;
		}

		for (n_digits = 0; iswdigit(word[n_digits]); n_digits++) // count consecutive digits
			;

		if (n_digits > 4 && n_digits <= 32) {
			// word is entirely digits, insert commas and break into 3 digit "words"
			int nw = 0;

			number_buf[0] = ' ';
			number_buf[1] = ' ';
			number_buf[2] = ' ';
			pn = &number_buf[3];
			nx = n_digits;

			if ((n_digits > tr->langopts.max_digits) || (word[0] == '0'))
				words[ix].flags |= FLAG_INDIVIDUAL_DIGITS;

			while (pn < &number_buf[sizeof(number_buf)-20] && nw < N_CLAUSE_WORDS-2) {
				if (!IsDigit09(c = *pw++) && (c != tr->langopts.decimal_sep))
					break;

				*pn++ = c;
				nx--;
				if ((nx > 0) && (tr->langopts.break_numbers & (1U << nx))) {
					memcpy(&num_wtab[nw++], &words[ix], sizeof(WORD_TAB)); // copy the 'words' entry for each word of numbers

					if (tr->langopts.thousands_sep != ' ')
						*pn++ = tr->langopts.thousands_sep;
					*pn++ = ' ';

					if ((words[ix].flags & FLAG_INDIVIDUAL_DIGITS) == 0) {
						if (tr->langopts.break_numbers & (1 << (nx-1))) {
							// the next group only has 1 digits, make it three
							*pn++ = '0';
							*pn++ = '0';
						}
						if (tr->langopts.break_numbers & (1 << (nx-2))) {
							// the next group only has 2 digits (eg. Indian languages), make it three
							*pn++ = '0';
						}
					}
				}
			}
			pw--;
			memcpy(&num_wtab[nw], &words[ix], sizeof(WORD_TAB)*2); // the original number word, and the word after it

			for (j = 1; j <= nw; j++)
				num_wtab[j].flags &= ~(FLAG_MULTIPLE_SPACES | FLAG_EMBEDDED); // don't use these flags for subsequent parts when splitting a number

			// include the next few characters, in case there are an ordinal indicator or other suffix
			strncpy(pn, pw, 16);
			pn[16] = 0;
			nw = 0;

			for (pw = &number_buf[3]; pw < pn && nw < N_CLAUSE_WORDS;) {
				// keep wflags for each part, for FLAG_HYPHEN_AFTER
				dict_flags = TranslateWord2(tr, pw, &num_wtab[nw++], words[ix].pre_pause);
				while (pw < pn && *pw++ != ' ')
					;
				words[ix].pre_pause = 0;
			}
		} else {
			pre_pause = 0;

			dict_flags = TranslateWord2(tr, word, &words[ix], words[ix].pre_pause);

			if (pre_pause > words[ix+1].pre_pause) {
				words[ix+1].pre_pause = pre_pause;
				pre_pause = 0;
			}

			if (dict_flags & FLAG_SPELLWORD) {
				// redo the word, speaking single letters
				for (pw = word; *pw != ' ';) {
					memset(number_buf, 0, sizeof(number_buf));
					memset(number_buf+1, ' ', 9);
					nx = utf8_in(&c_temp, pw);
					memcpy(&number_buf[3], pw, nx);
					TranslateWord2(tr, &number_buf[3], &words[ix], 0);
					pw += nx;
				}
			}

			if ((dict_flags & (FLAG_ALLOW_DOT | FLAG_NEEDS_DOT)) && (ix == word_count - 1 - dictionary_skipwords) && (terminator & CLAUSE_DOT_AFTER_LAST_WORD)) {
				// probably an abbreviation such as Mr. or B. rather than end of sentence
				clause_pause = 10;
				if (tone_out != NULL)
					*tone_out = 4;
			}
		}

		if (dict_flags & FLAG_SKIPWORDS) {
			// dictionary indicates skip next word(s)
			while (dictionary_skipwords > 0) {
				words[ix+dictionary_skipwords].flags |= FLAG_DELETE_WORD;
				dictionary_skipwords--;
			}
		}
	}

	if (embedded_read < embedded_ix) {
		// any embedded commands not yet processed?
		Word_EmbeddedCmd();
	}

	for (ix = 0; ix < 2; ix++) {
		// terminate the clause with 2 PAUSE phonemes
		PHONEME_LIST2 *p2;
		p2 = &ph_list2[n_ph_list2 + ix];
		p2->phcode = phonPAUSE;
		p2->stresslevel = 0;
		p2->sourceix = source_index;
		p2->synthflags = 0;
	}
	n_ph_list2 += 2;

	if (Eof() && ((word_count == 0) || (option_endpause == 0)))
		clause_pause = 10;

	MakePhonemeList(tr, clause_pause, new_sentence2);
	phoneme_list[N_PHONEME_LIST].ph = NULL; // recognize end of phoneme_list array, in Generate()
	phoneme_list[N_PHONEME_LIST].sourceix = 1;

	if (embedded_count) { // ???? is this needed
		phoneme_list[n_phoneme_list-2].synthflags = SFLAG_EMBEDDED;
		embedded_list[embedded_ix-1] |= 0x80;
		embedded_list[embedded_ix] = 0x80;
	}

	new_sentence = false;
	if (terminator & CLAUSE_TYPE_SENTENCE)
		new_sentence = true; // next clause is a new sentence

	if (voice_change != NULL) {
		// return new voice name if an embedded voice change command terminated the clause
		if (terminator & CLAUSE_TYPE_VOICE_CHANGE)
			*voice_change = voice_change_name;
		else
			*voice_change = NULL;
	}
}

void TranslateClause(Translator *tr, int *tone_out, char **voice_change)
{
	TranslateClauseWithTerminator(tr, tone_out, voice_change, NULL);
}

static int CalcWordLength(int source_index, int charix_top, short int *charix, WORD_TAB *words, int word_count) {
	int j;
	int k;

	for (j = source_index; j < charix_top && charix[j] <= 0; j++); // skip blanks
	words[word_count].sourceix = charix[j];
	k = 0;
	while (charix[j] != 0) {
		// count the number of characters (excluding multibyte continuation bytes)
		if (charix[j++] != -1)
			k++;
	}
	return k;
	}

static void CombineFlag(Translator *tr, WORD_TAB *wtab, char *word, int *flags, unsigned char *p, char *word_phonemes) {
	// combine a preposition with the following word


	int sylimit; // max. number of syllables in a word to be combined with a preceding preposition
	sylimit = tr->langopts.param[LOPT_COMBINE_WORDS];


	char *p2;
	p2 = word;
	while (*p2 != ' ') p2++;

	bool ok = true;
	int c_word2;

	utf8_in(&c_word2, p2+1); // first character of the next word;

	if (!iswalpha(c_word2))
		ok = false;

	int flags2[2];
    flags2[0] = 0;


	if (ok) {
		char ph_buf[N_WORD_PHONEMES];
		strcpy(ph_buf, word_phonemes);

		flags2[0] = TranslateWord(tr, p2+1, wtab+1, NULL);
		if ((flags2[0] & FLAG_WAS_UNPRONOUNCABLE) || (word_phonemes[0] == phonSWITCH))
			ok = false;

		if ((sylimit & 0x100) && ((flags2[0] & FLAG_ALT_TRANS) == 0)) {
			// only if the second word has $alt attribute
			ok = false;
		}

		if ((sylimit & 0x200) && ((wtab+1)->flags & FLAG_LAST_WORD)) {
			// not if the next word is end-of-sentence
			ok = false;
		}

		if (ok == false)
			strcpy(word_phonemes, ph_buf);
	}

	if (ok) {
		*p2 = '-'; // replace next space by hyphen
		wtab[0].flags &= ~FLAG_ALL_UPPER; // prevent it being considered an abbreviation
		*flags = TranslateWord(translator, word, wtab, NULL); // translate the combined word
		if ((sylimit > 0) && (CountSyllables(p) > (sylimit & 0x1f))) {
			// revert to separate words
			*p2 = ' ';
			*flags = TranslateWord(translator, word, wtab, NULL);
		} else {
			if (*flags == 0)
				*flags = flags2[0]; // no flags for the combined word, so use flags from the second word eg. lang-hu "nem december 7-e"
			*flags |= FLAG_SKIPWORDS;
			dictionary_skipwords = 1;
		}
	}
}

static void SwitchLanguage(char *word, char *word_phonemes) {
	char lang_name[12];
	int ix;

	word += 3;

	for (ix = 0;;) {
		int  c1;
		c1 = *word++;
		if ((c1 == ' ') || (c1 == 0))
			break;
		lang_name[ix++] = tolower(c1);
	}
	lang_name[ix] = 0;

	if ((ix = LookupPhonemeTable(lang_name)) > 0) {
		SelectPhonemeTable(ix);
		word_phonemes[0] = phonSWITCH;
		word_phonemes[1] = ix;
		word_phonemes[2] = 0;
	}
}

void InitText(int control)
{
	count_sentences = 0;
	count_words = 0;
	end_character_position = 0;
	skip_sentences = 0;
	skip_marker[0] = 0;
	skip_words = 0;
	skip_characters = 0;
	skipping_text = false;
	new_sentence = true;

	option_sayas = 0;
	option_sayas2 = 0;
	option_emphasis = 0;
	word_emphasis = 0;
	embedded_flag = 0;

	InitText2();

	if ((control & espeakKEEP_NAMEDATA) == 0)
		InitNamedata();
}
