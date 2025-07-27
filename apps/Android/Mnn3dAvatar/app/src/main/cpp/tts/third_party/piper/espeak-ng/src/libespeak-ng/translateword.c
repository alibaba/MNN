
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
#include "translateword.h"
#include "common.h"               // for strncpy0
#include "dictionary.h"           // for TranslateRules, LookupDictList
#include "numbers.h"              // for SetSpellingStress, ...
#include "phoneme.h"              // for phonSWITCH, PHONEME_TAB, phonPAUSE_...
#include "readclause.h"           // for towlower2
#include "synthdata.h"            // for SelectPhonemeTable, LookupPhonemeTable
#include "ucd/ucd.h"              // for ucd_toupper
#include "voice.h"                // for voice, voice_t


static void addPluralSuffixes(int flags, Translator *tr, char last_char, char *word_phonemes);
static void ApplySpecialAttribute2(Translator *tr, char *phonemes, int dict_flags);
static void ChangeWordStress(Translator *tr, char *word, int new_stress);
static int CheckDottedAbbrev(char *word1);
static int NonAsciiNumber(int letter);
static char *SpeakIndividualLetters(Translator *tr, char *word, char *phonemes, int spell_word, const ALPHABET *current_alphabet, char word_phonemes[]);
static int TranslateLetter(Translator *tr, char *word, char *phonemes, int control, const ALPHABET *current_alphabet);
static int Unpronouncable(Translator *tr, char *word, int posn);
static int Unpronouncable2(Translator *tr, char *word);

int TranslateWord3(Translator *tr, char *word_start, WORD_TAB *wtab, char *word_out, bool *any_stressed_words, ALPHABET *current_alphabet, char word_phonemes[], size_t size_word_phonemes)
{
	// word1 is terminated by space (0x20) character

	char *word1;
	int word_length;
	int ix;
	char *p;
	int pfix;
	int n_chars;
	unsigned int dictionary_flags[2];
	unsigned int dictionary_flags2[2];
	int end_type = 0;
	int end_type1 = 0;
	int prefix_type = 0;
	int prefix_stress;
	char *wordx;
	char phonemes[N_WORD_PHONEMES];
	char phonemes2[N_WORD_PHONEMES];
	char prefix_phonemes[N_WORD_PHONEMES];
	char unpron_phonemes[N_WORD_PHONEMES];
	char end_phonemes[N_WORD_PHONEMES];
	char end_phonemes2[N_WORD_PHONEMES];
	char word_copy[N_WORD_BYTES];
	char word_copy2[N_WORD_BYTES];
	int word_copy_length;
	char prefix_chars[0x3f + 2];
	bool found = false;
	int end_flags;
	int c_temp; // save a character byte while we temporarily replace it with space
	int first_char;
	int last_char = 0;
	int prefix_flags = 0;
	bool more_suffixes;
	bool confirm_prefix;
	int spell_word;
	int emphasize_allcaps = 0;
	int wflags;
	int was_unpronouncable = 0;
	int loopcount;
	int add_suffix_phonemes = 0;
	WORD_TAB wtab_null[8];

	if (wtab == NULL) {
		memset(wtab_null, 0, sizeof(wtab_null));
		wtab = wtab_null;
	}
	wflags = wtab->flags;

	dictionary_flags[0] = 0;
	dictionary_flags[1] = 0;
	dictionary_flags2[0] = 0;
	dictionary_flags2[1] = 0;
	dictionary_skipwords = 0;

	phonemes[0] = 0;
	unpron_phonemes[0] = 0;
	prefix_phonemes[0] = 0;
	end_phonemes[0] = 0;

	if (tr->data_dictlist == NULL) {
		// dictionary is not loaded
		word_phonemes[0] = 0;
		return 0;
	}

	// count the length of the word
	word1 = word_start;
	if (*word1 == ' ') word1++; // possibly a dot was replaced by space:  $dot
	wordx = word1;

	utf8_in(&first_char, wordx);
	word_length = 0;
	while ((*wordx != 0) && (*wordx != ' ')) {
		wordx += utf8_in(&last_char, wordx);
		word_length++;
	}

	word_copy_length = wordx - word_start;
	if (word_copy_length >= N_WORD_BYTES)
		word_copy_length = N_WORD_BYTES-1;
	memcpy(word_copy2, word_start, word_copy_length);

	spell_word = 0;

	if ((word_length == 1) && (wflags & FLAG_TRANSLATOR2)) {
		// retranslating a 1-character word using a different language, say its name
		utf8_in(&c_temp, wordx+1); // the next character
		if (!IsAlpha(c_temp) || (AlphabetFromChar(last_char) != AlphabetFromChar(c_temp)))
			spell_word = 1;
	}

	if (option_sayas == SAYAS_KEY) {
		if (word_length == 1)
			spell_word = 4;
		else {
			// is there a translation for this keyname ?
			word1--;
			*word1 = '_'; // prefix keyname with '_'
			found = LookupDictList(tr, &word1, phonemes, dictionary_flags, 0, wtab);
		}
	}

	// try an initial lookup in the dictionary list, we may find a pronunciation specified, or
	// we may just find some flags
	if (option_sayas & 0x10) {
		// SAYAS_CHAR, SAYAS_GYLPH, or SAYAS_SINGLE_CHAR
		spell_word = option_sayas & 0xf; // 2,3,4
	} else {
		if (!found)
			found = LookupDictList(tr, &word1, phonemes, dictionary_flags, FLAG_ALLOW_TEXTMODE, wtab);   // the original word

		if ((dictionary_flags[0] & (FLAG_ALLOW_DOT | FLAG_NEEDS_DOT)) && (wordx[1] == '.'))
			wordx[1] = ' '; // remove a Dot after this word

		if (dictionary_flags[0] & FLAG_TEXTMODE) {
			if (word_out != NULL)
				strcpy(word_out, word1);

			return dictionary_flags[0];
		} else if ((found == false) && (dictionary_flags[0] & FLAG_SKIPWORDS) && !(dictionary_flags[0] & FLAG_ABBREV)) {
			// grouped words, but no translation.  Join the words with hyphens.
			wordx = word1;
			ix = 0;
			while (ix < dictionary_skipwords) {
				if (*wordx == ' ') {
					*wordx = '-';
					ix++;
				}
				wordx++;
			}
		}

		if ((word_length == 1) && (dictionary_skipwords == 0)) {
			// is this a series of single letters separated by dots?
			if (CheckDottedAbbrev(word1)) {
				dictionary_flags[0] = 0;
				dictionary_flags[1] = 0;
				spell_word = 1;
				if (dictionary_skipwords)
					dictionary_flags[0] = FLAG_SKIPWORDS;
			}
		}

		if (phonemes[0] == phonSWITCH) {
			// change to another language in order to translate this word
			strcpy(word_phonemes, phonemes);
			return 0;
		}

		if (!found && (dictionary_flags[0] & FLAG_ABBREV)) {
			// the word has $abbrev flag, but no pronunciation specified.  Speak as individual letters
			spell_word = 1;
		}

		if (!found && iswdigit(first_char)) {
			Lookup(tr, "_0lang", word_phonemes);
			if (word_phonemes[0] == phonSWITCH)
				return 0;

			if ((tr->langopts.numbers2 & NUM2_ENGLISH_NUMERALS) && !(wtab->flags & FLAG_CHAR_REPLACED)) {
				// for this language, speak English numerals (0-9) with the English voice
				sprintf(word_phonemes, "%c", phonSWITCH);
				return 0;
			}

			found = TranslateNumber(tr, word1, phonemes, phonemes + sizeof(phonemes), dictionary_flags, wtab, 0);
		}

		if (!found && ((wflags & FLAG_UPPERS) != FLAG_FIRST_UPPER)) {
			// either all upper or all lower case

			if ((tr->langopts.numbers & NUM_ROMAN) || ((tr->langopts.numbers & NUM_ROMAN_CAPITALS) && (wflags & FLAG_ALL_UPPER))) {
				if ((wflags & FLAG_LAST_WORD) || !(wtab[1].flags & FLAG_NOSPACE)) {
					// don't use Roman number if this word is not separated from the next word (eg. "XLTest")
					if ((found = TranslateRoman(tr, word1, phonemes, phonemes + sizeof(phonemes), wtab)) != 0)
						dictionary_flags[0] |= FLAG_ABBREV; // prevent emphasis if capitals
				}
			}
		}

		if ((wflags & FLAG_ALL_UPPER) && (word_length > 1) && iswalpha(first_char)) {
			if ((option_tone_flags & OPTION_EMPHASIZE_ALLCAPS) && !(dictionary_flags[0] & FLAG_ABBREV)) {
				// emphasize words which are in capitals
				emphasize_allcaps = FLAG_EMPHASIZED;
			} else if (!found && !(dictionary_flags[0] &  FLAG_SKIPWORDS) && (word_length < 4) && (tr->clause_lower_count > 3)
			           && (tr->clause_upper_count <= tr->clause_lower_count)) {
				// An upper case word in a lower case clause. This could be an abbreviation.
				spell_word = 1;
			}
		}
	}

	if (spell_word > 0) {
		// Speak as individual letters
		phonemes[0] = 0;

		if (SpeakIndividualLetters(tr, word1, phonemes, spell_word, current_alphabet, word_phonemes) == NULL) {
			if (word_length > 1)
				return FLAG_SPELLWORD; // a mixture of languages, retranslate as individual letters, separated by spaces
			return 0;
		}
		strcpy(word_phonemes, phonemes);
		if (wflags & FLAG_TRANSLATOR2)
			return 0;

		addPluralSuffixes(wflags, tr, last_char, word_phonemes);
		return dictionary_flags[0] & FLAG_SKIPWORDS; // for "b.c.d"
	} else if (found == false) {
		// word's pronunciation is not given in the dictionary list, although
		// dictionary_flags may have ben set there

		int posn;
		bool non_initial = false;
		int length;

		posn = 0;
		length = 999;
		wordx = word1;

		while (((length < 3) && (length > 0)) || (word_length > 1 && Unpronouncable(tr, wordx, posn))) {
			// This word looks "unpronouncable", so speak letters individually until we
			// find a remainder that we can pronounce.
			was_unpronouncable = FLAG_WAS_UNPRONOUNCABLE;
			emphasize_allcaps = 0;

			if (wordx[0] == '\'')
				break;

			if (posn > 0)
				non_initial = true;

			wordx += TranslateLetter(tr, wordx, unpron_phonemes, non_initial, current_alphabet);
			posn++;
			if (unpron_phonemes[0] == phonSWITCH) {
				// change to another language in order to translate this word
				strcpy(word_phonemes, unpron_phonemes);
				if (strcmp(&unpron_phonemes[1], ESPEAKNG_DEFAULT_VOICE) == 0)
					return FLAG_SPELLWORD; // _^_en must have been set in TranslateLetter(), not *_rules which uses only _^_
				return 0;
			}

			length = 0;
			while (wordx[length] != ' ') length++;
		}
		SetSpellingStress(tr, unpron_phonemes, 0, posn);

		// anything left ?
		if (*wordx != ' ') {
			if ((unpron_phonemes[0] != 0) && (wordx[0] != '\'')) {
				// letters which have been spoken individually from affecting the pronunciation of the pronuncable part
				wordx[-1] = ' ';
			}

			// Translate the stem
			end_type = TranslateRules(tr, wordx, phonemes, N_WORD_PHONEMES, end_phonemes, wflags, dictionary_flags);

			if (phonemes[0] == phonSWITCH) {
				// change to another language in order to translate this word
				strcpy(word_phonemes, phonemes);
				return 0;
			}

			if ((phonemes[0] == 0) && (end_phonemes[0] == 0)) {
				int wc;
				// characters not recognised, speak them individually
				// ?? should we say super/sub-script numbers and letters here?
				utf8_in(&wc, wordx);
				if ((word_length == 1) && (IsAlpha(wc) || IsSuperscript(wc))) {
					if ((wordx = SpeakIndividualLetters(tr, wordx, phonemes, spell_word, current_alphabet, word_phonemes)) == NULL)
						return 0;
					strcpy(word_phonemes, phonemes);
					return 0;
				}
			}

			c_temp = wordx[-1];

			found = false;
			confirm_prefix = true;
			for (loopcount = 0; (loopcount < 50) && (end_type & SUFX_P); loopcount++) {
				// Found a standard prefix, remove it and retranslate
				// loopcount guards against an endless loop
				if (confirm_prefix && !(end_type & SUFX_B)) {
					int end2;
					char end_phonemes2[N_WORD_PHONEMES];

					// remove any standard suffix and confirm that the prefix is still recognised
					phonemes2[0] = 0;
					end2 = TranslateRules(tr, wordx, phonemes2, N_WORD_PHONEMES, end_phonemes2, wflags|FLAG_NO_PREFIX|FLAG_NO_TRACE, dictionary_flags);
					if (end2) {
						RemoveEnding(tr, wordx, end2, word_copy);
						end_type = TranslateRules(tr, wordx, phonemes, N_WORD_PHONEMES, end_phonemes, wflags|FLAG_NO_TRACE, dictionary_flags);
						memcpy(wordx, word_copy, strlen(word_copy));
						if ((end_type & SUFX_P) == 0) {
							// after removing the suffix, the prefix is no longer recognised.
							// Keep the suffix, but don't use the prefix
							end_type = end2;
							strcpy(phonemes, phonemes2);
							strcpy(end_phonemes, end_phonemes2);
							if (option_phonemes & espeakPHONEMES_TRACE) {
								DecodePhonemes(end_phonemes, end_phonemes2);
								fprintf(f_trans, "  suffix [%s]\n\n", end_phonemes2);
							}
						}
						confirm_prefix = false;
						continue;
					}
				}

				prefix_type = end_type;

				if (prefix_type & SUFX_V)
					tr->expect_verb = 1; // use the verb form of the word

				wordx[-1] = c_temp;

				if ((prefix_type & SUFX_B) == 0) {
					for (ix = (prefix_type & 0xf); ix > 0; ix--) { // num. of characters to remove
						wordx++;
						while ((*wordx & 0xc0) == 0x80) wordx++; // for multibyte characters
					}
				} else {
					pfix = 1;
					prefix_chars[0] = 0;
					n_chars = prefix_type & 0x3f;

					for (ix = 0; ix < n_chars; ix++) { // num. of bytes to remove
						prefix_chars[pfix++] = *wordx++;

						if ((prefix_type & SUFX_B) && (ix == (n_chars-1)))
							prefix_chars[pfix-1] = 0; // discard the last character of the prefix, this is the separator character
					}
					prefix_chars[pfix] = 0;
				}
				c_temp = wordx[-1];
				wordx[-1] = ' ';
				confirm_prefix = true;
				wflags |= FLAG_PREFIX_REMOVED;

				if (prefix_type & SUFX_B) {
					// SUFX_B is used for Turkish, tr_rules contains " ' (Pb"
					// examine the prefix part
					char *wordpf;
					char prefix_phonemes2[12];

					strncpy0(prefix_phonemes2, end_phonemes, sizeof(prefix_phonemes2));
					wordpf = &prefix_chars[1];
					strcpy(prefix_phonemes, phonemes);

					// look for stress marker or $abbrev
					found = LookupDictList(tr, &wordpf, phonemes, dictionary_flags, 0, wtab);
					if (found)
						strcpy(prefix_phonemes, phonemes);
					if (dictionary_flags[0] & FLAG_ABBREV) {
						prefix_phonemes[0] = 0;
						SpeakIndividualLetters(tr, wordpf, prefix_phonemes, 1, current_alphabet, word_phonemes);
					}
				} else
					strcat(prefix_phonemes, end_phonemes);
				end_phonemes[0] = 0;

				end_type = 0;
				found = LookupDictList(tr, &wordx, phonemes, dictionary_flags2, SUFX_P, wtab); // without prefix
				if (dictionary_flags[0] == 0) {
					dictionary_flags[0] = dictionary_flags2[0];
					dictionary_flags[1] = dictionary_flags2[1];
				} else
					prefix_flags = 1;
				if (found == false) {
					end_type = TranslateRules(tr, wordx, phonemes, N_WORD_PHONEMES, end_phonemes, wflags & (FLAG_HYPHEN_AFTER | FLAG_PREFIX_REMOVED), dictionary_flags);

					if (phonemes[0] == phonSWITCH) {
						// change to another language in order to translate this word
						wordx[-1] = c_temp;
						strcpy(word_phonemes, phonemes);
						return 0;
					}
				}
			}

			if ((end_type != 0) && !(end_type & SUFX_P)) {
				end_type1 = end_type;
				strcpy(phonemes2, phonemes);

				// The word has a standard ending, re-translate without this ending
				end_flags = RemoveEnding(tr, wordx, end_type, word_copy);
				more_suffixes = true;

				while (more_suffixes) {
					more_suffixes = false;
					phonemes[0] = 0;

					if (prefix_phonemes[0] != 0) {
						// lookup the stem without the prefix removed
						wordx[-1] = c_temp;
						found = LookupDictList(tr, &word1, phonemes, dictionary_flags2, end_flags, wtab);  // include prefix, but not suffix
						wordx[-1] = ' ';
						if (phonemes[0] == phonSWITCH) {
							// change to another language in order to translate this word
							memcpy(wordx, word_copy, strlen(word_copy));
							strcpy(word_phonemes, phonemes);
							return 0;
						}
						if (dictionary_flags[0] == 0) {
							dictionary_flags[0] = dictionary_flags2[0];
							dictionary_flags[1] = dictionary_flags2[1];
						}
						if (found)
							prefix_phonemes[0] = 0; // matched whole word, don't need prefix now

						if ((found == false) && (dictionary_flags2[0] != 0))
							prefix_flags = 1;
					}
					if (found == false) {
						found = LookupDictList(tr, &wordx, phonemes, dictionary_flags2, end_flags, wtab);  // without prefix and suffix
						if (phonemes[0] == phonSWITCH) {
							// change to another language in order to translate this word
							memcpy(wordx, word_copy, strlen(word_copy));
							strcpy(word_phonemes, phonemes);
							return 0;
						}

						if (dictionary_flags[0] == 0) {
							dictionary_flags[0] = dictionary_flags2[0];
							dictionary_flags[1] = dictionary_flags2[1];
						}
					}
					if (found == false) {
						if (end_type & SUFX_Q) {
							// don't retranslate, use the original lookup result
							strcpy(phonemes, phonemes2);
						} else {
							if (end_flags & FLAG_SUFX)
								wflags |= FLAG_SUFFIX_REMOVED;
							if (end_type & SUFX_A)
								wflags |= FLAG_SUFFIX_VOWEL;

							if (end_type & SUFX_M) {
								// allow more suffixes before this suffix
								strcpy(end_phonemes2, end_phonemes);
								end_type = TranslateRules(tr, wordx, phonemes, N_WORD_PHONEMES, end_phonemes, wflags, dictionary_flags);
								strcat(end_phonemes, end_phonemes2); // add the phonemes for the previous suffixes after this one

								if ((end_type != 0) && !(end_type & SUFX_P)) {
									// there is another suffix
									end_flags = RemoveEnding(tr, wordx, end_type, NULL);
									more_suffixes = true;
								}
							} else {
								// don't remove any previous suffix
								TranslateRules(tr, wordx, phonemes, N_WORD_PHONEMES, NULL, wflags, dictionary_flags);
								end_type = 0;
							}

							if (phonemes[0] == phonSWITCH) {
								// change to another language in order to translate this word
								strcpy(word_phonemes, phonemes);
								memcpy(wordx, word_copy, strlen(word_copy));
								wordx[-1] = c_temp;
								return 0;
							}
						}
					}
				}


				if ((end_type1 & SUFX_T) == 0) {
					// the default is to add the suffix and then determine the word's stress pattern
					AppendPhonemes(tr, phonemes, N_WORD_PHONEMES, end_phonemes);
					end_phonemes[0] = 0;
				}
				memcpy(wordx, word_copy, strlen(word_copy));
			}

			wordx[-1] = c_temp;
		}
	}

	addPluralSuffixes(wflags, tr, last_char, word_phonemes);
	wflags |= emphasize_allcaps;

	// determine stress pattern for this word

	add_suffix_phonemes = 0;
	if (end_phonemes[0] != 0)
		add_suffix_phonemes = 2;

	prefix_stress = 0;
	for (p = prefix_phonemes; *p != 0; p++) {
		if ((*p == phonSTRESS_P) || (*p == phonSTRESS_P2))
			prefix_stress = *p;
	}
	if (prefix_flags || (prefix_stress != 0)) {
		if ((tr->langopts.param[LOPT_PREFIXES]) || (prefix_type & SUFX_T)) {
			char *p;
			// German, keep a secondary stress on the stem
			SetWordStress(tr, phonemes, dictionary_flags, 3, 0);

			// reduce all but the first primary stress
			ix = 0;
			for (p = prefix_phonemes; *p != 0; p++) {
				if (*p == phonSTRESS_P) {
					if (ix == 0)
						ix = 1;
					else
						*p = phonSTRESS_3;
				}
			}
			snprintf(word_phonemes, size_word_phonemes, "%s%s%s", unpron_phonemes, prefix_phonemes, phonemes);

			word_phonemes[N_WORD_PHONEMES-1] = 0;
			SetWordStress(tr, word_phonemes, dictionary_flags, -1, 0);
		} else {
			// stress position affects the whole word, including prefix
			snprintf(word_phonemes, size_word_phonemes, "%s%s%s", unpron_phonemes, prefix_phonemes, phonemes);
			word_phonemes[N_WORD_PHONEMES-1] = 0;
			SetWordStress(tr, word_phonemes, dictionary_flags, -1, 0);
		}
	} else {
		SetWordStress(tr, phonemes, dictionary_flags, -1, add_suffix_phonemes);
		snprintf(word_phonemes, size_word_phonemes, "%s%s%s", unpron_phonemes, prefix_phonemes, phonemes);
		word_phonemes[N_WORD_PHONEMES-1] = 0;
	}

	if (end_phonemes[0] != 0) {
		// a suffix had the SUFX_T option set, add the suffix after the stress pattern has been determined
		ix = strlen(word_phonemes);
		end_phonemes[N_WORD_PHONEMES-1-ix] = 0; // ensure no buffer overflow
		strcpy(&word_phonemes[ix], end_phonemes);
	}

	if (wflags & FLAG_LAST_WORD) {
		// don't use $brk pause before the last word of a sentence
		// (but allow it for emphasis, see below
		dictionary_flags[0] &= ~FLAG_PAUSE1;
	}

	if ((wflags & FLAG_HYPHEN) && (tr->langopts.stress_flags & S_HYPEN_UNSTRESS))
		ChangeWordStress(tr, word_phonemes, 3);
	else if (wflags & FLAG_EMPHASIZED2) {
		// A word is indicated in the source text as stressed
		// Give it stress level 6 (for the intonation module)
		ChangeWordStress(tr, word_phonemes, 6);

		if (wflags & FLAG_EMPHASIZED)
			dictionary_flags[0] |= FLAG_PAUSE1; // precede by short pause
	} else if (wtab[dictionary_skipwords].flags & FLAG_LAST_WORD) {
		// the word has attribute to stress or unstress when at end of clause
		if (dictionary_flags[0] & (FLAG_STRESS_END | FLAG_STRESS_END2))
			ChangeWordStress(tr, word_phonemes, 4);
		else if ((dictionary_flags[0] & FLAG_UNSTRESS_END) && (any_stressed_words))
			ChangeWordStress(tr, word_phonemes, 3);
	}

	// dictionary flags for this word give a clue about which alternative pronunciations of
	// following words to use.
	if (end_type1 & SUFX_F) {
		// expect a verb form, with or without -s suffix
		tr->expect_verb = 2;
		tr->expect_verb_s = 2;
	}

	if (dictionary_flags[1] & FLAG_PASTF) {
		// expect perfect tense in next two words
		tr->expect_past = 3;
		tr->expect_verb = 0;
		tr->expect_noun = 0;
	} else if (dictionary_flags[1] & FLAG_VERBF) {
		// expect a verb in the next word
		tr->expect_verb = 2;
		tr->expect_verb_s = 0; // verb won't have -s suffix
		tr->expect_noun = 0;
	} else if (dictionary_flags[1] & FLAG_VERBSF) {
		// expect a verb, must have a -s suffix
		tr->expect_verb = 0;
		tr->expect_verb_s = 2;
		tr->expect_past = 0;
		tr->expect_noun = 0;
	} else if (dictionary_flags[1] & FLAG_NOUNF) {
		// not expecting a verb next
		tr->expect_noun = 2;
		tr->expect_verb = 0;
		tr->expect_verb_s = 0;
		tr->expect_past = 0;
	}

	if ((wordx[0] != 0) && (!(dictionary_flags[1] & FLAG_VERB_EXT))) {
		if (tr->expect_verb > 0)
			tr->expect_verb--;

		if (tr->expect_verb_s > 0)
			tr->expect_verb_s--;

		if (tr->expect_noun > 0)
			tr->expect_noun--;

		if (tr->expect_past > 0)
			tr->expect_past--;
	}

	if ((word_length == 1) && (tr->translator_name == L('e', 'n')) && iswalpha(first_char) && (first_char != 'i')) {
		// English Specific !!!!
		// any single letter before a dot is an abbreviation, except 'I'
		dictionary_flags[0] |= FLAG_ALLOW_DOT;
	}

	if ((tr->langopts.param[LOPT_ALT] & 2) && ((dictionary_flags[0] & (FLAG_ALT_TRANS | FLAG_ALT2_TRANS)) != 0))
		ApplySpecialAttribute2(tr, word_phonemes, dictionary_flags[0]);

	dictionary_flags[0] |= was_unpronouncable;
	memcpy(word_start, word_copy2, word_copy_length);
	return dictionary_flags[0];
}


void ApplySpecialAttribute2(Translator *tr, char *phonemes, int dict_flags)
{
	// apply after the translation is complete
	int len;


	len = strlen(phonemes);

	if (tr->langopts.param[LOPT_ALT] & 2) {
		for (int ix = 0; ix < (len-1); ix++) {
			if (phonemes[ix] == phonSTRESS_P) {
				char *p;
				p = &phonemes[ix+1];
				if ((dict_flags & FLAG_ALT2_TRANS) != 0) {
					if (*p == PhonemeCode('E'))
						*p = PhonemeCode('e');
					if (*p == PhonemeCode('O'))
						*p = PhonemeCode('o');
				} else {
					if (*p == PhonemeCode('e'))
						*p = PhonemeCode('E');
					if (*p == PhonemeCode('o'))
						*p = PhonemeCode('O');
				}
				break;
			}
		}
	}
}


static void ChangeWordStress(Translator *tr, char *word, int new_stress)
{
	int ix;
	unsigned char *p;
	int max_stress;
	int vowel_count; // num of vowels + 1
	int stressed_syllable = 0; // position of stressed syllable
	unsigned char phonetic[N_WORD_PHONEMES];
	signed char vowel_stress[N_WORD_PHONEMES/2];

	strcpy((char *)phonetic, word);
	max_stress = GetVowelStress(tr, phonetic, vowel_stress, &vowel_count, &stressed_syllable, 0);

	if (new_stress >= STRESS_IS_PRIMARY) {
		// promote to primary stress
		for (ix = 1; ix < vowel_count; ix++) {
			if (vowel_stress[ix] >= max_stress) {
				vowel_stress[ix] = new_stress;
				break;
			}
		}
	} else {
		// remove primary stress
		for (ix = 1; ix < vowel_count; ix++) {
			if (vowel_stress[ix] > new_stress) // >= allows for diminished stress (=1)
				vowel_stress[ix] = new_stress;
		}
	}

	// write out phonemes
	ix = 1;
	p = phonetic;
	while (*p != 0) {
		if ((phoneme_tab[*p]->type == phVOWEL) && !(phoneme_tab[*p]->phflags & phNONSYLLABIC)) {
			if ((vowel_stress[ix] == STRESS_IS_DIMINISHED) || (vowel_stress[ix] > STRESS_IS_UNSTRESSED))
				*word++ = stress_phonemes[(unsigned char)vowel_stress[ix]];

			ix++;
		}
		*word++ = *p++;
	}
	*word = 0;
}

static char *SpeakIndividualLetters(Translator *tr, char *word, char *phonemes, int spell_word, const ALPHABET *current_alphabet, char word_phonemes[])
{
	int posn = 0;
	int capitals = 0;
	bool non_initial = false;

	if (spell_word > 2)
		capitals = 2; // speak 'capital'
	if (spell_word > 1)
		capitals |= 4; // speak character code for unknown letters

	while ((*word != ' ') && (*word != 0)) {
		word += TranslateLetter(tr, word, phonemes, capitals | non_initial, current_alphabet);
		posn++;
		non_initial = true;
		if (phonemes[0] == phonSWITCH) {
			// change to another language in order to translate this word
			strcpy(word_phonemes, phonemes);
			return NULL;
		}
	}
	SetSpellingStress(tr, phonemes, spell_word, posn);
	return word;
}


static const char *const hex_letters[] = {"'e:j",	"b'i:",	"s'i:",	"d'i:",	"'i:",	"'ef"};
static const char *const modifiers[] = { NULL, "_sub", "_sup", NULL };
// unicode ranges for non-ascii digits 0-9 (these must be in ascending order)
static const int number_ranges[] = {
	0x660, 0x6f0, // arabic
	0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, // indic
	0xe50, 0xed0, 0xf20, 0x1040, 0x1090,
	0
};


static int TranslateLetter(Translator *tr, char *word, char *phonemes, int control, const ALPHABET *current_alphabet)
{
	// get pronunciation for an isolated letter
	// return number of bytes used by the letter
	// control bit 0:  a non-initial letter in a word
	//         bit 1:  say 'capital'
	//         bit 2:  say character code for unknown letters

	int n_bytes;
	int letter;
	int len;
	const ALPHABET *alphabet;
	int al_offset;
	int al_flags;
	int number;
	int phontab_1;
	char capital[30];
	char ph_buf[80];
	char ph_buf2[80];
	char ph_alphabet[80];
	char hexbuf[12];
	static const char pause_string[] = { phonPAUSE, 0 };

	ph_buf[0] = 0;
	ph_alphabet[0] = 0;
	capital[0] = 0;
	phontab_1 = translator->phoneme_tab_ix;

	n_bytes = utf8_in(&letter, word);

	if ((letter & 0xfff00) == 0x0e000)
		letter &= 0xff; // uncode private usage area

	if (control & 2) {
		// include CAPITAL information
		if (iswupper(letter))
			Lookup(tr, "_cap", capital);
	}
	letter = towlower2(letter, tr);
	LookupLetter(tr, letter, word[n_bytes], ph_buf, control & 1);

	if (ph_buf[0] == 0) {
		// is this a subscript or superscript letter ?
		int c;
		if ((c = IsSuperscript(letter)) != 0) {
			letter = c & 0x3fff;

			const char *modifier;
			if ((control & 4 ) && ((modifier = modifiers[c >> 14]) != NULL)) {
				// don't say "superscript" during normal text reading
				Lookup(tr, modifier, capital);
				if (capital[0] == 0) {
					capital[2] = SetTranslator3(ESPEAKNG_DEFAULT_VOICE); // overwrites previous contents of translator3
					Lookup(translator3, modifier, &capital[3]);
					if (capital[3] != 0) {
						capital[0] = phonPAUSE;
						capital[1] = phonSWITCH;
						len = strlen(&capital[3]);
						capital[len+3] = phonSWITCH;
						capital[len+4] = phontab_1;
						capital[len+5] = 0;
					}
				}
			}
		}
		LookupLetter(tr, letter, word[n_bytes], ph_buf, control & 1);
	}

	if (ph_buf[0] == phonSWITCH) {
		strcpy(phonemes, ph_buf);
		return 0;
	}


	if ((ph_buf[0] == 0) && ((number = NonAsciiNumber(letter)) > 0)) {
		// convert a non-ascii number to 0-9
		LookupLetter(tr, number, 0, ph_buf, control & 1);
	}

	al_offset = 0;
	al_flags = 0;
	if ((alphabet = AlphabetFromChar(letter)) != NULL) {
		al_offset = alphabet->offset;
		al_flags = alphabet->flags;
	}

	if (alphabet != current_alphabet) {
		// speak the name of the alphabet
		current_alphabet = alphabet;
		if ((alphabet != NULL) && !(al_flags & AL_DONT_NAME) && (al_offset != translator->letter_bits_offset)) {
			if ((al_flags & AL_DONT_NAME) || (al_offset == translator->langopts.alt_alphabet) || (al_offset == translator->langopts.our_alphabet)) {
				// don't say the alphabet name
			} else {
				ph_buf2[0] = 0;
				if (Lookup(translator, alphabet->name, ph_alphabet) == 0) { // the original language for the current voice
					// Can't find the local name for this alphabet, use the English name
					ph_alphabet[2] = SetTranslator3(ESPEAKNG_DEFAULT_VOICE); // overwrites previous contents of translator3
					Lookup(translator3, alphabet->name, ph_buf2);
				} else if (translator != tr) {
					phontab_1 = tr->phoneme_tab_ix;
					strcpy(ph_buf2, ph_alphabet);
					ph_alphabet[2] = translator->phoneme_tab_ix;
				}

				if (ph_buf2[0] != 0) {
					// we used a different language for the alphabet name (now in ph_buf2)
					ph_alphabet[0] = phonPAUSE;
					ph_alphabet[1] = phonSWITCH;
					strcpy(&ph_alphabet[3], ph_buf2);
					len = strlen(ph_buf2) + 3;
					ph_alphabet[len] = phonSWITCH;
					ph_alphabet[len+1] = phontab_1;
					ph_alphabet[len+2] = 0;
				}
			}
		}
	}

	// caution: SetWordStress() etc don't expect phonSWITCH + phoneme table number

	if (ph_buf[0] == 0) {
		int language;
		if ((al_offset != 0) && (al_offset == translator->langopts.alt_alphabet))
			language = translator->langopts.alt_alphabet_lang;
		else if ((alphabet != NULL) && (alphabet->language != 0) && !(al_flags & AL_NOT_LETTERS))
			language = alphabet->language;
		else
			language = L('e', 'n');

		if ((language != tr->translator_name) || (language == L('k', 'o'))) {
			char *p3;
			//int initial, code;
			char hangul_buf[12];

			// speak in the language for this alphabet (or English)
			char word_buf[5];
			ph_buf[2] = SetTranslator3(WordToString2(word_buf, language));

			if (translator3 != NULL) {
				int code;
				if (((code = letter - 0xac00) >= 0) && (letter <= 0xd7af)) {
					// Special case for Korean letters.
					// break a syllable hangul into 2 or 3 individual jamo

					hangul_buf[0] = ' ';
					p3 = &hangul_buf[1];
					int initial;
					if ((initial = (code/28)/21) != 11) {
						p3 += utf8_out(initial + 0x1100, p3);
					}
					utf8_out(((code/28) % 21) + 0x1161, p3); // medial
					utf8_out((code % 28) + 0x11a7, &p3[3]); // final
					p3[6] = ' ';
					p3[7] = 0;
					ph_buf[3] = 0;
					TranslateRules(translator3, &hangul_buf[1], &ph_buf[3], sizeof(ph_buf)-3, NULL, 0, NULL);
					SetWordStress(translator3, &ph_buf[3], NULL, -1, 0);
				} else
					LookupLetter(translator3, letter, word[n_bytes], &ph_buf[3], control & 1);

				if (ph_buf[3] == phonSWITCH) {
					// another level of language change
					ph_buf[2] = SetTranslator3(&ph_buf[4]);
					LookupLetter(translator3, letter, word[n_bytes], &ph_buf[3], control & 1);
				}

				SelectPhonemeTable(voice->phoneme_tab_ix); // revert to original phoneme table

				if (ph_buf[3] != 0) {
					ph_buf[0] = phonPAUSE;
					ph_buf[1] = phonSWITCH;
					len = strlen(&ph_buf[3]) + 3;
					ph_buf[len] = phonSWITCH; // switch back
					ph_buf[len+1] = tr->phoneme_tab_ix;
					ph_buf[len+2] = 0;
				}
			}
		}
	}

	if (ph_buf[0] == 0) {
		// character name not found
		int speak_letter_number = 1;
		if (!(al_flags & AL_NO_SYMBOL)) {
			if (iswalpha(letter))
				Lookup(translator, "_?A", ph_buf);

			if ((ph_buf[0] == 0) && !iswspace(letter))
				Lookup(translator, "_??", ph_buf);

			if (ph_buf[0] == 0)
				EncodePhonemes("l'et@", ph_buf, NULL);
		}

		if (!(control & 4) && (al_flags & AL_NOT_CODE)) {
			// don't speak the character code number, unless we want full details of this character
			speak_letter_number = 0;
		}

		if (speak_letter_number) {
			char *p2;
			if (al_offset == 0x2800) {
				// braille dots symbol, list the numbered dots
				p2 = hexbuf;
				for (int ix = 0; ix < 8; ix++) {
					if (letter & (1 << ix))
						*p2++ = '1'+ix;
				}
				*p2 = 0;
			} else {
				// speak the hexadecimal number of the character code
				sprintf(hexbuf, "%x", letter);
			}

			char *pbuf;
			pbuf = ph_buf;
			for (p2 = hexbuf; *p2 != 0; p2++) {
				pbuf += strlen(pbuf);
				*pbuf++ = phonPAUSE_VSHORT;
				LookupLetter(translator, *p2, 0, pbuf, 1);
				if (((pbuf[0] == 0) || (pbuf[0] == phonSWITCH)) && (*p2 >= 'a')) {
					// This language has no translation for 'a' to 'f', speak English names using base phonemes
					EncodePhonemes(hex_letters[*p2 - 'a'], pbuf, NULL);
				}
			}
			strcat(pbuf, pause_string);
		}
	}

	len = strlen(phonemes);

	if (tr->langopts.accents & 2)  // 'capital' before or after the word ?
		sprintf(ph_buf2, "%c%s%s%s", 0xff, ph_alphabet, ph_buf, capital);
	else
		sprintf(ph_buf2, "%c%s%s%s", 0xff, ph_alphabet, capital, ph_buf); // the 0xff marker will be removed or replaced in SetSpellingStress()
	if ((len + strlen(ph_buf2)) < N_WORD_PHONEMES)
		strcpy(&phonemes[len], ph_buf2);
	return n_bytes;
}

// append plural suffixes depending on preceding letter
static void addPluralSuffixes(int flags, Translator *tr, char last_char, char *word_phonemes)
{
	char word_zz[5] = { 0, ' ', 'z', 'z', 0 };
	char word_iz[5] = { 0, ' ', 'i', 'z', 0 };
	char word_ss[5] = { 0, ' ', 's', 's', 0 };
	if (flags & FLAG_HAS_PLURAL) {
		// s or 's suffix, append [s], [z] or [Iz] depending on previous letter
		if (last_char == 'f')
			TranslateRules(tr, &word_ss[2], word_phonemes, N_WORD_PHONEMES,
			NULL, 0, NULL);
		else if ((last_char == 0) || (strchr_w("hsx", last_char) == NULL))
			TranslateRules(tr, &word_zz[2], word_phonemes, N_WORD_PHONEMES,
			NULL, 0, NULL);
		else
			TranslateRules(tr, &word_iz[2], word_phonemes, N_WORD_PHONEMES,
			NULL, 0, NULL);
	}
}

static int CheckDottedAbbrev(char *word1)
{
	int wc;
	int count = 0;
	int ix;
	char *word;
	char *wbuf;
	char word_buf[80];

	word = word1;
	wbuf = word_buf;

	for (;;) {
		int ok = 0;
		int nbytes = utf8_in(&wc, word);
		if ((word[nbytes] == ' ') && IsAlpha(wc)) {
			if (word[nbytes+1] == '.') {
				if (word[nbytes+2] == ' ')
					ok = 1;
				else if (word[nbytes+2] == '\'' && word[nbytes+3] == 's') {
					nbytes += 2; // delete the final dot (eg. u.s.a.'s)
					ok = 2;
				}
			} else if ((count > 0))
				ok = 2;
		}

		if (ok == 0)
			break;

		for (ix = 0; ix < nbytes; ix++)
			*wbuf++ = word[ix];

		count++;

		if (ok == 2) {
			word += nbytes;
			break;
		}

		word += (nbytes + 3);
	}

	if (count > 1) {
		ix = wbuf - word_buf;
		memcpy(word1, word_buf, ix);
		while (&word1[ix] < word)
			word1[ix++] = ' ';
		dictionary_skipwords = (count - 1)*2;
	}
	return count;
}

static int NonAsciiNumber(int letter)
{
	// Change non-ascii digit into ascii digit '0' to '9', (or -1 if not)
	const int *p;
	int base;

	for (p = number_ranges; (base = *p) != 0; p++) {
		if (letter < base)
			break; // not found
		if (letter < (base+10))
			return letter-base+'0';
	}
	return -1;
}

static int Unpronouncable(Translator *tr, char *word, int posn)
{
	/* Determines whether a word in 'unpronouncable', i.e. whether it should
	    be spoken as individual letters.

	    This function may be language specific. This is a generic version.
	 */

	int c;
	int c1 = 0;
	int vowel_posn = 9;
	int index;
	int count;
	const ALPHABET *alphabet;

	utf8_in(&c, word);
	if ((tr->letter_bits_offset > 0) && (c < 0x241)) {
		// Latin characters for a language with a non-latin alphabet
		return 0;  // so we can re-translate the word as English
	}

	if (((alphabet = AlphabetFromChar(c)) != NULL)  && (alphabet->offset != tr->letter_bits_offset)) {
		// Character is not in our alphabet
		return 0;
	}

	if (tr->langopts.param[LOPT_UNPRONOUNCABLE] == 1)
		return 0;

	if (((c = *word) == ' ') || (c == 0) || (c == '\''))
		return 0;

	index = 0;
	count = 0;
	for (;;) {
		index += utf8_in(&c, &word[index]);
		if ((c == 0) || (c == ' '))
			break;

		if ((c == '\'') && ((count > 1) || (posn > 0)))
			break; // "tv'" but not "l'"

		if (count == 0)
			c1 = c;

		if ((c == '\'') && (tr->langopts.param[LOPT_UNPRONOUNCABLE] == 3)) {
			// don't count apostrophe
		} else
			count++;

		if (IsVowel(tr, c)) {
			vowel_posn = count; // position of the first vowel
			break;
		}

		if ((c != '\'') && !iswalpha(c))
			return 0;
	}

	if ((vowel_posn > 2) && (tr->langopts.param[LOPT_UNPRONOUNCABLE] == 2)) {
		// Lookup unpronounable rules in *_rules
		return Unpronouncable2(tr, word);
	}

	if (c1 == tr->langopts.param[LOPT_UNPRONOUNCABLE])
		vowel_posn--; // disregard this as the initial letter when counting

	if (vowel_posn > (tr->langopts.max_initial_consonants+1))
		return 1; // no vowel, or no vowel in first few letters

	return 0;
}

static int Unpronouncable2(Translator *tr, char *word)
{
	int c;
	int end_flags;
	char ph_buf[N_WORD_PHONEMES];

	ph_buf[0] = 0;
	c = word[-1];
	word[-1] = ' '; // ensure there is a space before the "word"
	end_flags = TranslateRules(tr, word, ph_buf, sizeof(ph_buf), NULL, FLAG_UNPRON_TEST, NULL);
	word[-1] = c;
	if ((end_flags == 0) || (end_flags & SUFX_UNPRON))
		return 1;
	return 0;
}
