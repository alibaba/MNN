/*
 * Copyright (C) 2005 to 2015 by Jonathan Duddington
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
#include <errno.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>
#include <ucd/ucd.h>

#include "readclause.h"
#include "common.h"               // for GetFileLength, strncpy0
#include "dictionary.h"           // for LookupDictList, DecodePhonemes, Set...
#include "error.h"                // for create_file_error_context
#include "phoneme.h"              // for phonSWITCH
#include "soundicon.h"               // for LookupSoundIcon
#include "speech.h"               // for LookupMnem, PATHSEP
#include "ssml.h"                 // for SSML_STACK, ProcessSsmlTag, N_PARAM...
#include "synthdata.h"            // for SelectPhonemeTable
#include "translate.h"            // for Translator, utf8_out, CLAUSE_OPTION...
#include "voice.h"                // for voice, voice_t, espeak_GetCurrentVoice

#define N_XML_BUF   500

static void DecodeWithPhonemeMode(char *buf, char *phonemes, Translator *tr, Translator *tr2, unsigned int flags[]);
static void TerminateBufWithSpaceAndZero(char *buf, int index, int *ungetc);

static const char *xmlbase = ""; // base URL from <speak>

static int namedata_ix = 0;
static int n_namedata = 0;
char *namedata = NULL;

static int ungot_char2 = 0;
espeak_ng_TEXT_DECODER *p_decoder = NULL;
static int ungot_char;

static bool ignore_text = false; // set during <sub> ... </sub>  to ignore text which has been replaced by an alias
static bool audio_text = false; // set during <audio> ... </audio>
static bool clear_skipping_text = false; // next clause should clear the skipping_text flag
int count_characters = 0;
static int sayas_mode;
static int sayas_start;

#define N_SSML_STACK  20
static int n_ssml_stack;
static SSML_STACK ssml_stack[N_SSML_STACK];

static espeak_VOICE base_voice;
static char base_voice_variant_name[40] = { 0 };
static char current_voice_id[40] = { 0 };

static int n_param_stack;
PARAM_STACK param_stack[N_PARAM_STACK];

static int speech_parameters[N_SPEECH_PARAM]; // current values, from param_stack
int saved_parameters[N_SPEECH_PARAM]; // Parameters saved on synthesis start

#define ESPEAKNG_CLAUSE_TYPE_PROPERTY_MASK 0xFFF0000000000000ull

int clause_type_from_codepoint(uint32_t c)
{
	ucd_category cat = ucd_lookup_category(c);
	ucd_property props = ucd_properties(c, cat);

	switch (props & ESPEAKNG_CLAUSE_TYPE_PROPERTY_MASK)
	{
	case ESPEAKNG_PROPERTY_FULL_STOP:
		return CLAUSE_PERIOD;
	case ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER:
		return CLAUSE_PERIOD | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_QUESTION_MARK:
		return CLAUSE_QUESTION;
	case ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER:
		return CLAUSE_QUESTION | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD:
		return CLAUSE_QUESTION | CLAUSE_PUNCTUATION_IN_WORD;
	case ESPEAKNG_PROPERTY_EXCLAMATION_MARK:
		return CLAUSE_EXCLAMATION;
	case ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER:
		return CLAUSE_EXCLAMATION | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD:
		return CLAUSE_EXCLAMATION | CLAUSE_PUNCTUATION_IN_WORD;
	case ESPEAKNG_PROPERTY_COMMA:
		return CLAUSE_COMMA;
	case ESPEAKNG_PROPERTY_COMMA | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER:
		return CLAUSE_COMMA | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_COLON:
		return CLAUSE_COLON;
	case ESPEAKNG_PROPERTY_COLON | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER:
		return CLAUSE_COLON | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_SEMI_COLON:
	case ESPEAKNG_PROPERTY_EXTENDED_DASH:
		return CLAUSE_SEMICOLON;
	case ESPEAKNG_PROPERTY_SEMI_COLON | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER:
	case ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER | ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION:
	case ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER | ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION:
		return CLAUSE_SEMICOLON | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_ELLIPSIS:
		return CLAUSE_SEMICOLON | CLAUSE_SPEAK_PUNCTUATION_NAME | CLAUSE_OPTIONAL_SPACE_AFTER;
	case ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR:
		return CLAUSE_PARAGRAPH;
	}

	return CLAUSE_NONE;
}

static int IsRomanU(unsigned int c)
{
	if ((c == 'I') || (c == 'V') || (c == 'X') || (c == 'L'))
		return 1;
	return 0;
}

int Eof(void)
{
	if (ungot_char != 0)
		return 0;

	return text_decoder_eof(p_decoder);
}

static int GetC(void)
{
	int c1;

	if ((c1 = ungot_char) != 0) {
		ungot_char = 0;
		return c1;
	}

	count_characters++;
	return text_decoder_getc(p_decoder);
}

static void UngetC(int c)
{
	ungot_char = c;
}

const char *WordToString2(char buf[5], unsigned int word)
{
	// Convert a language mnemonic word into a string
	int ix;
	char *p;

	p = buf;
	for (ix = 3; ix >= 0; ix--) {
		if ((*p = word >> (ix*8)) != 0)
			p++;
	}
	*p = 0;
	return buf;
}

static const char *LookupSpecial(Translator *tr, const char *string, char *text_out)
{
	unsigned int flags[2];
	char phonemes[55];
	char *string1 = (char *)string;

	flags[0] = flags[1] = 0;
	if (LookupDictList(tr, &string1, phonemes, flags, 0, NULL)) {
		DecodeWithPhonemeMode(text_out, phonemes, tr, NULL, flags);
		return text_out;
	}
	return NULL;
}

static const char *LookupCharName(char buf[60], Translator *tr, int c, bool only)
{
	// Find the phoneme string (in ascii) to speak the name of character c
	// Used for punctuation characters and symbols

	int ix;
	unsigned int flags[2];
	char single_letter[24];
	char phonemes[60];
	const char *lang_name = NULL;
	char *string;

	buf[0] = 0;
	flags[0] = 0;
	flags[1] = 0;
	single_letter[0] = 0;
	single_letter[1] = '_';
	ix = utf8_out(c, &single_letter[2]);
	single_letter[2+ix] = 0;

	if (only == true) {
		string = &single_letter[2];
		LookupDictList(tr, &string, phonemes, flags, 0, NULL);
	}

	if (only == false) {
		string = &single_letter[1];
		if (LookupDictList(tr, &string, phonemes, flags, 0, NULL) == 0) {
			// try _* then *
			string = &single_letter[2];
			if (LookupDictList(tr, &string, phonemes, flags, 0, NULL) == 0) {
				// now try the rules
				single_letter[1] = ' ';
				TranslateRules(tr, &single_letter[2], phonemes, sizeof(phonemes), NULL, 0, NULL);
			}
		}
		
		if (((phonemes[0] == 0) || (phonemes[0] == phonSWITCH)) && (tr->translator_name != L('e', 'n'))) {
    		// not found, try English
    		SetTranslator2(ESPEAKNG_DEFAULT_VOICE);
    		string = &single_letter[1];
    		single_letter[1] = '_';
    		if (LookupDictList(translator2, &string, phonemes, flags, 0, NULL) == 0) {
    			string = &single_letter[2];
    			LookupDictList(translator2, &string, phonemes, flags, 0, NULL);
    		}
    		if (phonemes[0])
    			lang_name = ESPEAKNG_DEFAULT_VOICE;
    		else
    			SelectPhonemeTable(voice->phoneme_tab_ix); // revert to original phoneme table
    	}
	}

	if (phonemes[0]) {
		if (lang_name) {
			DecodeWithPhonemeMode(buf, phonemes, tr, translator2, flags);
			SelectPhonemeTable(voice->phoneme_tab_ix); // revert to original phoneme table
		} else {
			DecodeWithPhonemeMode(buf, phonemes, tr, NULL, flags);
		}
	} else if (only == false)
		strcpy(buf, "[\002(X1)(X1)(X1)]]");

	return buf;
}

static int AnnouncePunctuation(Translator *tr, int c1, int *c2_ptr, char *output, int *bufix, int end_clause)
{
	// announce punctuation names
	// c1:  the punctuation character
	// c2:  the following character


	const char *punctname = NULL;
	int soundicon;
	int attributes;
	int short_pause;
	int c2;
	int len;
	int bufix1;
	char buf[200];
	char ph_buf[30];
	char cn_buf[60];

	c2 = *c2_ptr;
	buf[0] = 0;

	if ((soundicon = LookupSoundicon(c1)) >= 0) {
		// add an embedded command to play the soundicon
		sprintf(buf, "\001%dI ", soundicon);
		UngetC(c2);
	} else {
		if ((c1 == '.') && (end_clause) && (c2 != '.')) {
			if (LookupSpecial(tr, "_.p", ph_buf))
				punctname = ph_buf; // use word for 'period' instead of 'dot'
		}
		if (punctname == NULL)
			punctname = LookupCharName(cn_buf, tr, c1, false);

		if (punctname == NULL)
			return -1;


		if ((*bufix == 0) || (end_clause == 0) || (tr->langopts.param[LOPT_ANNOUNCE_PUNCT] & 2)) {
			int punct_count = 1;
			while (!Eof() && (c2 == c1) && (c1 != '<')) { // don't eat extra '<', it can miss XML tags
				punct_count++;
				c2 = GetC();
			}
			*c2_ptr = c2;
			if (end_clause)
				UngetC(c2);

			if (punct_count == 1)
				sprintf(buf, " %s", punctname); // we need the space before punctname, to ensure it doesn't merge with the previous word  (eg.  "2.-a")
			else if (punct_count < 4) {
				buf[0] = 0;
				if (embedded_value[EMBED_S] < 300)
					sprintf(buf, "\001+10S"); // Speak punctuation name faster, unless we are already speaking fast.  It would upset Sonic SpeedUp

				char buf2[80];
				while (punct_count-- > 0) {
					sprintf(buf2, " %s", punctname);
					strcat(buf, buf2);
				}

				if (embedded_value[EMBED_S] < 300) {
					sprintf(buf2, " \001-10S");
					strcat(buf, buf2);
				}
			} else
				sprintf(buf, " %s %d %s",
				        punctname, punct_count, punctname);
		} else {
			// end the clause now and pick up the punctuation next time
			ungot_char2 = c1;
			TerminateBufWithSpaceAndZero(buf, 0, &c2);
		}
	}

	bufix1 = *bufix;
	len = strlen(buf);
	strcpy(&output[*bufix], buf);
	*bufix += len;

	if (end_clause == 0)
		return -1;

	if (c1 == '-')
		return CLAUSE_NONE; // no pause

	attributes = clause_type_from_codepoint(c1);

	short_pause = CLAUSE_SHORTFALL;
	if ((attributes & CLAUSE_INTONATION_TYPE) == 0x1000)
		short_pause = CLAUSE_SHORTCOMMA;

	if ((bufix1 > 0) && !(tr->langopts.param[LOPT_ANNOUNCE_PUNCT] & 2)) {
		if ((attributes & ~CLAUSE_OPTIONAL_SPACE_AFTER) == CLAUSE_SEMICOLON)
			return CLAUSE_SHORTFALL;
		return short_pause;
	}

	if (attributes & CLAUSE_TYPE_SENTENCE)
		return attributes;

	return short_pause;
}

int AddNameData(const char *name, int wide)
{
	// Add the name to the namedata and return its position
	// (Used by the Windows SAPI wrapper)

	int ix;
	int len;

	if (wide) {
		len = (wcslen((const wchar_t *)name)+1)*sizeof(wchar_t);
		n_namedata = (n_namedata + sizeof(wchar_t) - 1) % sizeof(wchar_t);  // round to wchar_t boundary
	} else
		len = strlen(name)+1;

	if (namedata_ix+len >= n_namedata) {
		// allocate more space for marker names
		void *vp;
		if ((vp = realloc(namedata, namedata_ix+len + 1000)) == NULL)
			return -1;  // failed to allocate, original data is unchanged but ignore this new name
		// !!! Bug?? If the allocated data shifts position, then pointers given to user application will be invalid

		namedata = (char *)vp;
		n_namedata = namedata_ix+len + 1000;
	}
	memcpy(&namedata[ix = namedata_ix], name, len);
	namedata_ix += len;
	return ix;
}

void SetVoiceStack(espeak_VOICE *v, const char *variant_name)
{
	SSML_STACK *sp;
	sp = &ssml_stack[0];

	if (v == NULL) {
		memset(sp, 0, sizeof(ssml_stack[0]));
		return;
	}
	if (v->languages != NULL)
		strcpy(sp->language, v->languages);
	if (v->name != NULL)
		strncpy0(sp->voice_name, v->name, sizeof(sp->voice_name));
	sp->voice_variant_number = v->variant;
	sp->voice_age = v->age;
	sp->voice_gender = v->gender;

	if (variant_name[0] == '!' && variant_name[1] == 'v' && variant_name[2] == PATHSEP)
		variant_name += 3; // strip variant directory name, !v plus PATHSEP
	strncpy0(base_voice_variant_name, variant_name, sizeof(base_voice_variant_name));
	memcpy(&base_voice, espeak_GetCurrentVoice(), sizeof(base_voice));
}

static void RemoveChar(char *p)
{
	// Replace a UTF-8 character by spaces
	int c;

	memset(p, ' ', utf8_in(&c, p));
}

static int lookupwchar2(const unsigned short *list, int c)
{
	// Replace character c by another character.
	// Returns 0 = not found, 1 = delete character

	int ix;

	for (ix = 0; list[ix] != 0; ix += 2) {
		if (list[ix] == c)
			return list[ix+1];
	}
	return 0;
}

static bool IgnoreOrReplaceChar(Translator *tr, int *c1) {
    int i;
    if ((i = lookupwchar2(tr->chars_ignore, *c1)) != 0) {
        if (i == 1) {
            // ignore this character
            return true;
        }
        *c1 = i; // replace current character with the result
    }
    return false;
}

static int CheckPhonemeMode(int option_phoneme_input, int phoneme_mode, int c1, int c2) {
		if (option_phoneme_input) {
			if (phoneme_mode > 0)
				phoneme_mode--;
			else if ((c1 == '[') && (c2 == '['))
				phoneme_mode = -1; // input is phoneme mnemonics, so don't look for punctuation
			else if ((c1 == ']') && (c2 == ']'))
				phoneme_mode = 2; // set phoneme_mode to zero after the next two characters

		}
    return phoneme_mode;
}

int ReadClause(Translator *tr, char *buf, short *charix, int *charix_top, int n_buf, int *tone_type, char *voice_change)
{
	/* Find the end of the current clause.
	    Write the clause into  buf

	    returns: clause type (bits 0-7: pause x10mS, bits 8-11 intonation type)

	    Also checks for blank line (paragraph) as end-of-clause indicator.

	    Does not end clause for:
	        punctuation immediately followed by alphanumeric  eg.  1.23  !Speak  :path
	        repeated punctuation, eg.   ...   !!!
	 */

	int c1 = ' '; // current character
	int c2; // next character
	int cprev = ' '; // previous character
	int c_next = 0;
	int parag;
	int ix = 0;
	int j;
	int nl_count;
	int linelength = 0;
	int phoneme_mode = 0;
	int n_xml_buf;
	int terminator;
	bool any_alnum = false;
	int punct_data = 0;
	bool is_end_clause;
	int announced_punctuation = 0;
	bool stressed_word = false;
	int end_clause_after_tag = 0;
	int end_clause_index = 0;
	wchar_t xml_buf[N_XML_BUF+1];

	#define N_XML_BUF2 20
	char xml_buf2[N_XML_BUF2+2]; // for &<name> and &<number> sequences
	static char ungot_string[N_XML_BUF2+4];
	static int ungot_string_ix = -1;

	if (clear_skipping_text) {
		skipping_text = false;
		clear_skipping_text = false;
	}

	tr->phonemes_repeat_count = 0;
	tr->clause_upper_count = 0;
	tr->clause_lower_count = 0;
	*tone_type = 0;
	*voice_change = 0;

	if (ungot_char2 != 0) {
		c2 = ungot_char2;
	} else if (Eof()) {
		c2 = 0;
	} else {
		c2 = GetC();
	}

	while (!Eof() || (ungot_char != 0) || (ungot_char2 != 0) || (ungot_string_ix >= 0)) {
		if (!iswalnum(c1)) {
			if ((end_character_position > 0) && (count_characters > end_character_position)) {
				return CLAUSE_EOF;
			}

			if ((skip_characters > 0) && (count_characters >= skip_characters)) {
				// reached the specified start position
				// don't break a word
				clear_skipping_text = true;
				skip_characters = 0;
				UngetC(c2);
				return CLAUSE_NONE;
			}
		}
		int cprev2 = cprev;
		cprev = c1;
		c1 = c2;

		if (ungot_string_ix >= 0) {
			if (ungot_string[ungot_string_ix] == 0) {
				MAKE_MEM_UNDEFINED(&ungot_string, sizeof(ungot_string));
				ungot_string_ix = -1;
			}
		}

		if ((ungot_string_ix == 0) && (ungot_char2 == 0))
			c1 = ungot_string[ungot_string_ix++];
		if (ungot_string_ix >= 0) {
			c2 = ungot_string[ungot_string_ix++];
		} else if (Eof()) {
			c2 = ' ';
		} else {
			c2 = GetC();
		}

		ungot_char2 = 0;

		if ((option_ssml) && (phoneme_mode == 0)) {
			if ((c1 == '&') && ((c2 == '#') || ((c2 >= 'a') && (c2 <= 'z')))) {
				n_xml_buf = 0;
				c1 = c2;
				while (!Eof() && (iswalnum(c1) || (c1 == '#')) && (n_xml_buf < N_XML_BUF2)) {
					xml_buf2[n_xml_buf++] = c1;
					c1 = GetC();
				}
				xml_buf2[n_xml_buf] = 0;
				if (Eof()) {
					c2 = '\0';
				} else {
					c2 = GetC();
				}
				sprintf(ungot_string, "%s%c%c", &xml_buf2[0], c1, c2);

				int found = -1;
				if (c1 == ';') {
					found = ParseSsmlReference(xml_buf2, &c1, &c2);
				}

				if (found <= 0) {
					ungot_string_ix = 0;
					c1 = '&';
					c2 = ' ';
				}

				if ((c1 <= 0x20) && ((sayas_mode == SAYAS_SINGLE_CHARS) || (sayas_mode == SAYAS_KEY)))
					c1 += 0xe000; // move into unicode private usage area
			} else if (c1 == '<') {
				if ((c2 == '/') || iswalpha(c2) || c2 == '!' || c2 == '?') {
					// check for space in the output buffer for embedded commands produced by the SSML tag
					if (ix > (n_buf - 20)) {
						// Perhaps not enough room, end the clause before the SSML tag
						ungot_char2 = c1;
						TerminateBufWithSpaceAndZero(buf, ix, &c2);
						return CLAUSE_NONE;
					}

					// SSML Tag
					n_xml_buf = 0;
					c1 = c2;
					while (!Eof() && (c1 != '>') && (n_xml_buf < N_XML_BUF)) {
						xml_buf[n_xml_buf++] = c1;
						c1 = GetC();
					}
					xml_buf[n_xml_buf] = 0;
					c2 = ' ';

					if (base_voice.identifier)
						strcpy(current_voice_id, base_voice.identifier);
					terminator = ProcessSsmlTag(xml_buf, buf, &ix, n_buf, xmlbase, &audio_text, current_voice_id, &base_voice, base_voice_variant_name, &ignore_text, &clear_skipping_text, &sayas_mode, &sayas_start, ssml_stack, &n_ssml_stack, &n_param_stack, (int *)speech_parameters);

					if (terminator != 0) {
						TerminateBufWithSpaceAndZero(buf, ix, NULL);

						if (terminator & CLAUSE_TYPE_VOICE_CHANGE)
							strcpy(voice_change, current_voice_id);
						return terminator;
					}
					c1 = ' ';
					if (!Eof()) {
						c2 = GetC();
					}
					continue;
				}
			}
		}

		if (ignore_text)
			continue;

		if ((c2 == '\n') && (option_linelength == -1)) {
			// single-line mode, return immediately on NL
			if ((terminator = clause_type_from_codepoint(c1)) == CLAUSE_NONE) {
				charix[ix] = count_characters - clause_start_char;
				*charix_top = ix;
				ix += utf8_out(c1, &buf[ix]);
				terminator = CLAUSE_PERIOD; // line doesn't end in punctuation, assume period
			}
			TerminateBufWithSpaceAndZero(buf, ix, NULL);
			return terminator;
		}

		if (c1 == CTRL_EMBEDDED) {
 			// an embedded command. If it's a voice change, end the clause
			if (c2 == 'V') {
				buf[ix++] = 0; // end the clause at this point
				while (!Eof() && !iswspace(c1 = GetC()) && (ix < (n_buf-1)))
					buf[ix++] = c1; // add voice name to end of buffer, after the text
				buf[ix++] = 0;
				return CLAUSE_VOICE;
			} else if (c2 == 'B') {
				// set the punctuation option from an embedded command
				//  B0     B1     B<punct list><space>
				strcpy(&buf[ix], "   ");
				ix += 3;

				if (!Eof() && (c2 = GetC()) == '0')
					option_punctuation = 0;
				else {
					option_punctuation = 1;
					option_punctlist[0] = 0;
					if (c2 != '1') {
						// a list of punctuation characters to be spoken, terminated by space
						j = 0;
						while (!Eof() && !iswspace(c2) && (j < N_PUNCTLIST-1)) {
							option_punctlist[j++] = c2;
							c2 = GetC();
							buf[ix++] = ' ';
						}
						option_punctlist[j] = 0; // terminate punctuation list
						option_punctuation = 2;
					}
				}
				if (!Eof())
					c2 = GetC();
				continue;
			}
		}

		linelength++;

		 if (IgnoreOrReplaceChar(tr, &c1) == true)
		    continue;


		if (iswalnum(c1))
			any_alnum = true;
		else {
			if (stressed_word) {
				stressed_word = false;
				c1 = CHAR_EMPHASIS; // indicate this word is stressed
				UngetC(c2);
				c2 = ' ';
			}

			if (c1 == 0xf0b)
				c1 = ' '; // Tibet inter-syllabic mark, ?? replace by space ??

			if (c1 == 0xd4d) {
				// Malayalam virama, check if next character is Zero-width-joiner
				if (c2 == 0x200d)
					c1 = 0xd4e; // use this unofficial code for chillu-virama
			}
		}

		if (iswupper(c1)) {
			tr->clause_upper_count++;

			if ((option_capitals == 2) && (sayas_mode == 0) && !iswupper(cprev)) {
				char text_buf[30];
				if (LookupSpecial(tr, "_cap", text_buf) != NULL) {
					j = strlen(text_buf);
					if ((ix + j) < n_buf) {
						strcpy(&buf[ix], text_buf);
						ix += j;
					}
				}
			}
		} else if (iswalpha(c1))
			tr->clause_lower_count++;

		phoneme_mode = CheckPhonemeMode(option_phoneme_input, phoneme_mode, c1, c2);

		if (c1 == '\n') {
			parag = 0;

			// count consecutive newlines, ignoring other spaces
			while (!Eof() && iswspace(c2)) {
				if (c2 == '\n')
					parag++;
				c2 = GetC();
			}
			if (parag > 0) {
				// 2nd newline, assume paragraph
				if (end_clause_after_tag)
					RemoveChar(&buf[end_clause_index]); // delete clause-end punctiation
				TerminateBufWithSpaceAndZero(buf, ix, &c2);
				if (parag > 3)
					parag = 3;
				if (option_ssml) parag = 1;
				return (CLAUSE_PARAGRAPH-30) + 30*parag; // several blank lines, longer pause
			}

			if (linelength <= option_linelength) {
				// treat lines shorter than a specified length as end-of-clause
				TerminateBufWithSpaceAndZero(buf, ix, &c2);
				return CLAUSE_COLON;
			}

			linelength = 0;
		}

		announced_punctuation = 0;

		if ((phoneme_mode == 0) && (sayas_mode == 0)) {
			is_end_clause = false;

			if (end_clause_after_tag) {
				// Because of an xml tag, we are waiting for the
				// next non-blank character to decide whether to end the clause
				// i.e. is dot followed by an upper-case letter?

				if (!iswspace(c1)) {
					if (!IsAlpha(c1) || !iswlower(c1)) {
						ungot_char2 = c1;
						TerminateBufWithSpaceAndZero(buf, end_clause_index, &c2);
						return end_clause_after_tag;
					}
					end_clause_after_tag = 0;
				}
			}

			if ((c1 == '.') && (c2 == '.')) {
				while (!Eof() && (c_next = GetC()) == '.') {
					// 3 or more dots, replace by elipsis
					c1 = 0x2026;
					c2 = ' ';
				}
				if (c1 == 0x2026)
					c2 = c_next;
				else
					UngetC(c_next);
			}

			punct_data = 0;
			if ((punct_data = clause_type_from_codepoint(c1)) != CLAUSE_NONE) {

				// Handling of sequences of ? and ! like ??!?, !!??!, ?!! etc
				// Use only first char as determinant
				if(punct_data & (CLAUSE_QUESTION | CLAUSE_EXCLAMATION)) {
					while(!Eof() && clause_type_from_codepoint(c2) & (CLAUSE_QUESTION | CLAUSE_EXCLAMATION)) {
						c_next = GetC();
						c2 = c_next;
					}
				}

				if (punct_data & CLAUSE_PUNCTUATION_IN_WORD) {
					// Armenian punctuation inside a word
					stressed_word = true;
					*tone_type = punct_data >> 12 & 0xf; // override the end-of-sentence type
					continue;
				}

				if (iswspace(c2) || (punct_data & CLAUSE_OPTIONAL_SPACE_AFTER) || IsBracket(c2) || (c2 == '?') || Eof() || c2 == CTRL_EMBEDDED) { // don't check for '-' because it prevents recognizing ':-)'
					// note: (c2='?') is for when a smart-quote has been replaced by '?'
					is_end_clause = true;
				}
			}

			// don't announce punctuation for the alternative text inside inside <audio> ... </audio>
			if (c1 == 0xe000+'<')  c1 = '<';
			if (option_punctuation && iswpunct(c1) && (audio_text == false)) {
				// option is set to explicitly speak punctuation characters
				// if a list of allowed punctuation has been set up, check whether the character is in it
				if ((option_punctuation == 1) || (wcschr(option_punctlist, c1) != NULL)) {
					tr->phonemes_repeat_count = 0;
					if ((terminator = AnnouncePunctuation(tr, c1, &c2, buf, &ix, is_end_clause)) >= 0)
						return terminator;
					announced_punctuation = c1;
				}
			}

			if ((punct_data & CLAUSE_SPEAK_PUNCTUATION_NAME) && (announced_punctuation == 0)) {
				// used for elipsis (and 3 dots) if a pronunciation for elipsis is given in *_list
				char *p2;

				p2 = &buf[ix];
				char cn_buf[60];
				sprintf(p2, "%s", LookupCharName(cn_buf, tr, c1, true));
				if (p2[0] != 0) {
					ix += strlen(p2);
					announced_punctuation = c1;
					punct_data = punct_data & ~CLAUSE_INTONATION_TYPE; // change intonation type to 0 (full-stop)
				}
			}

			if (is_end_clause) {
				nl_count = 0;
				c_next = c2;

				if (iswspace(c_next)) {
					while (!Eof() && iswspace(c_next)) {
						if (c_next == '\n')
							nl_count++;
						c_next = GetC(); // skip past space(s)
					}
				}

				if ((c1 == '.') && (nl_count < 2))
					punct_data |= CLAUSE_DOT_AFTER_LAST_WORD;

				if (nl_count == 0) {
					if ((c1 == ',') && (cprev == '.') && (tr->translator_name == L('h', 'u')) && iswdigit(cprev2) && (iswdigit(c_next) || (iswlower(c_next)))) {
						// lang=hu, fix for ordinal numbers, eg:  "december 2., szerda", ignore ',' after ordinal number
						c1 = CHAR_COMMA_BREAK;
						is_end_clause = false;
					}

					if (c1 == '.' && c_next == '\'' && text_decoder_peekc(p_decoder) == 's') {
					 	// A special case to handle english acronym + genitive, eg. u.s.a.'s
						// But avoid breaking clause handling if anything else follows the apostrophe.
						is_end_clause = false;
					}

					if (c1 == '.') {
						if ((tr->langopts.numbers & NUM_ORDINAL_DOT) &&
						    (iswdigit(cprev) || (IsRomanU(cprev) && (IsRomanU(cprev2) || iswspace(cprev2))))) { // lang=hu
							// dot after a number indicates an ordinal number
							if (!iswdigit(cprev))
								is_end_clause = false; // Roman number followed by dot
							else if (iswlower(c_next) || (c_next == '-')) // hyphen is needed for lang-hu (eg. 2.-kal)
								is_end_clause = false; // only if followed by lower-case, (or if there is a XML tag)
						} 
						if (iswlower(c_next) && tr->langopts.lowercase_sentence == false) {
							// next word has no capital letter, this dot is probably from an abbreviation
							is_end_clause = false;
						}
						if (any_alnum == false) {
							// no letters or digits yet, so probably not a sentence terminator
							// Here, dot is followed by space or bracket
							c1 = ' ';
							is_end_clause = false;
						}
					} else {
						if (any_alnum == false) {
							// no letters or digits yet, so probably not a sentence terminator
							is_end_clause = false;
						}
					}

					if (is_end_clause && (c1 == '.') && (c_next == '<') && option_ssml) {
						// wait until after the end of the xml tag, then look for upper-case letter
						is_end_clause = false;
						end_clause_index = ix;
						end_clause_after_tag = punct_data;
					}
				}

				if (is_end_clause) {
					TerminateBufWithSpaceAndZero(buf, ix, &c_next);

					if (iswdigit(cprev) && !IsAlpha(c_next)) // ????
						punct_data &= ~CLAUSE_DOT_AFTER_LAST_WORD;
					if (nl_count > 1) {
						if ((punct_data == CLAUSE_QUESTION) || (punct_data == CLAUSE_EXCLAMATION))
							return punct_data + 35; // with a longer pause
						return CLAUSE_PARAGRAPH;
					}
					return punct_data; // only recognise punctuation if followed by a blank or bracket/quote
				} else if (!Eof()) {
					if (iswspace(c2))
						UngetC(c_next);
				}
			}
		}

		if (speech_parameters[espeakSILENCE] == 1)
			continue;

		if (c1 == announced_punctuation) {
			// This character has already been announced, so delete it so that it isn't spoken a second time.
			// Unless it's a hyphen or apostrophe (which is used by TranslateClause() )
			if (IsBracket(c1))
				c1 = 0xe000 + '('; // Unicode private useage area.  So TranslateRules() knows the bracket name has been spoken
			else if (c1 != '-')
				c1 = ' ';
		}

		j = ix+1;

		if (c1 == 0xe000 + '<') c1 = '<';

		ix += utf8_out(c1, &buf[ix]);
		if (!iswspace(c1) && !IsBracket(c1)) {
			charix[ix] = count_characters - clause_start_char;
			while (j < ix)
				charix[j++] = -1; // subsequent bytes of a multibyte character
		}
		*charix_top = ix;

		if (((ix > (n_buf-75)) && !IsAlpha(c1) && !iswdigit(c1))  ||  (ix >= (n_buf-4))) {
			// clause too long, getting near end of buffer, so break here
			// try to break at a word boundary (unless we actually reach the end of buffer).
			// (n_buf-4) is to allow for 3 bytes of multibyte character plus terminator.
			TerminateBufWithSpaceAndZero(buf, ix, &c2);
			return CLAUSE_NONE;
		}
	}

	if (stressed_word)
		ix += utf8_out(CHAR_EMPHASIS, &buf[ix]);
	if (end_clause_after_tag)
		RemoveChar(&buf[end_clause_index]); // delete clause-end punctiation
	TerminateBufWithSpaceAndZero(buf, ix, NULL);
	return CLAUSE_EOF; // end of file
}

void InitNamedata(void)
{
	namedata_ix = 0;
	if (namedata != NULL) {
		free(namedata);
		namedata = NULL;
		n_namedata = 0;
	}
}

void InitText2(void)
{
	int param;

	ungot_char = 0;
	ungot_char2 = 0;

	n_ssml_stack = 1;
	MAKE_MEM_UNDEFINED(&ssml_stack[1], sizeof(ssml_stack) - sizeof(ssml_stack[0]));
	n_param_stack = 1;
	MAKE_MEM_UNDEFINED(&param_stack[1], sizeof(param_stack) - sizeof(param_stack[0]));
	ssml_stack[0].tag_type = 0;

	for (param = 0; param < N_SPEECH_PARAM; param++)
		speech_parameters[param] = param_stack[0].parameter[param]; // set all speech parameters to defaults

	option_punctuation = speech_parameters[espeakPUNCTUATION];
	option_capitals = speech_parameters[espeakCAPITALS];

	current_voice_id[0] = 0;

	ignore_text = false;
	audio_text = false;
	clear_skipping_text = false;
	count_characters = -1;
	sayas_mode = 0;

	xmlbase = NULL;
}

static void TerminateBufWithSpaceAndZero(char *buf, int index, int *ungetc) {
	buf[index] = ' ';
	buf[index+1] = 0;

	if (ungetc != NULL) {
		UngetC(*ungetc);
	}
}

static void DecodeWithPhonemeMode(char *buf, char *phonemes, Translator *tr, Translator *tr2, unsigned int flags[]) {
	char phonemes2[55];
	if (tr2 == NULL) {
		SetWordStress(tr, phonemes, flags, -1, 0);
		DecodePhonemes(phonemes, phonemes2);
		sprintf(buf, "[\002%s]]", phonemes2);
	} else {
		SetWordStress(tr2, phonemes, flags, -1, 0);
	    DecodePhonemes(phonemes, phonemes2);
			char wbuf[5];
	    sprintf(buf, "[\002_^_%s %s _^_%s]]", ESPEAKNG_DEFAULT_VOICE, phonemes2, WordToString2(wbuf, tr->translator_name));
    }
}
