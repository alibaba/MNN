/*
 * Copyright (C) 2005 to 2015 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2017 Reece H. Dunn
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

#include "ssml.h"
#include "common.h"           // for strncpy0
#include "mnemonics.h"               // for LookupMnemName, MNEM_TAB, 
#include "readclause.h"           // for PARAM_STACK, param_stack, AddNameData
#include "soundicon.h"               // for LoadSoundFile2
#include "synthesize.h"           // for SPEED_FACTORS, speed
#include "translate.h"            // for CTRL_EMBEDDED
#include "voice.h"                // for SelectVoice, SelectVoiceByName
#include "speech.h"               // for MAKE_MEM_UNDEFINED

static const MNEM_TAB ssmltags[] = {
	{ "speak",     SSML_SPEAK },
	{ "voice",     SSML_VOICE },
	{ "prosody",   SSML_PROSODY },
	{ "say-as",    SSML_SAYAS },
	{ "mark",      SSML_MARK },
	{ "s",         SSML_SENTENCE },
	{ "p",         SSML_PARAGRAPH },
	{ "phoneme",   SSML_PHONEME },
	{ "sub",       SSML_SUB },
	{ "tts:style", SSML_STYLE },
	{ "audio",     SSML_AUDIO },
	{ "emphasis",  SSML_EMPHASIS },
	{ "break",     SSML_BREAK },
	{ "metadata",  SSML_IGNORE_TEXT },

	{ "br",     HTML_BREAK },
	{ "li",     HTML_BREAK },
	{ "dd",     HTML_BREAK },
	{ "img",    HTML_BREAK },
	{ "td",     HTML_BREAK },
	{ "h1",     SSML_PARAGRAPH },
	{ "h2",     SSML_PARAGRAPH },
	{ "h3",     SSML_PARAGRAPH },
	{ "h4",     SSML_PARAGRAPH },
	{ "hr",     SSML_PARAGRAPH },
	{ "script", SSML_IGNORE_TEXT },
	{ "style",  SSML_IGNORE_TEXT },
	{ "font",   HTML_NOSPACE },
	{ "b",      HTML_NOSPACE },
	{ "i",      HTML_NOSPACE },
	{ "strong", HTML_NOSPACE },
	{ "em",     HTML_NOSPACE },
	{ "code",   HTML_NOSPACE },

	{ NULL, 0 }
};

static int (*uri_callback)(int, const char *, const char *) = NULL;

static int attrcmp(const wchar_t *string1, const char *string2)
{
	int ix;

	if (string1 == NULL)
		return 1;

	for (ix = 0; (string1[ix] == string2[ix]) && (string1[ix] != 0); ix++)
		;
	if (((string1[ix] == '"') || (string1[ix] == '\'')) && (string2[ix] == 0))
		return 0;
	return 1;
}


static int attrlookup(const wchar_t *string1, const MNEM_TAB *mtab)
{
	int ix;

	for (ix = 0; mtab[ix].mnem != NULL; ix++) {
		if (attrcmp(string1, mtab[ix].mnem) == 0)
			return mtab[ix].value;
	}
	return mtab[ix].value;
}

static int attrnumber(const wchar_t *pw, int default_value, int type)
{
	int value = 0;

	if ((pw == NULL) || !IsDigit09(*pw))
		return default_value;

	while (IsDigit09(*pw))
		value = value*10 + *pw++ - '0';
	if ((type == 1) && (ucd_tolower(*pw) == 's')) {
		// time: seconds rather than ms
		value *= 1000;
	}
	return value;
}

static int attrcopy_utf8(char *buf, const wchar_t *pw, int len)
{
	// Convert attribute string into utf8, write to buf, and return its utf8 length
	int ix = 0;

	if (pw != NULL) {
		wchar_t quote = pw[-1];
		if ((quote != '"') && (quote != '\'')) quote = 0;

		unsigned int c;
		int prev_c = 0;
		while ((ix < (len-4)) && ((c = *pw++) != 0)) {
			if ((quote == 0) && (isspace(c) || (c == '/')))
				break;
			if ((quote != 0) && (c == quote) && (prev_c != '\\'))
				break; // " indicates end of attribute, unless preceded by backstroke

			int n = utf8_out(c, &buf[ix]);
			ix += n;
			prev_c = c;
		}
	}
	buf[ix] = 0;
	return ix;
}

static int attr_prosody_value(int param_type, const wchar_t *pw, int *value_out)
{
	int sign = 0;
	wchar_t *tail;
	double value;

	while (iswspace(*pw)) pw++;
	if (*pw == '+') {
		pw++;
		sign = 1;
	}
	if (*pw == '-') {
		pw++;
		sign = -1;
	}
	value = (double)wcstod(pw, &tail);
	if (tail == pw) {
		// failed to find a number, return 100%
		*value_out = 100;
		return 2;
	}

	if (*tail == '%') {
		if (sign != 0)
			value = 100 + (sign * value);
		*value_out = (int)value;
		return 2; // percentage
	}

	if ((tail[0] == 's') && (tail[1] == 't')) {
		double x;
		// convert from semitones to a  frequency percentage
		x = pow((double)2.0, (double)((value*sign)/12)) * 100;
		*value_out = (int)x;
		return 2; // percentage
	}

	if (param_type == espeakRATE) {
		if (sign == 0)
			*value_out = (int)(value * 100);
		else
			*value_out = 100 + (int)(sign * value * 100);
		return 2; // percentage
	}

	*value_out = (int)value;
	return sign;   // -1, 0, or 1
}

static const char *VoiceFromStack(SSML_STACK *ssml_stack, int n_ssml_stack, espeak_VOICE *base_voice, char base_voice_variant_name[40])
{
	// Use the voice properties from the SSML stack to choose a voice, and switch
	// to that voice if it's not the current voice

	int ix;
	const char *p;
	SSML_STACK *sp;
	const char *v_id;
	int voice_found;
	espeak_VOICE voice_select;
	static char voice_name[40];
	static char identifier[40];
	char language[40];

	MAKE_MEM_UNDEFINED(&voice_name, sizeof(voice_name));

	strcpy(voice_name, ssml_stack[0].voice_name);
	strcpy(language, ssml_stack[0].language);
	voice_select.age = ssml_stack[0].voice_age;
	voice_select.gender = ssml_stack[0].voice_gender;
	voice_select.variant = ssml_stack[0].voice_variant_number;
	voice_select.identifier = NULL;

	for (ix = 0; ix < n_ssml_stack; ix++) {
		espeak_VOICE *v;
		sp = &ssml_stack[ix];
		int voice_name_specified = 0;

		if ((sp->voice_name[0] != 0) && ((v = SelectVoiceByName(NULL, sp->voice_name)) != NULL)) {
			voice_name_specified = 1;
			strcpy(voice_name, sp->voice_name);
			strcpy(identifier, v->identifier);
			language[0] = 0;
			voice_select.gender = ENGENDER_UNKNOWN;
			voice_select.age = 0;
			voice_select.variant = 0;
		}
		if (sp->language[0] != 0) {
			strcpy(language, sp->language);

			// is this language provided by the base voice?
			p = base_voice->languages;
			while (*p++ != 0) {
				if (strcmp(p, language) == 0) {
					// yes, change the language to the main language of the base voice
					strcpy(language, &base_voice->languages[1]);
					break;
				}
				p += (strlen(p) + 1);
			}

			if (voice_name_specified == 0)
			{
				voice_name[0] = 0; // forget a previous voice name if a language is specified
				identifier[0] = 0;
			}
		}
		if (sp->voice_gender != ENGENDER_UNKNOWN)
			voice_select.gender = sp->voice_gender;

		if (sp->voice_age != 0)
			voice_select.age = sp->voice_age;
		if (sp->voice_variant_number != 0)
			voice_select.variant = sp->voice_variant_number;
	}

	voice_select.name = voice_name;
	voice_select.identifier = identifier;
	voice_select.languages = language;

	v_id = SelectVoice(&voice_select, &voice_found);
	if (v_id == NULL)
		return "default";

	if ((strchr(v_id, '+') == NULL) && ((voice_select.gender == ENGENDER_UNKNOWN) || (voice_select.gender == base_voice->gender)) && (base_voice_variant_name[0] != 0)) {
		// a voice variant has not been selected, use the original voice variant
		char buf[80];
		sprintf(buf, "%s+%s", v_id, base_voice_variant_name);
		strncpy0(voice_name, buf, sizeof(voice_name));
		return voice_name;
	}
	return v_id;
}


static const wchar_t *GetSsmlAttribute(wchar_t *pw, const char *name)
{
	// Gets the value string for an attribute.
	// Returns NULL if the attribute is not present

	int ix;
	static const wchar_t empty[1] = { 0 };

	while (*pw != 0) {
		if (iswspace(pw[-1])) {
			ix = 0;
			while (*pw == name[ix]) {
				pw++;
				ix++;
			}
			if (name[ix] == 0) {
				// found the attribute, now get the value
				while (iswspace(*pw)) pw++;
				if (*pw == '=') pw++;
				while (iswspace(*pw)) pw++;
				if ((*pw == '"') || (*pw == '\'')) // allow single-quotes ?
					return pw+1;
				else if (iswspace(*pw) || (*pw == '/')) // end of attribute
					return empty;
				else
					return pw;
			}
		}
		pw++;
	}
	return NULL;
}


static int GetVoiceAttributes(wchar_t *pw, int tag_type, SSML_STACK *ssml_sp, SSML_STACK *ssml_stack, int n_ssml_stack, char current_voice_id[40], espeak_VOICE *base_voice, char *base_voice_variant_name)
{
	// Determines whether voice attribute are specified in this tag, and if so, whether this means
	// a voice change.
	// If it's a closing tag, delete the top frame of the stack and determine whether this implies
	// a voice change.
	// Returns  CLAUSE_TYPE_VOICE_CHANGE if there is a voice change

	const char *new_voice_id;

	static const MNEM_TAB mnem_gender[] = {
		{ "male", ENGENDER_MALE },
		{ "female", ENGENDER_FEMALE },
		{ "neutral", ENGENDER_NEUTRAL },
		{ NULL, ENGENDER_UNKNOWN }
	};

	if (tag_type & SSML_CLOSE) {
		// delete a stack frame
		if (n_ssml_stack > 1)
			n_ssml_stack--;
	} else {
		const wchar_t *lang;
    	const wchar_t *gender;
    	const wchar_t *name;
    	const wchar_t *age;
    	const wchar_t *variant;

		// add a stack frame if any voice details are specified
		lang = GetSsmlAttribute(pw, "xml:lang");

		if (tag_type != SSML_VOICE) {
			// only expect an xml:lang attribute
			name = NULL;
			variant = NULL;
			age = NULL;
			gender = NULL;
		} else {
			name = GetSsmlAttribute(pw, "name");
			variant = GetSsmlAttribute(pw, "variant");
			age = GetSsmlAttribute(pw, "age");
			gender = GetSsmlAttribute(pw, "gender");
		}

		if ((tag_type != SSML_VOICE) && (lang == NULL))
			return 0; // <s> or <p> without language spec, nothing to do

		ssml_sp = &ssml_stack[n_ssml_stack++];

		int value;

		attrcopy_utf8(ssml_sp->language, lang, sizeof(ssml_sp->language));
		attrcopy_utf8(ssml_sp->voice_name, name, sizeof(ssml_sp->voice_name));
		if ((value = attrnumber(variant, 1, 0)) > 0)
			value--; // variant='0' and variant='1' the same
		ssml_sp->voice_variant_number = value;
		ssml_sp->voice_age = attrnumber(age, 0, 0);
		ssml_sp->voice_gender = attrlookup(gender, mnem_gender);
		ssml_sp->tag_type = tag_type;
	}

	new_voice_id = VoiceFromStack(ssml_stack, n_ssml_stack, base_voice, base_voice_variant_name);
	if (strcmp(new_voice_id, current_voice_id) != 0) {
		// add an embedded command to change the voice
		strcpy(current_voice_id, new_voice_id);
		return CLAUSE_TYPE_VOICE_CHANGE;
	}

	return 0;
}

static void ProcessParamStack(char *outbuf, int *outix, int n_param_stack, PARAM_STACK *param_stack, int *speech_parameters)
{
	// Set the speech parameters from the parameter stack
	int param;
	int ix;
	char buf[20];
	int new_parameters[N_SPEECH_PARAM];
	static const char cmd_letter[N_SPEECH_PARAM] = { 0, 'S', 'A', 'P', 'R', 0, 'C', 0, 0, 0, 0, 0, 'F' }; // embedded command letters

	for (param = 0; param < N_SPEECH_PARAM; param++)
		new_parameters[param] = -1;

	for (ix = 0; ix < n_param_stack; ix++) {
		for (param = 0; param < N_SPEECH_PARAM; param++) {
			if (param_stack[ix].parameter[param] >= 0)
				new_parameters[param] = param_stack[ix].parameter[param];
		}
	}

	for (param = 0; param < N_SPEECH_PARAM; param++) {
		int value;
		if ((value = new_parameters[param]) != speech_parameters[param]) {
			buf[0] = 0;

			switch (param)
			{
			case espeakPUNCTUATION:
				option_punctuation = value-1;
				break;
			case espeakCAPITALS:
				option_capitals = value;
				break;
			case espeakRATE:
			case espeakVOLUME:
			case espeakPITCH:
			case espeakRANGE:
			case espeakEMPHASIS:
				sprintf(buf, "%c%d%c", CTRL_EMBEDDED, value, cmd_letter[param]);
				break;
			}

			speech_parameters[param] = new_parameters[param];
			strcpy(&outbuf[*outix], buf);
			*outix += strlen(buf);
		}
	}
}

static PARAM_STACK *PushParamStack(int tag_type, int *n_param_stack, PARAM_STACK *param_stack)
{
	int ix;
	PARAM_STACK *sp;

	sp = &param_stack[*n_param_stack];
	if (*n_param_stack < (N_PARAM_STACK-1))
		(*n_param_stack)++;

	sp->type = tag_type;
	for (ix = 0; ix < N_SPEECH_PARAM; ix++)
		sp->parameter[ix] = -1;
	return sp;
}

static void PopParamStack(int tag_type, char *outbuf, int *outix, int *n_param_stack, PARAM_STACK *param_stack, int *speech_parameters)
{
	// unwind the stack up to and including the previous tag of this type
	int ix;
	int top = 0;

	if (tag_type >= SSML_CLOSE)
		tag_type -= SSML_CLOSE;

	for (ix = 0; ix < *n_param_stack; ix++) {
		if (param_stack[ix].type == tag_type)
			top = ix;
	}
	if (top > 0)
		*n_param_stack = top;
	ProcessParamStack(outbuf, outix, *n_param_stack, param_stack, speech_parameters);
}

static int ReplaceKeyName(char *outbuf, int index, int *outix)
{
	// Replace some key-names by single characters, so they can be pronounced in different languages
	static const MNEM_TAB keynames[] = {
		{ "space ",        0xe020 },
		{ "tab ",          0xe009 },
		{ "underscore ",   0xe05f },
		{ "double-quote ", '"' },
		{ NULL,            0 }
	};

	int letter;
	char *p;

	p = &outbuf[index];

	if ((letter = LookupMnem(keynames, p)) != 0) {
		int ix;
		 ix = utf8_out(letter, p);
		*outix = index + ix;
		return letter;
	}
	return 0;
}

static void SetProsodyParameter(int param_type, const wchar_t *attr1, PARAM_STACK *sp, PARAM_STACK *param_stack, int *speech_parameters)
{
	int value;


	static const MNEM_TAB mnem_volume[] = {
		{ "default", 100 },
		{ "silent",    0 },
		{ "x-soft",   30 },
		{ "soft",     65 },
		{ "medium",  100 },
		{ "loud",    150 },
		{ "x-loud",  230 },
		{ NULL,       -1 }
	};

	static const MNEM_TAB mnem_rate[] = {
		{ "default", 100 },
		{ "x-slow",   60 },
		{ "slow",     80 },
		{ "medium",  100 },
		{ "fast",    125 },
		{ "x-fast",  160 },
		{ NULL,       -1 }
	};

	static const MNEM_TAB mnem_pitch[] = {
		{ "default", 100 },
		{ "x-low",    70 },
		{ "low",      85 },
		{ "medium",  100 },
		{ "high",    110 },
		{ "x-high",  120 },
		{ NULL,       -1 }
	};

	static const MNEM_TAB mnem_range[] = {
		{ "default", 100 },
		{ "x-low",    20 },
		{ "low",      50 },
		{ "medium",  100 },
		{ "high",    140 },
		{ "x-high",  180 },
		{ NULL,       -1 }
	};

	static const MNEM_TAB * const mnem_tabs[5] = {
		NULL, mnem_rate, mnem_volume, mnem_pitch, mnem_range
	};

	if ((value = attrlookup(attr1, mnem_tabs[param_type])) >= 0) {
		// mnemonic specifies a value as a percentage of the base pitch/range/rate/volume
		sp->parameter[param_type] = (param_stack[0].parameter[param_type] * value)/100;
	} else {
		int sign = attr_prosody_value(param_type, attr1, &value);

		if (sign == 0)
			sp->parameter[param_type] = value; // absolute value in Hz
		else if (sign == 2) {
			// change specified as percentage or in semitones
			sp->parameter[param_type] = (speech_parameters[param_type] * value)/100;
		} else {
			// change specified as plus or minus Hz
			sp->parameter[param_type] = speech_parameters[param_type] + (value*sign);
		}
	}
}

int ProcessSsmlTag(wchar_t *xml_buf, char *outbuf, int *outix, int n_outbuf, const char *xmlbase, bool *audio_text, char *current_voice_id, espeak_VOICE *base_voice, char *base_voice_variant_name, bool *ignore_text, bool *clear_skipping_text, int *sayas_mode, int *sayas_start, SSML_STACK *ssml_stack, int *n_ssml_stack, int *n_param_stack, int *speech_parameters)
{
	// xml_buf is the tag and attributes with a zero terminator in place of the original '>'
	// returns a clause terminator value.

	unsigned int ix;
	int index;
	int tag_type;
	int value;
	int value2;
	int value3;
	int voice_change_flag;
	wchar_t *px;
	const wchar_t *attr1;
	const wchar_t *attr2;
	const wchar_t *attr3;
	int terminator;
	int param_type;
	char tag_name[40];
	char buf[160];
	PARAM_STACK *sp;
	SSML_STACK *ssml_sp;

	// don't process comments and xml declarations
	if (wcsncmp(xml_buf, (wchar_t *) "!--", 3) == 0 || wcsncmp(xml_buf, (wchar_t *) "?xml", 4) == 0) {
		return 0;
		}

	// these tags have no effect if they are self-closing, eg. <voice />
	static const char ignore_if_self_closing[] = { 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0 };

	bool self_closing = false;
	int len;
	len = wcslen(xml_buf);
	if (xml_buf[len - 1] == '/') {
		// a self-closing tag
		xml_buf[len - 1] = ' ';
		self_closing = true;
	}

	static const MNEM_TAB mnem_phoneme_alphabet[] = {
		{ "espeak", 1 },
		{ NULL,    -1 }
	};

	static const MNEM_TAB mnem_punct[] = {
		{ "none", 1 },
		{ "all",  2 },
		{ "some", 3 },
		{ NULL,  -1 }
	};

	static const MNEM_TAB mnem_capitals[] = {
		{ "no",        0 },
		{ "icon",      1 },
		{ "spelling",  2 },
		{ "pitch",    20 },  // this is the amount by which to raise the pitch
		{ NULL,       -1 }
	};

	static const MNEM_TAB mnem_interpret_as[] = {
		{ "characters", SAYAS_CHARS },
		{ "tts:char",   SAYAS_SINGLE_CHARS },
		{ "tts:key",    SAYAS_KEY },
		{ "tts:digits", SAYAS_DIGITS },
		{ "telephone",  SAYAS_DIGITS1 },
		{ NULL,         -1 }
	};

	static const MNEM_TAB mnem_sayas_format[] = {
		{ "glyphs", 1 },
		{ NULL,    -1 }
	};

	static const MNEM_TAB mnem_break[] = {
		{ "none",     0 },
		{ "x-weak",   1 },
		{ "weak",     2 },
		{ "medium",   3 },
		{ "strong",   4 },
		{ "x-strong", 5 },
		{ NULL,      -1 }
	};

	static const MNEM_TAB mnem_emphasis[] = {
		{ "none",     1 },
		{ "reduced",  2 },
		{ "moderate", 3 },
		{ "strong",   4 },
		{ "x-strong", 5 },
		{ NULL,      -1 }
	};

	static const char * const prosody_attr[5] = {
		NULL, "rate", "volume", "pitch", "range"
	};

	for (ix = 0; ix < (sizeof(tag_name)-1); ix++) {
		int c;
		if (((c = xml_buf[ix]) == 0) || iswspace(c))
			break;
		tag_name[ix] = tolower((char)c);
	}
	tag_name[ix] = 0;

	px = &xml_buf[ix]; // the tag's attributes

	if (tag_name[0] == '/') {
		// closing tag
		if ((tag_type = LookupMnem(ssmltags, &tag_name[1])) != HTML_NOSPACE)
			outbuf[(*outix)++] = ' ';
		tag_type += SSML_CLOSE;
	} else {
		if ((tag_type = LookupMnem(ssmltags, tag_name)) != HTML_NOSPACE) {
			// separate SSML tags from the previous word (but not HMTL tags such as <b> <font> which can occur inside a word)
			outbuf[(*outix)++] = ' ';
		}

		if (self_closing && ignore_if_self_closing[tag_type])
			return 0;
	}

	voice_change_flag = 0;
	ssml_sp = &ssml_stack[*n_ssml_stack-1];

	switch (tag_type)
	{
	case SSML_STYLE:
		sp = PushParamStack(tag_type, n_param_stack, (PARAM_STACK *) param_stack);
		attr1 = GetSsmlAttribute(px, "field");
		attr2 = GetSsmlAttribute(px, "mode");


		if (attrcmp(attr1, "punctuation") == 0) {
			value = attrlookup(attr2, mnem_punct);
			sp->parameter[espeakPUNCTUATION] = value;
		} else if (attrcmp(attr1, "capital_letters") == 0) {
			value = attrlookup(attr2, mnem_capitals);
			sp->parameter[espeakCAPITALS] = value;
		}
		ProcessParamStack(outbuf, outix, *n_param_stack, param_stack, speech_parameters);
		break;
	case SSML_PROSODY:
		sp = PushParamStack(tag_type, n_param_stack, (PARAM_STACK *) param_stack);

		// look for attributes:  rate, volume, pitch, range
		for (param_type = espeakRATE; param_type <= espeakRANGE; param_type++) {
			if ((attr1 = GetSsmlAttribute(px, prosody_attr[param_type])) != NULL)
				SetProsodyParameter(param_type, attr1, sp, param_stack, speech_parameters);
		}

		ProcessParamStack(outbuf, outix, *n_param_stack, param_stack, speech_parameters);
		break;
	case SSML_EMPHASIS:
		sp = PushParamStack(tag_type, n_param_stack, (PARAM_STACK *) param_stack);
		value = 3; // default is "moderate"
		if ((attr1 = GetSsmlAttribute(px, "level")) != NULL)
			value = attrlookup(attr1, mnem_emphasis);

		if (translator->langopts.tone_language == 1) {
			static const unsigned char emphasis_to_pitch_range[] = { 50, 50, 40, 70, 90, 100 };
			static const unsigned char emphasis_to_volume[] = { 100, 100, 70, 110, 135, 150 };
			// tone language (eg.Chinese) do emphasis by increasing the pitch range.
			sp->parameter[espeakRANGE] = emphasis_to_pitch_range[value];
			sp->parameter[espeakVOLUME] = emphasis_to_volume[value];
		} else {
			static const unsigned char emphasis_to_volume2[] = { 100, 100, 75, 100, 120, 150 };
			sp->parameter[espeakVOLUME] = emphasis_to_volume2[value];
			sp->parameter[espeakEMPHASIS] = value;
		}
		ProcessParamStack(outbuf, outix, *n_param_stack, param_stack, speech_parameters);
		break;
	case SSML_STYLE + SSML_CLOSE:
	case SSML_PROSODY + SSML_CLOSE:
	case SSML_EMPHASIS + SSML_CLOSE:
		PopParamStack(tag_type, outbuf, outix, n_param_stack, (PARAM_STACK *) param_stack, (int *) speech_parameters);
		break;
	case SSML_PHONEME:
		attr1 = GetSsmlAttribute(px, "alphabet");
		attr2 = GetSsmlAttribute(px, "ph");
		value = attrlookup(attr1, mnem_phoneme_alphabet);
		if (value == 1) { // alphabet="espeak"
			outbuf[(*outix)++] = '[';
			outbuf[(*outix)++] = '[';
			*outix += attrcopy_utf8(&outbuf[*outix], attr2, n_outbuf-*outix);
			outbuf[(*outix)++] = ']';
			outbuf[(*outix)++] = ']';
		}
		break;
	case SSML_SAYAS:
		attr1 = GetSsmlAttribute(px, "interpret-as");
		attr2 = GetSsmlAttribute(px, "format");
		attr3 = GetSsmlAttribute(px, "detail");
		value = attrlookup(attr1, mnem_interpret_as);
		value2 = attrlookup(attr2, mnem_sayas_format);
		if (value2 == 1)
			value = SAYAS_GLYPHS;

		value3 = attrnumber(attr3, 0, 0);

		if (value == SAYAS_DIGITS) {
			if (value3 <= 1)
				value = SAYAS_DIGITS1;
			else
				value = SAYAS_DIGITS + value3;
		}

		sprintf(buf, "%c%dY", CTRL_EMBEDDED, value);
		strcpy(&outbuf[*outix], buf);
		*outix += strlen(buf);

		*sayas_start = *outix;
		*sayas_mode = value; // punctuation doesn't end clause during SAY-AS
		break;
	case SSML_SAYAS + SSML_CLOSE:
		if (*sayas_mode == SAYAS_KEY) {
			outbuf[*outix] = 0;
			ReplaceKeyName(outbuf, *sayas_start, outix);
		}

		outbuf[(*outix)++] = CTRL_EMBEDDED;
		outbuf[(*outix)++] = 'Y';
		*sayas_mode = 0;
		break;
	case SSML_SUB:
		if ((attr1 = GetSsmlAttribute(px, "alias")) != NULL) {
			// use the alias  rather than the text
			*ignore_text = true;
			*outix += attrcopy_utf8(&outbuf[*outix], attr1, n_outbuf-*outix);
		}
		break;
	case SSML_IGNORE_TEXT:
		*ignore_text = true;
		break;
	case SSML_SUB + SSML_CLOSE:
	case SSML_IGNORE_TEXT + SSML_CLOSE:
		*ignore_text = false;
		break;
	case SSML_MARK:
		if ((attr1 = GetSsmlAttribute(px, "name")) != NULL) {
			// add name to circular buffer of marker names
			attrcopy_utf8(buf, attr1, sizeof(buf));

			if ((buf[0] != 0) && (strcmp(skip_marker, buf) == 0)) {
				// This is the marker we are waiting for before starting to speak
				*clear_skipping_text = true;
				skip_marker[0] = 0;
				return CLAUSE_NONE;
			}

			if ((index = AddNameData(buf, 0)) >= 0) {
				sprintf(buf, "%c%dM", CTRL_EMBEDDED, index);
				strcpy(&outbuf[*outix], buf);
				*outix += strlen(buf);
			}
		}
		break;
	case SSML_AUDIO:
		sp = PushParamStack(tag_type, n_param_stack, (PARAM_STACK *)param_stack);

		if ((attr1 = GetSsmlAttribute(px, "src")) != NULL) {
			attrcopy_utf8(buf, attr1, sizeof(buf));

			if (uri_callback == NULL) {
				if ((xmlbase != NULL) && (buf[0] != '/')) {
					char fname[256];
					sprintf(fname, "%s/%s", xmlbase, buf);
					index = LoadSoundFile2(fname);
				} else
					index = LoadSoundFile2(buf);
				if (index >= 0) {
					sprintf(buf, "%c%dI", CTRL_EMBEDDED, index);
					strcpy(&outbuf[*outix], buf);
					*outix += strlen(buf);
					sp->parameter[espeakSILENCE] = 1;
				}
			} else {
				if ((index = AddNameData(buf, 0)) >= 0) {
					char *uri;
					uri = &namedata[index];
					if (uri_callback(1, uri, xmlbase) == 0) {
						sprintf(buf, "%c%dU", CTRL_EMBEDDED, index);
						strcpy(&outbuf[*outix], buf);
						*outix += strlen(buf);
						sp->parameter[espeakSILENCE] = 1;
					}
				}
			}
		}
		ProcessParamStack(outbuf, outix, *n_param_stack, param_stack, speech_parameters);

		if (self_closing)
			PopParamStack(tag_type, outbuf, outix, n_param_stack, (PARAM_STACK *) param_stack, (int *) speech_parameters);
		else
			*audio_text = true;
		return CLAUSE_NONE;
	case SSML_AUDIO + SSML_CLOSE:
		PopParamStack(tag_type, outbuf, outix, n_param_stack, (PARAM_STACK *) param_stack, (int *) speech_parameters);
		*audio_text = false;
		return CLAUSE_NONE;
	case SSML_BREAK:
		value = 21;
		terminator = CLAUSE_NONE;

		if ((attr1 = GetSsmlAttribute(px, "strength")) != NULL) {
			static const int break_value[6] = { 0, 7, 14, 21, 40, 80 }; // *10mS
			value = attrlookup(attr1, mnem_break);
			if (value < 0) value = 2;
			if (value < 3) {
				// adjust prepause on the following word
				sprintf(&outbuf[*outix], "%c%dB", CTRL_EMBEDDED, value);
				*outix += 3;
				terminator = 0;
			}
			value = break_value[value];
		}
		if ((attr2 = GetSsmlAttribute(px, "time")) != NULL) {
			value2 = attrnumber(attr2, 0, 1);   // pause in mS

			value2 = value2 * speech_parameters[espeakSSML_BREAK_MUL] / 100;

			int wpm = speech_parameters[espeakRATE];
			espeak_SetParameter(espeakRATE, wpm, 0);

			#if USE_LIBSONIC
			if (wpm >= espeakRATE_MAXIMUM) {
				// Compensate speedup with libsonic, see function SetSpeed()
				double sonic = ((double)wpm)/espeakRATE_NORMAL;
				value2 = value2 * sonic;
			}
			#endif

			// compensate for speaking speed to keep constant pause length, see function PauseLength()
			// 'value' here is x 10mS
			value = (value2 * 256) / (speed.clause_pause_factor * 10);
			if (value < 200)
				value = (value2 * 256) / (speed.pause_factor * 10);

			if (terminator == 0)
				terminator = CLAUSE_NONE;
		}
		if (terminator) {
			if (value > 0xfff) {
				// scale down the value and set a scaling indicator bit
				value = value / 32;
				if (value > 0xfff)
					value = 0xfff;
				terminator |= CLAUSE_PAUSE_LONG;
			}
			return terminator + value;
		}
		break;
	case SSML_SPEAK:
		if ((attr1 = GetSsmlAttribute(px, "xml:base")) != NULL) {
			attrcopy_utf8(buf, attr1, sizeof(buf));
			if ((index = AddNameData(buf, 0)) >= 0)
				xmlbase = &namedata[index];
		}
		if (GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name) == 0)
			return 0; // no voice change
		return CLAUSE_VOICE;
	case SSML_VOICE:
		if (GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name) == 0)
			return 0; // no voice change
		return CLAUSE_VOICE;
	case SSML_SPEAK + SSML_CLOSE:
		// unwind stack until the previous <voice> or <speak> tag
		while ((*n_ssml_stack > 1) && (ssml_stack[*n_ssml_stack-1].tag_type != SSML_SPEAK))
			(*n_ssml_stack)--;
		return CLAUSE_PERIOD + GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
	case SSML_VOICE + SSML_CLOSE:
		// unwind stack until the previous <voice> or <speak> tag
		while ((*n_ssml_stack > 1) && (ssml_stack[*n_ssml_stack-1].tag_type != SSML_VOICE))
			(*n_ssml_stack)--;

		terminator = 0; // ??  Sentence intonation, but no pause ??
		return terminator + GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
	case HTML_BREAK:
	case HTML_BREAK + SSML_CLOSE:
		return CLAUSE_COLON;
	case SSML_SENTENCE:
		if (ssml_sp->tag_type == SSML_SENTENCE) {
			// new sentence implies end-of-sentence
			voice_change_flag = GetVoiceAttributes(px, SSML_SENTENCE+SSML_CLOSE, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
		}
		voice_change_flag |= GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
		return CLAUSE_PARAGRAPH + voice_change_flag;
	case SSML_PARAGRAPH:
		if (ssml_sp->tag_type == SSML_SENTENCE) {
			// new paragraph implies end-of-sentence or end-of-paragraph
			voice_change_flag = GetVoiceAttributes(px, SSML_SENTENCE+SSML_CLOSE, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
		}
		if (ssml_sp->tag_type == SSML_PARAGRAPH) {
			// new paragraph implies end-of-sentence or end-of-paragraph
			voice_change_flag |= GetVoiceAttributes(px, SSML_PARAGRAPH+SSML_CLOSE, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
		}
		voice_change_flag |= GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
		return CLAUSE_PARAGRAPH + voice_change_flag;
	case SSML_SENTENCE + SSML_CLOSE:
		if (ssml_sp->tag_type == SSML_SENTENCE) {
			// end of a sentence which specified a language
			voice_change_flag = GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name);
		}
		return CLAUSE_PERIOD + voice_change_flag;
	case SSML_PARAGRAPH + SSML_CLOSE:
		if ((ssml_sp->tag_type == SSML_SENTENCE) || (ssml_sp->tag_type == SSML_PARAGRAPH)) {
			// End of a paragraph which specified a language.
			// (End-of-paragraph also implies end-of-sentence)
			return GetVoiceAttributes(px, tag_type, ssml_sp, ssml_stack, *n_ssml_stack, current_voice_id, base_voice, base_voice_variant_name) + CLAUSE_PARAGRAPH;
		}
		return CLAUSE_PARAGRAPH;
	}
	return 0;
}

#pragma GCC visibility push(default)
ESPEAK_API void espeak_SetUriCallback(int (*UriCallback)(int, const char *, const char *))
{
	uri_callback = UriCallback;
}
#pragma GCC visibility pop

static const MNEM_TAB xml_entity_mnemonics[] = {
	{ "gt",   '>' },
	{ "lt",   0xe000 + '<' },   // private usage area, to avoid confusion with XML tag
	{ "amp",  '&' },
	{ "quot", '"' },
	{ "nbsp", ' ' },
	{ "apos", '\'' },
	{ NULL,   -1 }
};

int ParseSsmlReference(char *ref, int *c1, int *c2) {
	// Check if buffer *ref contains an XML character or entity reference
	// if found, set *c1 to the replacement char
	// change *c2 for entity references
	// returns >= 0 on success

	if (ref[0] == '#') {
		// character reference
		if (ref[1] == 'x')
			return sscanf(&ref[2], "%x", c1);
		else
			return sscanf(&ref[1], "%d", c1);
	} else { 
		// entity reference
		int found;
		if ((found = LookupMnem(xml_entity_mnemonics, ref)) != -1) {
			*c1 = found;
			if (*c2 == 0)
				*c2 = ' ';
			return found;
		}
	}
	return -1;
}
