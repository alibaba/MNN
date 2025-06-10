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
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "synthdata.h"
#include "common.h"                    // for GetFileLength
#include "error.h"                    // for create_file_error_context, crea...
#include "phoneme.h"                  // for PHONEME_TAB, PHONEME_TAB_LIST
#include "speech.h"                   // for path_home, PATHSEP
#include "mbrola.h"                   // for mbrola_name
#include "soundicon.h"               // for soundicon_tab
#include "synthesize.h"               // for PHONEME_LIST, frameref_t, PHONE...
#include "translate.h"                // for Translator, LANGUAGE_OPTIONS
#include "voice.h"                    // for ReadTonePoints, tone_points, voice

int n_tunes = 0;
TUNE *tunes = NULL;

const int version_phdata  = 0x014801;

// copy the current phoneme table into here
int n_phoneme_tab;
static int current_phoneme_table;
PHONEME_TAB *phoneme_tab[N_PHONEME_TAB];

static unsigned short *phoneme_index = NULL;
static char *phondata_ptr = NULL;
unsigned char *wavefile_data = NULL;
static unsigned char *phoneme_tab_data = NULL;

static int n_phoneme_tables;
PHONEME_TAB_LIST phoneme_tab_list[N_PHONEME_TABS];
int phoneme_tab_number = 0;

int seq_len_adjust;

static espeak_ng_STATUS ReadPhFile(void **ptr, const char *fname, int *size, espeak_ng_ERROR_CONTEXT *context)
{
	if (!ptr) return EINVAL;

	FILE *f_in;
	int length;
	char buf[sizeof(path_home)+40];

	sprintf(buf, "%s%c%s", path_home, PATHSEP, fname);
	length = GetFileLength(buf);
	if (length < 0) // length == -errno
		return create_file_error_context(context, -length, buf);

	if ((f_in = fopen(buf, "rb")) == NULL)
		return create_file_error_context(context, errno, buf);

	if (*ptr != NULL) {
		free(*ptr);
		*ptr = NULL;
	}
	
	if (length == 0) {
		*ptr = NULL;
		return 0;
	}

	if ((*ptr = malloc(length)) == NULL) {
		fclose(f_in);
		return ENOMEM;
	}
	if (fread(*ptr, 1, length, f_in) != length) {
		int error = errno;
		fclose(f_in);
		free(*ptr);
		*ptr = NULL;
		return create_file_error_context(context, error, buf);
	}

	fclose(f_in);
	if (size != NULL)
		*size = length;
	return ENS_OK;
}

espeak_ng_STATUS LoadPhData(int *srate, espeak_ng_ERROR_CONTEXT *context)
{
	int ix;
	int version;
	int length = 0;
	int rate;
	unsigned char *p;

	espeak_ng_STATUS status;
	if ((status = ReadPhFile((void **)&phoneme_tab_data, "phontab", NULL, context)) != ENS_OK)
		return status;
	if ((status = ReadPhFile((void **)&phoneme_index, "phonindex", NULL, context)) != ENS_OK)
		return status;
	if ((status = ReadPhFile((void **)&phondata_ptr, "phondata", NULL, context)) != ENS_OK)
		return status;
	if ((status = ReadPhFile((void **)&tunes, "intonations", &length, context)) != ENS_OK)
		return status;
	wavefile_data = (unsigned char *)phondata_ptr;
	n_tunes = length / sizeof(TUNE);

	// read the version number and sample rate from the first 8 bytes of phondata
	version = 0; // bytes 0-3, version number
	rate = 0;    // bytes 4-7, sample rate
	if (wavefile_data) {
		for (ix = 0; ix < 4; ix++) {
			version += (wavefile_data[ix] << (ix*8));
			rate += (wavefile_data[ix+4] << (ix*8));
		}
	}

	if (version != version_phdata)
		return create_version_mismatch_error_context(context, path_home, version, version_phdata);

	// set up phoneme tables
	p = phoneme_tab_data;
	n_phoneme_tables = p[0];
	p += 4;

	for (ix = 0; ix < n_phoneme_tables; ix++) {
		int n_phonemes = p[0];
		phoneme_tab_list[ix].n_phonemes = p[0];
		phoneme_tab_list[ix].includes = p[1];
		p += 4;
		memcpy(phoneme_tab_list[ix].name, p, N_PHONEME_TAB_NAME);
		p += N_PHONEME_TAB_NAME;
		phoneme_tab_list[ix].phoneme_tab_ptr = (PHONEME_TAB *)p;
		p += (n_phonemes * sizeof(PHONEME_TAB));
	}

	if (phoneme_tab_number >= n_phoneme_tables)
		phoneme_tab_number = 0;

	if (srate != NULL)
		*srate = rate;
	return ENS_OK;
}

void FreePhData(void)
{
	free(phoneme_tab_data);
	free(phoneme_index);
	free(phondata_ptr);
	free(tunes);
	phoneme_tab_data = NULL;
	phoneme_index = NULL;
	phondata_ptr = NULL;
	tunes = NULL;
	current_phoneme_table = -1;
}

int PhonemeCode(unsigned int mnem)
{
	int ix;

	for (ix = 0; ix < n_phoneme_tab; ix++) {
		if (phoneme_tab[ix] == NULL)
			continue;
		if (phoneme_tab[ix]->mnemonic == mnem)
			return phoneme_tab[ix]->code;
	}
	return 0;
}

int LookupPhonemeString(const char *string)
{
	int ix;
	unsigned int mnem;

	// Pack up to 4 characters into a word
	mnem = 0;
	for (ix = 0; ix < 4; ix++) {
		if (string[ix] == 0) break;
		unsigned char c = string[ix];
		mnem |= (c << (ix*8));
	}

	return PhonemeCode(mnem);
}

frameref_t *LookupSpect(PHONEME_TAB *this_ph, int which, FMT_PARAMS *fmt_params,  int *n_frames, PHONEME_LIST *plist)
{
	int ix;
	int nf;
	int nf1;
	int seq_break;
	frameref_t *frames;
	int length1;
	SPECT_SEQ *seq, *seq2;
	SPECT_SEQK *seqk, *seqk2;
	frame_t *frame;
	static frameref_t frames_buf[N_SEQ_FRAMES];

	MAKE_MEM_UNDEFINED(&frames_buf, sizeof(frames_buf));

	seq = (SPECT_SEQ *)(&phondata_ptr[fmt_params->fmt_addr]);
	seqk = (SPECT_SEQK *)seq;
	nf = seq->n_frames;

	if (nf >= N_SEQ_FRAMES)
		nf = N_SEQ_FRAMES - 1;

	seq_len_adjust = fmt_params->fmt2_lenadj + fmt_params->fmt_length;
	seq_break = 0;

	for (ix = 0; ix < nf; ix++) {
		if (seq->frame[0].frflags & FRFLAG_KLATT)
			frame = &seqk->frame[ix];
		else
			frame = (frame_t *)&seq->frame[ix];
		frames_buf[ix].frame = frame;
		frames_buf[ix].frflags = frame->frflags;
		frames_buf[ix].length = frame->length;
		if (frame->frflags & FRFLAG_VOWEL_CENTRE)
			seq_break = ix;
	}

	frames = &frames_buf[0];
	if (seq_break > 0) {
		if (which == 1)
			nf = seq_break + 1;
		else {
			frames = &frames_buf[seq_break]; // body of vowel, skip past initial frames
			nf -= seq_break;
		}
	}

	// do we need to modify a frame for blending with a consonant?
	if ((this_ph->type == phVOWEL) && (fmt_params->fmt2_addr == 0) && (fmt_params->use_vowelin))
		seq_len_adjust += FormantTransition2(frames, &nf, fmt_params->transition0, fmt_params->transition1, NULL, which);

	length1 = 0;
	nf1 = nf - 1;
	for (ix = 0; ix < nf1; ix++)
		length1 += frames[ix].length;

	if (fmt_params->fmt2_addr != 0) {
		// a secondary reference has been returned, which is not a wavefile
		// add these spectra to the main sequence
		seq2 = (SPECT_SEQ *)(&phondata_ptr[fmt_params->fmt2_addr]);
		seqk2 = (SPECT_SEQK *)seq2;

		// first frame of the addition just sets the length of the last frame of the main seq
		nf--;
		for (ix = 0; ix < seq2->n_frames; ix++) {
			if (seq2->frame[0].frflags & FRFLAG_KLATT)
				frame = &seqk2->frame[ix];
			else
				frame = (frame_t *)&seq2->frame[ix];

			frames[nf].length = frame->length;
			if (ix > 0) {
				frames[nf].frame = frame;
				frames[nf].frflags = frame->frflags;
			}
			nf++;
		}
	}

	if (length1 > 0) {
		int length_factor;
		if (which == 2) {
			// adjust the length of the main part to match the standard length specified for the vowel
			// less the front part of the vowel and any added suffix

			int length_std = fmt_params->std_length + seq_len_adjust - 45;
			if (length_std < 10)
				length_std = 10;
			if (plist->synthflags & SFLAG_LENGTHEN)
				length_std += (phoneme_tab[phonLENGTHEN]->std_length * 2); // phoneme was followed by an extra : symbol

			// can adjust vowel length for stressed syllables here

			length_factor = (length_std * 256)/ length1;

			for (ix = 0; ix < nf1; ix++)
				frames[ix].length = (frames[ix].length * length_factor)/256;
		} else {
			if (which == 1) {
				// front of a vowel
				if (fmt_params->fmt_control == 1) {
					// This is the default start of a vowel.
					// Allow very short vowels to have shorter front parts
					if (fmt_params->std_length < 130)
						frames[0].length = (frames[0].length * fmt_params->std_length)/130;
				}
			} else {
				// not a vowel
				if (fmt_params->std_length > 0)
					seq_len_adjust += (fmt_params->std_length - length1);
			}

			if (seq_len_adjust != 0) {
				length_factor = ((length1 + seq_len_adjust) * 256)/length1;
				for (ix = 0; ix < nf1; ix++)
					frames[ix].length = (frames[ix].length * length_factor)/256;
			}
		}
	}

	*n_frames = nf;
	return frames;
}

const unsigned char *GetEnvelope(int index)
{
	if (index == 0) {
		fprintf(stderr, "espeak: No envelope\n");
		return envelope_data[0]; // not found, use a default envelope
	}
	return (unsigned char *)&phondata_ptr[index];
}

static void SetUpPhonemeTable(int number)
{
	int ix;
	int includes;
	PHONEME_TAB *phtab;

	if ((includes = phoneme_tab_list[number].includes) > 0) {
		// recursively include base phoneme tables
		SetUpPhonemeTable(includes - 1);
	}

	// now add the phonemes from this table
	phtab = phoneme_tab_list[number].phoneme_tab_ptr;
	for (ix = 0; ix < phoneme_tab_list[number].n_phonemes; ix++) {
		int ph_code = phtab[ix].code;
		phoneme_tab[ph_code] = &phtab[ix];
		if (ph_code > n_phoneme_tab) {
			memset(&phoneme_tab[n_phoneme_tab+1], 0, (ph_code - (n_phoneme_tab+1)) * sizeof(*phoneme_tab));
			n_phoneme_tab = ph_code;
		}
	}
}

void SelectPhonemeTable(int number)
{
	if (current_phoneme_table == number) return;
	n_phoneme_tab = 0;
	MAKE_MEM_UNDEFINED(&phoneme_tab, sizeof(phoneme_tab));
	SetUpPhonemeTable(number); // recursively for included phoneme tables
	n_phoneme_tab++;
	current_phoneme_table = number;
}

int LookupPhonemeTable(const char *name)
{
	int ix;

	for (ix = 0; ix < n_phoneme_tables; ix++) {
		if (strcmp(name, phoneme_tab_list[ix].name) == 0) {
			phoneme_tab_number = ix;
			break;
		}
	}
	if (ix == n_phoneme_tables)
		return -1;

	return ix;
}

int SelectPhonemeTableName(const char *name)
{
	// Look up a phoneme set by name, and select it if it exists
	// Returns the phoneme table number
	int ix;

	if ((ix = LookupPhonemeTable(name)) == -1)
		return -1;

	SelectPhonemeTable(ix);
	return ix;
}

static void InvalidInstn(PHONEME_TAB *ph, int instn)
{
	char buf[5];
	fprintf(stderr, "Invalid instruction %.4x for phoneme '%s'\n", instn, WordToString(buf, ph->mnemonic));
}

static bool StressCondition(Translator *tr, PHONEME_LIST *plist, int condition, int control)
{
	int stress_level;
	PHONEME_LIST *pl;
	static const int condition_level[4] = { 1, 2, 4, 15 };

	if (phoneme_tab[plist[0].phcode]->type == phVOWEL)
		pl = plist;
	else {
		// consonant, get stress from the following vowel
		if (phoneme_tab[plist[1].phcode]->type == phVOWEL)
			pl = &plist[1];
		else
			return false; // no stress elevel for this consonant
	}

	stress_level = pl->stresslevel & 0xf;

	if (tr != NULL) {
		if ((control & 1) && (plist->synthflags & SFLAG_DICTIONARY) && ((tr->langopts.param[LOPT_REDUCE] & 1) == 0)) {
			// change phoneme.  Don't change phonemes which are given for the word in the dictionary.
			return false;
		}

		if ((tr->langopts.param[LOPT_REDUCE] & 0x2) && (stress_level >= pl->wordstress)) {
			// treat the most stressed syllable in an unstressed word as stressed
			stress_level = STRESS_IS_PRIMARY;
		}
	}

	if (condition == STRESS_IS_PRIMARY)
		return stress_level >= pl->wordstress;

	if (condition == STRESS_IS_SECONDARY) {
		if (stress_level > STRESS_IS_SECONDARY)
			return true;
	} else {
		if (stress_level < condition_level[condition])
			return true;
	}
	return false;

}

static int CountVowelPosition(PHONEME_LIST *plist, PHONEME_LIST *plist_start)
{
	int count = 0;

	for (;;) {
		if (plist->ph->type == phVOWEL)
			count++;
		if (plist->sourceix != 0 || plist == plist_start)
			break;
		plist--;
	}
	return count;
}

static bool InterpretCondition(Translator *tr, int control, PHONEME_LIST *plist, PHONEME_LIST *plist_start, unsigned short *p_prog, WORD_PH_DATA *worddata)
{
	unsigned int data;
	int instn;
	int instn2;

	// instruction: 2xxx, 3xxx

	// bits 8-10 = 0 to 5,  which phoneme, =6 the 'which' information is in the next instruction.
	// bit 11 = 0, bits 0-7 are a phoneme code
	// bit 11 = 1, bits 5-7 type of data, bits 0-4 data value

	// bits 8-10 = 7,  other conditions

	instn = (*p_prog) & 0xfff;
	data = instn & 0xff;
	instn2 = instn >> 8;

	if (instn2 < 14) {
		PHONEME_LIST *plist_this;
		plist_this = plist;
		int which = (instn2) % 7;

		if (which == 6) {
			// the 'which' code is in the next instruction
			p_prog++;
			which = (*p_prog);
		}

		if (which == 4) {
			// nextPhW not word boundary
			if (plist[1].sourceix)
				return false;
		}
		if (which == 5) {
			// prevPhW, not word boundary
			if (plist[0].sourceix)
				return false;
		}
		if (which == 6) {
			// next2PhW, not word boundary
			if (plist[1].sourceix || plist[2].sourceix)
				return false;
		}

		bool check_endtype = false;
		switch (which)
		{
		case 0: // prevPh
		case 5: // prevPhW
			if (plist < plist_start+1)
				return false;
			plist--;
			check_endtype = true;
			break;
		case 1: // thisPh
			break;
		case 2: // nextPh
		case 4: // nextPhW
			plist++;
			break;
		case 3: // next2Ph
		case 6: // next2PhW
			plist += 2;
			break;
		case 7:
			// nextVowel, not word boundary
			for (which = 1;; which++) {
				if (plist[which].sourceix)
					return false;
				if (phoneme_tab[plist[which].phcode]->type == phVOWEL) {
					plist = &plist[which];
					break;
				}
			}
			break;
		case 8: // prevVowel in this word
			if ((worddata == NULL) || (worddata->prev_vowel.ph == NULL))
				return false; // no previous vowel
			plist = &(worddata->prev_vowel);
			check_endtype = true;
			break;
		case 9: // next3PhW
			for (int ix = 1; ix <= 3; ix++) {
				if (plist[ix].sourceix)
					return false;
			}
			plist = &plist[3];
			break;
		case 10: // prev2PhW
			if (plist < plist_start + 2)
				return false;
			if ((plist[0].sourceix) || (plist[-1].sourceix))
				return false;
			plist -= 2;
			check_endtype = true;
			break;
		}

		if ((which == 0) || (which == 5)) {
			if (plist->phcode == 1) {
				if (plist <= plist_start)
					return false;
				// This is a NULL phoneme, a phoneme has been deleted so look at the previous phoneme
				plist--;
			}
		}

		if (control & 0x100) {
			// "change phonemes" pass
			plist->ph = phoneme_tab[plist->phcode];
		}
		PHONEME_TAB *ph;
		ph = plist->ph;

		if (instn2 < 7) {
			// 'data' is a phoneme number
			if ((phoneme_tab[data]->mnemonic == ph->mnemonic) == true)
				return true;

			//  not an exact match, check for a vowel type (eg. #i )
			if ((check_endtype) && (ph->type == phVOWEL))
				return data == ph->end_type; // prevPh() match on end_type
			return data == ph->start_type; // thisPh() or nextPh(), match on start_type
		}

		data = instn & 0x1f;

		switch (instn & 0xe0)
		{
		case CONDITION_IS_PHONEME_TYPE:
			return ph->type == data;
		case CONDITION_IS_PLACE_OF_ARTICULATION:
			return ((ph->phflags >> 16) & 0xf) == data;
		case CONDITION_IS_PHFLAG_SET:
			return (ph->phflags & (1 << data)) != 0;
		case CONDITION_IS_OTHER:
			switch (data)
			{
			case STRESS_IS_DIMINISHED:
			case STRESS_IS_UNSTRESSED:
			case STRESS_IS_NOT_STRESSED:
			case STRESS_IS_SECONDARY:
			case STRESS_IS_PRIMARY:
				return StressCondition(tr, plist, data, 0);
			case isBreak:
				return (ph->type == phPAUSE) || (plist_this->synthflags & SFLAG_NEXT_PAUSE);
			case isWordStart:
				return plist->sourceix != 0;
			case isWordEnd:
				return plist[1].sourceix || (plist[1].ph->type == phPAUSE);
			case isAfterStress:
				if (plist->sourceix != 0)
					return false;
				do {
					if (plist <= plist_start)
						return false;
					plist--;
					if ((plist->stresslevel & 0xf) >= 4)
						return true;

				} while (plist->sourceix == 0);
				break;
			case isNotVowel:
				return ph->type != phVOWEL;
			case isFinalVowel:
				for (;;) {
					plist++;
					if (plist->sourceix != 0)
						return true; // start of next word, without finding another vowel
					if (plist->ph->type == phVOWEL)
						return false;
				}
			case isVoiced:
				return (ph->type == phVOWEL) || (ph->type == phLIQUID) || (ph->phflags & phVOICED);
			case isFirstVowel:
				return CountVowelPosition(plist, plist_start) == 1;
			case isSecondVowel:
				return CountVowelPosition(plist, plist_start) == 2;
			case isTranslationGiven:
				return (plist->synthflags & SFLAG_DICTIONARY) != 0;
			}
			break;

		}
		return false;
	} else if (instn2 == 0xf) {
		// Other conditions
		switch (data)
		{
		case 1: // PreVoicing
			return control & 1;
#if USE_KLATT
		case 2: // KlattSynth
			return voice->klattv[0] != 0;
#endif
#if USE_MBROLA
		case 3: // MbrolaSynth
			return mbrola_name[0] != 0;
#endif
		}
	}
	return false;
}

static void SwitchOnVowelType(PHONEME_LIST *plist, PHONEME_DATA *phdata, unsigned short **p_prog, int instn_type)
{
	int voweltype;


	if (instn_type == 2) {
		phdata->pd_control |= pd_FORNEXTPH;
		voweltype = plist[1].ph->start_type; // SwitchNextVowelType
	} else
		voweltype = plist[-1].ph->end_type; // SwitchPrevVowelType

	voweltype -= phonVOWELTYPES;
	if ((voweltype >= 0) && (voweltype < 6)) {
		unsigned short *prog;
		signed char x;

		prog = *p_prog + voweltype*2;
		phdata->sound_addr[instn_type] = (((prog[1] & 0xf) << 16) + prog[2]) * 4;
		x = (prog[1] >> 4) & 0xff;
		phdata->sound_param[instn_type] = x; // sign extend
	}

	*p_prog += 12;
}

static int NumInstnWords(unsigned short *prog)
{
	int instn;
	int instn2;
	int instn_type;
	int n;
	int type2;
	static const char n_words[16] = { 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 4, 0, 0, 0, 0, 0 };

	instn = *prog;
	instn_type = instn >> 12;
	if ((n = n_words[instn_type]) > 0)
		return n;

	switch (instn_type)
	{
	case 0:
		if (((instn & 0xf00) >> 8) == i_IPA_NAME) {
			n = ((instn & 0xff) + 1) / 2;
			return n+1;
		}
		return 1;
	case 6:
		type2 = (instn & 0xf00) >> 9;
		if ((type2 == 5) || (type2 == 6))
			return 12; // switch on vowel type
		return 1;
	case 2:
	case 3:
		// a condition, check for a 2-word instruction
		if (((n = instn & 0x0f00) == 0x600) || (n == 0x0d00))
			return 2;
		return 1;
	default:
		// instn_type 11 to 15, 2 words
		instn2 = prog[2];
		if ((instn2 >> 12) == 0xf) {
			// This instruction is followed by addWav(), 2 more words
			return 4;
		}
		if (instn2 == INSTN_CONTINUE)
			return 3;
		return 2;
	}
}

void InterpretPhoneme(Translator *tr, int control, PHONEME_LIST *plist, PHONEME_LIST *plist_start, PHONEME_DATA *phdata, WORD_PH_DATA *worddata)
{
	// control:
	// bit 0:  PreVoicing
	// bit 8:  change phonemes

	PHONEME_TAB *ph;
	unsigned short *prog;
	int or_flag;
	bool truth;
	bool truth2;
	int data;
	int end_flag;
	int ix;
	signed char param_sc;

	#define N_RETURN 10
	int n_return = 0;
	unsigned short *return_addr[N_RETURN]; // return address stack

	ph = plist->ph;

	if ((worddata != NULL) && (plist->sourceix)) {
		// start of a word, reset word data
		worddata->prev_vowel.ph = NULL;
	}

	memset(phdata, 0, sizeof(PHONEME_DATA));
	phdata->pd_param[i_SET_LENGTH] = ph->std_length;
	phdata->pd_param[i_LENGTH_MOD] = ph->length_mod;

	if (ph->program == 0)
		return;

	end_flag = 0;

	for (prog = &phoneme_index[ph->program]; end_flag != 1; prog++) {
		unsigned short instn;
		int instn2;

		instn = *prog;
		instn2 = (instn >> 8) & 0xf;

		switch (instn >> 12)
		{
		case 0: // 0xxx
			data = instn & 0xff;

			if (instn2 == 0) {
				// instructions with no operand
				switch (data)
				{
				case INSTN_RETURN:
					end_flag = 1;
					break;
				case INSTN_CONTINUE:
					break;
				default:
					InvalidInstn(ph, instn);
					break;
				}
			} else if (instn2 == i_APPEND_IFNEXTVOWEL) {
				if (phoneme_tab[plist[1].phcode]->type == phVOWEL)
					phdata->pd_param[i_APPEND_PHONEME] = data;
			} else if (instn2 == i_ADD_LENGTH) {
				if (data & 0x80) {
					// a negative value, do sign extension
					data = -(0x100 - data);
				}
				phdata->pd_param[i_SET_LENGTH] += data;
			} else if (instn2 == i_IPA_NAME) {
				// followed by utf-8 characters, 2 per instn word
				for (ix = 0; (ix < data) && (ix < 16); ix += 2) {
					prog++;
					phdata->ipa_string[ix] = prog[0] >> 8;
					phdata->ipa_string[ix+1] = prog[0] & 0xff;
				}
				phdata->ipa_string[ix] = 0;
			} else if (instn2 < N_PHONEME_DATA_PARAM) {
				phdata->pd_param[instn2] = data;
				if ((instn2 == i_CHANGE_PHONEME) && (control & 0x100)) {
					// found ChangePhoneme() in PhonemeList mode, exit
					end_flag = 1;
				}
			} else
				InvalidInstn(ph, instn);
			break;
		case 1:
			if (tr == NULL)
				break; // ignore if in synthesis stage

			if (instn2 < 8) {
				// ChangeIf
				if (StressCondition(tr, plist, instn2 & 7, 1) == true) {
					phdata->pd_param[i_CHANGE_PHONEME] = instn & 0xff;
					end_flag = 1; // change phoneme, exit
				}
			}
			break;
		case 2:
		case 3:
			// conditions
			or_flag = 0;
			truth = true;
			while ((instn & 0xe000) == 0x2000) {
				// process a sequence of conditions, using  boolean accumulator
				truth2 = InterpretCondition(tr, control, plist, plist_start, prog, worddata);
				prog += NumInstnWords(prog);
				if (*prog == i_NOT) {
					truth2 = truth2 ^ 1;
					prog++;
				}

				if (or_flag)
					truth = truth || truth2;
				else
					truth = truth && truth2;
				or_flag = instn & 0x1000;
				instn = *prog;
			}

			if (truth == false) {
				if ((instn & 0xf800) == i_JUMP_FALSE)
					prog += instn & 0xff;
				else {
					// instruction after a condition is not JUMP_FALSE, so skip the instruction.
					prog += NumInstnWords(prog);
					if ((prog[0] & 0xfe00) == 0x6000)
						prog++; // and skip ELSE jump
				}
			}
			prog--;
			break;
		case 6:
			// JUMP
			switch (instn2 >> 1)
			{
			case 0:
				prog += (instn & 0xff) - 1;
				break;
			case 4:
				// conditional jumps should have been processed in the Condition section
				break;
			case 5: // NexttVowelStarts
				SwitchOnVowelType(plist, phdata, &prog, 2);
				break;
			case 6: // PrevVowelTypeEndings
				SwitchOnVowelType(plist, phdata, &prog, 3);
				break;
			}
			break;
		case 9:
			data = ((instn & 0xf) << 16) + prog[1];
			prog++;
			switch (instn2)
			{
			case 1:
				// call a procedure or another phoneme
				if (n_return < N_RETURN) {
					return_addr[n_return++] = prog;
					prog = &phoneme_index[data] - 1;
				}
				break;
			case 2:
				// pitch envelope
				phdata->pitch_env = data;
				break;
			case 3:
				// amplitude envelope
				phdata->amp_env = data;
				break;
			}
			break;
		case 10: //  Vowelin, Vowelout
			if (instn2 == 1)
				ix = 0;
			else
				ix = 2;

			phdata->vowel_transition[ix] = ((prog[0] & 0xff) << 16) + prog[1];
			phdata->vowel_transition[ix+1] = (prog[2] << 16) + prog[3];
			prog += 3;
			break;
		case 11: // FMT
		case 12: // WAV
		case 13: // VowelStart
		case 14: // VowelEnd
		case 15: // addWav
			instn2 = (instn >> 12) - 11;
			phdata->sound_addr[instn2] = ((instn & 0xf) << 18) + (prog[1] << 2);
			param_sc = phdata->sound_param[instn2] = (instn >> 4) & 0xff;
			prog++;

			if (prog[1] != INSTN_CONTINUE) {
				if (instn2 < 2) {
					// FMT() and WAV() imply Return
					end_flag = 1;
					if ((prog[1] >> 12) == 0xf) {
						// Return after the following addWav()
						end_flag = 2;
					}
				} else if (instn2 == pd_ADDWAV) {
					// addWav(), return if previous instruction was FMT() or WAV()
					end_flag--;
				}

				if ((instn2 == pd_VWLSTART) || (instn2 == pd_VWLEND)) {
					// VowelStart or VowelEnding.
					phdata->sound_param[instn2] = param_sc;   // sign extend
				}
			}
			break;
		default:
			InvalidInstn(ph, instn);
			break;
		}

		if ((end_flag == 1) && (n_return > 0)) {
			// return from called procedure or phoneme
			end_flag = 0;
			prog = return_addr[--n_return];
		}
	}

	if ((worddata != NULL) && (plist->type == phVOWEL))
		memcpy(&worddata->prev_vowel, &plist[0], sizeof(PHONEME_LIST));

	plist->std_length = phdata->pd_param[i_SET_LENGTH];
	if (phdata->sound_addr[0] != 0) {
		plist->phontab_addr = phdata->sound_addr[0]; // FMT address
		plist->sound_param = phdata->sound_param[0];
	} else {
		plist->phontab_addr = phdata->sound_addr[1]; // WAV address
		plist->sound_param = phdata->sound_param[1];
	}
}

void InterpretPhoneme2(int phcode, PHONEME_DATA *phdata)
{
	// Examine the program of a single isolated phoneme
	int ix;
	PHONEME_LIST plist[4];
	memset(plist, 0, sizeof(plist));

	for (ix = 0; ix < 4; ix++) {
		plist[ix].phcode = phonPAUSE;
		plist[ix].ph = phoneme_tab[phonPAUSE];
	}

	plist[1].phcode = phcode;
	plist[1].ph = phoneme_tab[phcode];
	plist[2].sourceix = 1;

	InterpretPhoneme(NULL, 0, &plist[1], plist, phdata, NULL);
}
