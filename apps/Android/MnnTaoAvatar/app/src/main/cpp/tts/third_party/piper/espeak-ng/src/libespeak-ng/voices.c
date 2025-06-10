
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
#include <wctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dirent.h>
#endif

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "voice.h"                    // for voice_t, DoVoiceChange, N_PEAKS
#include "common.h"                    // for GetFileLength, strncpy0
#include "dictionary.h"               // for LoadDictionary
#include "langopts.h"                 // for LoadLanguageOptions
#include "mnemonics.h"               // for LookupMnemName, MNEM_TAB
#include "phoneme.h"                  // for REPLACE_PHONEMES, n_replace_pho...
#include "speech.h"                   // for PATHSEP
#include "mbrola.h"                   // for LoadMbrolaTable
#include "synthdata.h"                // for SelectPhonemeTableName, LookupP...
#include "synthesize.h"               // for SetSpeed, SPEED_FACTORS, speed
#include "translate.h"                // for LANGUAGE_OPTIONS, DeleteTranslator
#include "wavegen.h"                  // for InitBreath

static int AddToVoicesList(const char *fname, int len_path_voices, int is_language_file);


static const MNEM_TAB genders[] = {
	{ "male", ENGENDER_MALE },
	{ "female", ENGENDER_FEMALE },
	{ NULL, ENGENDER_MALE }
};

int tone_points[12] = { 600, 170, 1200, 135, 2000, 110, 3000, 110, -1, 0 };

// limit the rate of change for each formant number
static const int formant_rate_22050[9] = { 240, 170, 170, 170, 170, 170, 170, 170, 170 }; // values for 22kHz sample rate
int formant_rate[9]; // values adjusted for actual sample rate

#define DEFAULT_LANGUAGE_PRIORITY  5
#define N_VOICES_LIST  350
static int n_voices_list = 0;
static espeak_VOICE *voices_list[N_VOICES_LIST];

static espeak_VOICE current_voice_selected;

#define N_VOICE_VARIANTS   12
static const char variants_either[N_VOICE_VARIANTS] = { 1, 2, 12, 3, 13, 4, 14, 5, 11, 0 };
static const char variants_male[N_VOICE_VARIANTS] = { 1, 2, 3, 4, 5, 6, 0 };
static const char variants_female[N_VOICE_VARIANTS] = { 11, 12, 13, 14, 0 };
static const char *const variant_lists[3] = { variants_either, variants_male, variants_female };

static voice_t voicedata;
voice_t *voice = &voicedata;

static char *fgets_strip(char *buf, int size, FILE *f_in)
{
	// strip trailing spaces, and truncate lines at // comment
	int len;
	char *p;

	if (fgets(buf, size, f_in) == NULL)
		return NULL;

	if (buf[0] == '#') {
		buf[0] = 0;
		return buf;
	}

	len = strlen(buf);
	while ((--len > 0) && isspace(buf[len]))
		buf[len] = 0;

	if ((p = strstr(buf, "//")) != NULL)
		*p = 0;

	return buf;
}

static void SetToneAdjust(voice_t *voice, int *tone_pts)
{
	int ix;
	int pt;
	int y;
	int freq1 = 0;
	int height1 = tone_pts[1];

	for (pt = 0; pt < 12; pt += 2) {
		if (tone_pts[pt] == -1) {
			tone_pts[pt] = N_TONE_ADJUST*8;
			if (pt > 0)
				tone_pts[pt+1] = tone_pts[pt-1];
		}
		int freq2;
		int height2;
		
		freq2 = tone_pts[pt] / 8; // 8Hz steps
		height2 = tone_pts[pt+1];
		if ((freq2 - freq1) > 0) {
			for (ix = freq1; ix < freq2; ix++) {
				y = height1 + (int)((ix-freq1) * (height2-height1) / (freq2-freq1));
				if (y > 255)
					y = 255;
				voice->tone_adjust[ix] = y;
			}
		}
		freq1 = freq2;
		height1 = height2;
	}
}

void ReadTonePoints(char *string, int *tone_pts)
{
	// tone_pts[] is int[12]
	int ix;

	for (ix = 0; ix < 12; ix++)
		tone_pts[ix] = -1;

	sscanf(string, "%d %d %d %d %d %d %d %d %d %d",
	       &tone_pts[0], &tone_pts[1], &tone_pts[2], &tone_pts[3],
	       &tone_pts[4], &tone_pts[5], &tone_pts[6], &tone_pts[7],
	       &tone_pts[8], &tone_pts[9]);
}

static espeak_VOICE *ReadVoiceFile(FILE *f_in, const char *fname, int is_language_file)
{
	// Read a Voice file, allocate a VOICE_DATA and set data from the
	// file's  language, gender, name  lines

	char linebuf[120];
	char vname[80];
	char vgender[80];
	char vlanguage[80];
	char languages[300]; // allow space for several alternate language names and priorities

	unsigned int len;
	int langix = 0;
	int n_languages = 0;
	char *p;
	espeak_VOICE *voice_data;
	int priority;
	int age;
	int n_variants = 4; // default, number of variants of this voice before using another voice
	int gender;

	vname[0] = 0;
	vgender[0] = 0;
	age = 0;

	while (fgets_strip(linebuf, sizeof(linebuf), f_in) != NULL) {
		// isolate the attribute name
		for (p = linebuf; (*p != 0) && !iswspace(*p); p++) ;
		*p++ = 0;

		if (linebuf[0] == 0) continue;

		switch (LookupMnem(keyword_tab, linebuf))
		{
		case V_NAME:
			while (isspace(*p)) p++;
			strncpy0(vname, p, sizeof(vname));
			break;
		case V_LANGUAGE:
			priority = DEFAULT_LANGUAGE_PRIORITY;
			vlanguage[0] = 0;

			sscanf(p, "%s %d", vlanguage, &priority);
			len = strlen(vlanguage) + 2;
			// check for space in languages[]
			if (len < (sizeof(languages)-langix-1)) {
				languages[langix] = priority;

				strcpy(&languages[langix+1], vlanguage);
				langix += len;
				n_languages++;
			}
			break;
		case V_GENDER:
			sscanf(p, "%s %d", vgender, &age);
			if (is_language_file)
				fprintf(stderr, "Error (%s): gender attribute specified on a language file\n", fname);
			break;
		case V_VARIANTS:
			sscanf(p, "%d", &n_variants);
		}
	}
	languages[langix++] = 0;

	gender = LookupMnem(genders, vgender);

	if (n_languages == 0)
		return NULL; // no language lines in the voice file

	p = (char *)calloc(sizeof(espeak_VOICE) + langix + strlen(fname) + strlen(vname) + 3, 1);
	voice_data = (espeak_VOICE *)p;
	p = &p[sizeof(espeak_VOICE)];

	memcpy(p, languages, langix);
	voice_data->languages = p;

	strcpy(&p[langix], fname);
	voice_data->identifier = &p[langix];
	voice_data->name = &p[langix];

	if (vname[0] != 0) {
		langix += strlen(fname)+1;
		strcpy(&p[langix], vname);
		voice_data->name = &p[langix];
	}

	voice_data->age = age;
	voice_data->gender = gender;
	voice_data->variant = 0;
	voice_data->xx1 = n_variants;
	return voice_data;
}

void VoiceReset(int tone_only)
{
	// Set voice to the default values

	int pk;
	static const unsigned char default_heights[N_PEAKS] = { 130, 128, 120, 116, 100, 100, 128, 128, 128 }; // changed for v.1.47
	static const unsigned char default_widths[N_PEAKS] = { 140, 128, 128, 160, 171, 171, 128, 128, 128 };

	static const int breath_widths[N_PEAKS] = { 0, 200, 200, 400, 400, 400, 600, 600, 600 };

	// default is:  pitch 80,118
	voice->pitch_base = 0x47000;
	voice->pitch_range = 4104;

	voice->formant_factor = 256;

	voice->speed_percent = 100;
	voice->echo_delay = 0;
	voice->echo_amp = 0;
	voice->flutter = 64;
	voice->n_harmonic_peaks = 5;
	voice->peak_shape = 0;
	voice->voicing = 64;
	voice->consonant_amp = 90; // change from 100 to 90 for v.1.47
	voice->consonant_ampv = 100;
	voice->samplerate = samplerate;
	memset(voice->klattv, 0, sizeof(voice->klattv));

	speed.fast_settings = espeakRATE_MAXIMUM;

	voice->roughness = 2;

	InitBreath();
	for (pk = 0; pk < N_PEAKS; pk++) {
		voice->freq[pk] = 256;
		voice->freq2[pk] = voice->freq[pk];
		voice->height[pk] = default_heights[pk]*2;
		voice->height2[pk] = voice->height[pk];
		voice->width[pk] = default_widths[pk]*2;
		voice->breath[pk] = 0;
		voice->breathw[pk] = breath_widths[pk]; // default breath formant widths
		voice->freqadd[pk] = 0;

		// adjust formant smoothing depending on sample rate
		formant_rate[pk] = (formant_rate_22050[pk] * 22050)/samplerate;
	}

	// This table provides the opportunity for tone control.
	// Adjustment of harmonic amplitudes, steps of 8Hz
	// value of 128 means no change
	SetToneAdjust(voice, tone_points);

	// default values of speed factors
	voice->speedf1 = 256;
	voice->speedf2 = 238;
	voice->speedf3 = 232;

	if (tone_only == 0) {
		n_replace_phonemes = 0;
#if USE_MBROLA
		LoadMbrolaTable(NULL, NULL, 0);
#endif
	}

// probably unnecessary, but removing this would break tests
voice->width[0] = (voice->width[0] * 105)/100;
}

static void VoiceFormant(char *p)
{
	// Set parameters for a formant
	int ix;
	int formant;
	int freq = 100;
	int height = 100;
	int width = 100;
	int freqadd = 0;

	ix = sscanf(p, "%d %d %d %d %d", &formant, &freq, &height, &width, &freqadd);
	if (ix < 2)
		return;

	if ((formant < 0) || (formant > 8))
		return;

	if (freq >= 0) {
		voice->freq[formant] = (int)(freq * 2.56001);
		voice->freq2[formant] = voice->freq[formant];
	}
	if (height >= 0) {
		voice->height[formant] = (int)(height * 2.56001);
		voice->height2[formant] = voice->height[formant];
	}
	if (width >= 0)
		voice->width[formant] = (int)(width * 2.56001);
	voice->freqadd[formant] = freqadd;

	// probably unnecessary, but removing this would break tests
	if (formant == 0)
		voice->width[0] = (voice->width[0] * 105)/100;
}

static void PhonemeReplacement(char *p)
{
	int n;
	int phon;
	int flags = 0;
	char phon_string1[12];
	char phon_string2[12];

	strcpy(phon_string2, "NULL");
	n = sscanf(p, "%d %s %s", &flags, phon_string1, phon_string2);
	if ((n < 2) || (n_replace_phonemes >= N_REPLACE_PHONEMES))
		return;

	if ((phon = LookupPhonemeString(phon_string1)) == 0)
		return; // not recognised

	replace_phonemes[n_replace_phonemes].old_ph = phon;
	replace_phonemes[n_replace_phonemes].new_ph = LookupPhonemeString(phon_string2);
	replace_phonemes[n_replace_phonemes++].type = flags;
}

int Read8Numbers(char *data_in, int data[8])
{
	// Read 8 integer numbers
	memset(data, 0, 8*sizeof(int));
	return sscanf(data_in, "%d %d %d %d %d %d %d %d",
	              &data[0], &data[1], &data[2], &data[3], &data[4], &data[5], &data[6], &data[7]);
}

void ReadNumbers(char *p, int *flags, int maxValue,  const MNEM_TAB *keyword_tab, int key) {
	// read a list of numbers from string p
	// store them as flags in *flags
	// the meaning of the  numbers is bit ordinals, not integer values
	// give an error if number > maxValue is read
	while (*p != 0) {
		while (isspace(*p)) p++;
		int n;
		if ((n = atoi(p)) > 0) {
			p++;
			if (n < maxValue) {
				*flags |= (1 << n);
			} else {
				fprintf(stderr, "%s: Bad option number %d\n", LookupMnemName(keyword_tab, key), n);
			}
		}
	while (isalnum(*p)) p++;
	}
}

voice_t *LoadVoice(const char *vname, int control)
{
	// control, bit 0  1= no_default
	//          bit 1  1 = change tone only, not language
	//          bit 2  1 = don't report error on LoadDictionary
	//          bit 4  1 = vname = full path
        //          bit 8  1 = INTERNAL: compiling phonemes; do not try to
        //                     load the phoneme table
        //          bit 16 1 = UNDOCUMENTED

	FILE *f_voice = NULL;
	char *p;
	int key;
	int ix;
	int value;
	int langix = 0;
	int tone_only = control & 2;
	bool language_set = false;
	bool phonemes_set = false;

	char voicename[40];
	char language_name[40];
	char translator_name[40];
	char new_dictionary[40];
	char phonemes_name[40] = "";
	const char *language_type;
	char buf[sizeof(path_home)+30];
#if USE_MBROLA
	char name1[40];
	char name2[80];
#endif

	int pitch1;
	int pitch2;

	static char voice_identifier[40]; // file name for  current_voice_selected
	static char voice_name[40];       // voice name for current_voice_selected
	static char voice_languages[100]; // list of languages and priorities for current_voice_selected

	if (!tone_only) {
		MAKE_MEM_UNDEFINED(&voice_identifier, sizeof(voice_identifier));
		MAKE_MEM_UNDEFINED(&voice_name, sizeof(voice_name));
		MAKE_MEM_UNDEFINED(&voice_languages, sizeof(voice_languages));
	}

	if ((vname == NULL || vname[0] == 0) && !(control & 8)) {
		return NULL;
	}

	strncpy0(voicename, vname, sizeof(voicename));
	if (control & 0x10) {
		strcpy(buf, vname);
		if (GetFileLength(buf) <= 0)
			return NULL;
	} else {
		if (voicename[0] == 0 && !(control & 8)/*compiling phonemes*/)
			strcpy(voicename, ESPEAKNG_DEFAULT_VOICE);

		char path_voices[sizeof(path_home)+12];
		sprintf(path_voices, "%s%cvoices%c", path_home, PATHSEP, PATHSEP);
		sprintf(buf, "%s%s", path_voices, voicename); // look in the main voices directory

		if (GetFileLength(buf) <= 0) {
			sprintf(path_voices, "%s%clang%c", path_home, PATHSEP, PATHSEP);
			sprintf(buf, "%s%s", path_voices, voicename); // look in the main languages directory
		}
	}

	f_voice = fopen(buf, "r");

        if (!(control & 8)/*compiling phonemes*/)
            language_type = ESPEAKNG_DEFAULT_VOICE; // default
        else
            language_type = "";

	if (f_voice == NULL) {
		if (control & 3)
			return NULL; // can't open file

		if (SelectPhonemeTableName(voicename) >= 0)
			language_type = voicename;
	}

	if (!tone_only && (translator != NULL)) {
		DeleteTranslator(translator);
		translator = NULL;
	}

	strcpy(translator_name, language_type);
	strcpy(new_dictionary, language_type);

	if (!tone_only) {
		voice = &voicedata;
		strncpy0(voice_identifier, vname, sizeof(voice_identifier));
		voice_name[0] = 0;
		voice_languages[0] = 0;

		current_voice_selected.identifier = voice_identifier;
		current_voice_selected.name = voice_name;
		current_voice_selected.languages = voice_languages;
	} else {
		// append the variant file name to the voice identifier
		if ((p = strchr(voice_identifier, '+')) != NULL)
			*p = 0;    // remove previous variant name
		sprintf(buf, "+%s", &vname[3]);    // omit  !v/  from the variant filename
		strcat(voice_identifier, buf);
	}
	VoiceReset(tone_only);

	while ((f_voice != NULL) && (fgets_strip(buf, sizeof(buf), f_voice) != NULL)) {
		// isolate the attribute name
		for (p = buf; (*p != 0) && !isspace(*p); p++) ;
		*p++ = 0;

		if (buf[0] == 0) continue;

		key = LookupMnem(langopts_tab, buf);

        if (key != 0) {
            LoadLanguageOptions(translator, key, p);
        } else {
            key = LookupMnem(keyword_tab, buf);
            switch (key)
            {
            case V_LANGUAGE:
            {
                unsigned int len;
                int priority;

                if (tone_only)
                    break;

                priority = DEFAULT_LANGUAGE_PRIORITY;
                language_name[0] = 0;

                sscanf(p, "%s %d", language_name, &priority);
                if (strcmp(language_name, "variant") == 0)
                    break;

                len = strlen(language_name) + 2;
                // check for space in languages[]
                if (len < (sizeof(voice_languages)-langix-1)) {
                    voice_languages[langix] = priority;

                    strcpy(&voice_languages[langix+1], language_name);
                    langix += len;
                }

                // only act on the first language line
                if (language_set == false) {
                    language_type = strtok(language_name, "-");
                    language_set = true;
                    strcpy(translator_name, language_type);
                    strcpy(new_dictionary, language_type);
                    strcpy(phonemes_name, language_type);
                    SelectPhonemeTableName(phonemes_name);

                    translator = SelectTranslator(translator_name);
                    strncpy0(voice->language_name, language_name, sizeof(voice->language_name));
                }
            }
                break;
            case V_NAME:
                if (tone_only == 0) {
                    while (isspace(*p)) p++;
                    strncpy0(voice_name, p, sizeof(voice_name));
                }
                break;
            case V_GENDER:
            {
                int age = 0;
                char vgender[80];
                sscanf(p, "%s %d", vgender, &age);
                current_voice_selected.gender = LookupMnem(genders, vgender);
                current_voice_selected.age = age;
            }
                break;
            case V_DICTIONARY: // dictionary
                sscanf(p, "%s", new_dictionary);
                break;
            case V_PHONEMES: // phoneme table
                sscanf(p, "%s", phonemes_name);
                break;
            case V_FORMANT:
                VoiceFormant(p);
                break;
            case V_PITCH:
                // default is  pitch 82 118
                if (sscanf(p, "%d %d", &pitch1, &pitch2) == 2) {
                    voice->pitch_base = (pitch1 - 9) << 12;
                    voice->pitch_range = (pitch2 - pitch1) * 108;
                    double factor = (double)(pitch1 - 82)/82;
                    voice->formant_factor = (int)((1+factor/4) * 256); // nominal formant shift for a different voice pitch
                }
                break;





            case V_REPLACE:
                if (phonemes_set == false) {
                    // must set up a phoneme table before we can lookup phoneme mnemonics
                    SelectPhonemeTableName(phonemes_name);
                    phonemes_set = true;
                }
                PhonemeReplacement(p);
                break;

            case V_ECHO:
                // echo.  suggest: 135mS  11%
                value = 0;
                voice->echo_amp = 0;
                sscanf(p, "%d %d", &voice->echo_delay, &voice->echo_amp);
                break;
            case V_FLUTTER: // flutter
                if (sscanf(p, "%d", &value) == 1)
                    voice->flutter = value * 32;
                break;
            case V_ROUGHNESS: // roughness
                if (sscanf(p, "%d", &value) == 1)
                    voice->roughness = value;
                break;
            case V_CLARITY: // formantshape
                if (sscanf(p, "%d", &value) == 1) {
                    if (value > 4) {
                        voice->peak_shape = 1; // squarer formant peaks
                        value = 4;
                    }
                    voice->n_harmonic_peaks = 1+value;
                }
                break;
            case V_TONE:
            {
                int tone_data[12];
                ReadTonePoints(p, tone_data);
                SetToneAdjust(voice, tone_data);
            }
                break;
            case V_VOICING:
                if (sscanf(p, "%d", &value) == 1)
                    voice->voicing = (value * 64)/100;
                break;
            case V_BREATH:
                voice->breath[0] = Read8Numbers(p, &voice->breath[1]);
                for (ix = 1; ix < 8; ix++) {
                    if (ix % 2)
                        voice->breath[ix] = -voice->breath[ix];
                }
                break;
            case V_BREATHW:
                voice->breathw[0] = Read8Numbers(p, &voice->breathw[1]);
                break;
            case V_CONSONANTS:
                value = sscanf(p, "%d %d", &voice->consonant_amp, &voice->consonant_ampv);
                break;
            case V_SPEED:
                sscanf(p, "%d", &voice->speed_percent);
                SetSpeed(3);
                break;
#if USE_MBROLA
            case V_MBROLA:
            {
                int srate = 16000;

                name2[0] = 0;
                sscanf(p, "%s %s %d", name1, name2, &srate);
                espeak_ng_STATUS status = LoadMbrolaTable(name1, name2, &srate);
                if (status != ENS_OK) {
                    espeak_ng_PrintStatusCodeMessage(status, stderr, NULL);
                    fclose(f_voice);
                    return NULL;
                }
                else
                    voice->samplerate = srate;
            }
                break;
#else
            case V_MBROLA:
                fprintf(stderr, "espeak-ng was built without mbrola support\n");
                break;
#endif
#if USE_KLATT
            case V_KLATT:
                voice->klattv[0] = 1; // default source: IMPULSIVE
                Read8Numbers(p, voice->klattv);
                voice->klattv[KLATT_Kopen] -= 40;
                break;
#else
            case V_KLATT:
                fprintf(stderr, "espeak-ng was built without klatt support\n");
                break;
#endif
            case V_FAST:
                sscanf(p, "%d", &speed.fast_settings);
                SetSpeed(3);
                break;

            case V_MAINTAINER:
            case V_STATUS:
                break;
            default:
                fprintf(stderr, "Bad voice attribute: %s\n", buf);
                break;
            }
        }
	}
	if (f_voice != NULL)
		fclose(f_voice);

	if ((translator == NULL) && (!tone_only)) {
		// not set by language attribute
		translator = SelectTranslator(translator_name);
	}

	if (!tone_only) {
		if (!!(control & 8/*compiling phonemes*/)) {
			/* Set by espeak_ng_CompilePhonemeDataPath when it
				* calls LoadVoice("", 8) to set up a dummy(?) voice.
				* As phontab may not yet exist this avoids the spurious
				* error message and guarantees consistent results by
				* not actually reading a potentially bogus phontab...
				*/
			ix = 0;
		} else if ((ix = SelectPhonemeTableName(phonemes_name)) < 0) {
			fprintf(stderr, "Unknown phoneme table: '%s'\n", phonemes_name);
			ix = 0;
		}

		voice->phoneme_tab_ix = ix;
		translator->phoneme_tab_ix = ix;

		if (!(control & 8/*compiling phonemes*/)) {
			LoadDictionary(translator, new_dictionary, control & 4);
			if (dictionary_name[0] == 0) {
				DeleteTranslator(translator);
				return NULL; // no dictionary loaded
			}
		}

		/* Terminate languages list with a zero-priority entry */
		voice_languages[langix] = 0;
	}

	return voice;
}

static char *ExtractVoiceVariantName(char *vname, int variant_num, int add_dir)
{
	// Remove any voice variant suffix (name or number) from a voice name
	// Returns the voice variant name

	static char variant_name[40];
	char variant_prefix[5];

	MAKE_MEM_UNDEFINED(&variant_name, sizeof(variant_name));
	variant_name[0] = 0;
	sprintf(variant_prefix, "!v%c", PATHSEP);
	if (add_dir == 0)
		variant_prefix[0] = 0;

	if (vname != NULL) {
		char *p;
		if ((p = strchr(vname, '+')) != NULL) {
			// The voice name has a +variant suffix
			variant_num = 0;
			*p++ = 0; // delete the suffix from the voice name
			if (IsDigit09(*p))
				variant_num = atoi(p); // variant number
			else {
				// voice variant name, not number
				sprintf(variant_name, "%s%s", variant_prefix, p);
			}
		}
	}

	if (variant_num > 0) {
		if (variant_num < 10)
			sprintf(variant_name, "%sm%d", variant_prefix, variant_num); // male
		else
			sprintf(variant_name, "%sf%d", variant_prefix, variant_num-10); // female
	}

	return variant_name;
}

voice_t *LoadVoiceVariant(const char *vname, int variant_num)
{
	// Load a voice file.
	// Also apply a voice variant if specified by "variant", or by "+number" or "+name" in the "vname"

	voice_t *v;
	char *variant_name;
	char buf[60];

	strncpy0(buf, vname, sizeof(buf));
	variant_name = ExtractVoiceVariantName(buf, variant_num, 1);

	if ((v = LoadVoice(buf, 0)) == NULL)
		return NULL;

	if (variant_name[0] != 0)
		v = LoadVoice(variant_name, 2);
	return v;
}

static int __cdecl VoiceNameSorter(const void *p1, const void *p2)
{
	int ix;
	espeak_VOICE *v1 = *(espeak_VOICE **)p1;
	espeak_VOICE *v2 = *(espeak_VOICE **)p2;

	if ((ix = strcmp(&v1->languages[1], &v2->languages[1])) != 0) // primary language name
		return ix;
	if ((ix = v1->languages[0] - v2->languages[0]) != 0) // priority number
		return ix;
	return strcmp(v1->name, v2->name);
}

static int __cdecl VoiceScoreSorter(const void *p1, const void *p2)
{
	int ix;
	espeak_VOICE *v1 = *(espeak_VOICE **)p1;
	espeak_VOICE *v2 = *(espeak_VOICE **)p2;

	if ((ix = v2->score - v1->score) != 0)
		return ix;
	return strcmp(v1->name, v2->name);
}

static int ScoreVoice(espeak_VOICE *voice_spec, const char *spec_language, int spec_n_parts, int spec_lang_len, espeak_VOICE *voice)
{
	const char *p;
	int score = 0;
	int x;

	p = voice->languages; // list of languages+dialects for which this voice is suitable

	if (spec_n_parts < 0) {
		// match on the subdirectory
		if (memcmp(voice->identifier, spec_language, spec_lang_len) == 0)
			return 100;
		return 0;
	}

	if (spec_n_parts == 0)
		score = 100;
	else {
		if ((*p == 0) && (strcmp(spec_language, "variants") == 0)) {
			// match on a voice with no languages if the required language is "variants"
			score = 100;
		}

		// compare the required language with each of the languages of this voice
		while (*p != 0) {
			int language_priority = *p++;


			int n_parts = 1;
			int matching = 1;
			int matching_parts = 0;
			int ix;
			for (ix = 0;; ix++) {
				int c1, c2;
				if ((ix >= spec_lang_len) || ((c1 = spec_language[ix]) == '-'))
					c1 = 0;
				if ((c2 = p[ix]) == '-')
					c2 = 0;

				if (c1 != c2)
					matching = 0;

				if (p[ix] == '-') {
					n_parts++;
					if (matching)
						matching_parts++;
				}
				if (p[ix] == 0)
					break;
			}
			p += (ix+1);
			matching_parts += matching; // number of parts which match

			if (matching_parts == 0)
				continue; // no matching parts for this language

			x = 5;
			// reduce the score if not all parts of the required language match
			int diff;
			if ((diff = (spec_n_parts - matching_parts)) > 0)
				x -= diff;

			// reduce score if the language is more specific than required
			if ((diff = (n_parts - matching_parts)) > 0)
				x -= diff;

			x = x*100 - (language_priority * 2);

			if (x > score)
				score = x;
		}
	}
	if (score == 0)
		return 0;

	if (voice_spec->name != NULL) {
		if (strcmp(voice_spec->name, voice->name) == 0) {
			// match on voice name
			score += 500;
		} else if (strcmp(voice_spec->name, voice->identifier) == 0)
			score += 400;
	}

	if (((voice_spec->gender == ENGENDER_MALE) || (voice_spec->gender == ENGENDER_FEMALE)) &&
	    ((voice->gender == ENGENDER_MALE) || (voice->gender == ENGENDER_FEMALE))) {
		if (voice_spec->gender == voice->gender)
			score += 50;
		else
			score -= 50;
	}

	if ((voice_spec->age <= 12) && (voice->gender == ENGENDER_FEMALE) && (voice->age > 12))
		score += 5; // give some preference for non-child female voice if a child is requested

	if (voice->age != 0) {
		int required_age;
		if (voice_spec->age == 0)
			required_age = 30;
		else
			required_age = voice_spec->age;

		int ratio;
		ratio = (required_age*100)/voice->age;
		if (ratio < 100)
			ratio = 10000/ratio;
		ratio = (ratio - 100)/10; // 0=exact match, 10=out by factor of 2
		x = 5 - ratio;
		if (x > 0) x = 0;

		score = score + x;

		if (voice_spec->age > 0)
			score += 10; // required age specified, favour voices with a specified age (near it)
	}
	if (score < 1)
		score = 1;
	return score;
}

static int SetVoiceScores(espeak_VOICE *voice_select, espeak_VOICE **voices, int control)
{
	// control: bit0=1  include mbrola voices
	int ix;
	int score;
	int nv; // number of candidates
	int n_parts = 0;
	int lang_len = 0;
	espeak_VOICE *vp;
	char language[80];

	// count number of parts in the specified language
	if ((voice_select->languages != NULL) && (voice_select->languages[0] != 0)) {
		n_parts = 1;
		lang_len = strlen(voice_select->languages);
		for (ix = 0; (ix <= lang_len) && ((unsigned)ix < sizeof(language)); ix++) {
			if ((language[ix] = tolower(voice_select->languages[ix])) == '-')
				n_parts++;
		}
	}

	if ((n_parts == 1) && (control & 1)) {
		if (strcmp(language, "mbrola") == 0) {
			language[2] = 0; // truncate to "mb"
			lang_len = 2;
		}

		char buf[sizeof(path_home)+80];
		sprintf(buf, "%s/voices/%s", path_home, language);
		if (GetFileLength(buf) == -EISDIR) {
			// A subdirectory name has been specified.  List all the voices in that subdirectory
			language[lang_len++] = PATHSEP;
			language[lang_len] = 0;
			n_parts = -1;
		}
	}

	// select those voices which match the specified language
	nv = 0;
	for (ix = 0; ix < n_voices_list; ix++) {
		vp = voices_list[ix];

		if (((control & 1) == 0) && (memcmp(vp->identifier, "mb/", 3) == 0))
			continue;

		if (voice_select->languages == NULL || memcmp(voice_select->languages,"all", 3) == 0) {
			voices[nv++] = vp;
			continue;
		}

		if ((score = ScoreVoice(voice_select, language, n_parts, lang_len, voices_list[ix])) > 0) {
			voices[nv++] = vp;
			vp->score = score;
		}
	}
	voices[nv] = NULL; // list terminator

	if (nv == 0)
		return 0;

	// sort the selected voices by their score
	qsort(voices, nv, sizeof(espeak_VOICE *), (int(__cdecl *)(const void *, const void *))VoiceScoreSorter);

	return nv;
}

espeak_VOICE *SelectVoiceByName(espeak_VOICE **voices, const char *name2)
{
	int ix;
	int match_fname = -1;
	int match_fname2 = -1;
	int match_name = -1;
	const char *id; // this is the filename within espeak-ng-data/voices
	int last_part_len;
	char last_part[41];
	char name[40];

	if (voices == NULL) {
		if (n_voices_list == 0)
			espeak_ListVoices(NULL); // create the voices list
		voices = voices_list;
	}

	strncpy0(name, name2, sizeof(name));

	sprintf(last_part, "%c%s", PATHSEP, name);
	last_part_len = strlen(last_part);

	for (ix = 0; voices[ix] != NULL; ix++) {
		if (strcasecmp(name, voices[ix]->name) == 0) {
			match_name = ix; // found matching voice name
			break;
		} else {
			id = voices[ix]->identifier;
			if (strcasecmp(name, id) == 0)
				match_fname = ix; // matching identifier, use this if no matching name
			else if (strcasecmp(last_part, &id[strlen(id)-last_part_len]) == 0)
				match_fname2 = ix;
		}
	}

	if (match_name < 0) {
		match_name = match_fname; // no matching name, try matching filename
		if (match_name < 0)
			match_name = match_fname2; // try matching just the last part of the filename
	}

	if (match_name < 0)
		return NULL;

	return voices[match_name];
}

char const *SelectVoice(espeak_VOICE *voice_select, int *found)
{
	// Returns a path within espeak-voices, with a possible +variant suffix
	// variant is an output-only parameter

	int nv; // number of candidates
	int ix, ix2;
	int j;
	int n_variants;
	int variant_number;
	int gender;
	int aged = 1;
	char *variant_name;
	const char *p, *p_start;
	espeak_VOICE *vp = NULL;
	espeak_VOICE *vp2;
	espeak_VOICE voice_select2;
	espeak_VOICE *voices[N_VOICES_LIST]; // list of candidates
	espeak_VOICE *voices2[N_VOICES_LIST+N_VOICE_VARIANTS];
	static espeak_VOICE voice_variants[N_VOICE_VARIANTS];
	static char voice_id[50];

	MAKE_MEM_UNDEFINED(&voice_variants, sizeof(voice_variants));
	MAKE_MEM_UNDEFINED(&voice_id, sizeof(voice_id));

	*found = 1;
	memcpy(&voice_select2, voice_select, sizeof(voice_select2));

	if (n_voices_list == 0)
		espeak_ListVoices(NULL); // create the voices list

	if ((voice_select2.languages == NULL) || (voice_select2.languages[0] == 0)) {
		// no language is specified. Get language from the named voice
		char buf[60];

		if (voice_select2.name == NULL) {
			if ((voice_select2.name = voice_select2.identifier) == NULL)
				voice_select2.name = ESPEAKNG_DEFAULT_VOICE;
		}

		strncpy0(buf, voice_select2.name, sizeof(buf));
		variant_name = ExtractVoiceVariantName(buf, 0, 0);

		vp = SelectVoiceByName(voices_list, buf);
		if (vp != NULL) {
			voice_select2.languages = &(vp->languages[1]);

			if ((voice_select2.gender == ENGENDER_UNKNOWN) && (voice_select2.age == 0) && (voice_select2.variant == 0)) {
				if (variant_name[0] != 0) {
					sprintf(voice_id, "%s+%s", vp->identifier, variant_name);
					return voice_id;
				}

				return vp->identifier;
			}
		}
	}

	// select and sort voices for the required language
	nv = SetVoiceScores(&voice_select2, voices,
			voice_select2.identifier && strncmp(voice_select2.identifier, "mb/", 3) == 0 ? 1 : 0);

	if (nv == 0) {
		// no matching voice, choose the default
		*found = 0;
		if ((voices[0] = SelectVoiceByName(voices_list, ESPEAKNG_DEFAULT_VOICE)) != NULL)
			nv = 1;
	}

	gender = 0;
	if ((voice_select2.gender == ENGENDER_FEMALE) || ((voice_select2.age > 0) && (voice_select2.age < 13)))
		gender = ENGENDER_FEMALE;
	else if (voice_select2.gender == ENGENDER_MALE)
		gender = ENGENDER_MALE;

	#define AGE_OLD 60
	if (voice_select2.age < AGE_OLD)
		aged = 0;

	p = p_start = variant_lists[gender];
	if (aged == 0)
		p++; // the first voice in the variants list is older

	// add variants for the top voices
	n_variants = 0;
	for (ix = 0, ix2 = 0; ix < nv; ix++) {
		vp = voices[ix];
		// is the main voice the required gender?
		bool skip = false;

		if ((gender != ENGENDER_UNKNOWN) && (vp->gender != gender))
			skip = true;
		if ((ix2 == 0) && aged && (vp->age < AGE_OLD))
			skip = true;

		if (skip == false)
			voices2[ix2++] = vp;

		for (j = 0; (j < vp->xx1) && (n_variants < N_VOICE_VARIANTS);) {
			if ((variant_number = *p) == 0) {
				p = p_start;
				continue;
			}

			vp2 = &voice_variants[n_variants++]; // allocate space for voice variant
			memcpy(vp2, vp, sizeof(espeak_VOICE)); // copy from the original voice
			vp2->variant = variant_number;
			voices2[ix2++] = vp2;
			p++;
			j++;
		}
	}
	// add any more variants to the end of the list
	while ((vp != NULL) && ((variant_number = *p++) != 0) && (n_variants < N_VOICE_VARIANTS)) {
		vp2 = &voice_variants[n_variants++]; // allocate space for voice variant
		memcpy(vp2, vp, sizeof(espeak_VOICE)); // copy from the original voice
		vp2->variant = variant_number;
		voices2[ix2++] = vp2;
	}

	// index the sorted list by the required variant number
	if (ix2 == 0)
		return NULL;
	vp = voices2[voice_select2.variant % ix2];

	if (vp->variant != 0) {
		variant_name = ExtractVoiceVariantName(NULL, vp->variant, 0);
		sprintf(voice_id, "%s+%s", vp->identifier, variant_name);
		return voice_id;
	}

	return vp->identifier;
}

static void GetVoices(const char *path, int len_path_voices, int is_language_file)
{
	char fname[sizeof(path_home)+100];

#if PLATFORM_WINDOWS
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	#undef UNICODE // we need FindFirstFileA() which takes an 8-bit c-string
	sprintf(fname, "%s\\*", path);
	hFind = FindFirstFileA(fname, &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE)
		return;

	do {
		if (n_voices_list >= (N_VOICES_LIST-2)) {
			fprintf(stderr, "Warning: maximum number %d of (N_VOICES_LIST = %d - 1) reached\n", n_voices_list + 1, N_VOICES_LIST);
			break; // voices list is full
		}

		if (FindFileData.cFileName[0] != '.') {
			sprintf(fname, "%s%c%s", path, PATHSEP, FindFileData.cFileName);
			if (AddToVoicesList(fname, len_path_voices, is_language_file) != 0) {
				continue;
			}
		}
	} while (FindNextFileA(hFind, &FindFileData) != 0);
	FindClose(hFind);
#else
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir((char *)path)) == NULL) // note: (char *) is needed for WINCE
		return;

	while ((ent = readdir(dir)) != NULL) {
		if (n_voices_list >= (N_VOICES_LIST-2)) {
			fprintf(stderr, "Warning: maximum number %d of (N_VOICES_LIST = %d - 1) reached\n", n_voices_list + 1, N_VOICES_LIST);
			break; // voices list is full
		}

		if (ent->d_name[0] == '.')
			continue;

			 sprintf(fname, "%s%c%s", path, PATHSEP, ent->d_name);
			if (AddToVoicesList(fname, len_path_voices, is_language_file) != 0) {
				continue;
			}

	}
	closedir(dir);
#endif
}

#pragma GCC visibility push(default)

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SetVoiceByFile(const char *filename)
{
	int ix;
	espeak_VOICE voice_selector;
	char *variant_name;
	char buf[60];

	strncpy0(buf, filename, sizeof(buf));

	variant_name = ExtractVoiceVariantName(buf, 0, 1);

	for (ix = 0;; ix++) {
		// convert voice name to lower case  (ascii)
		if ((buf[ix] = tolower(buf[ix])) == 0)
			break;
	}

	memset(&voice_selector, 0, sizeof(voice_selector));
	voice_selector.name = (char *)filename; // include variant name in voice stack ??

	// first check for a voice with this filename
	// This may avoid the need to call espeak_ListVoices().

	if (LoadVoice(buf, 0x10) != NULL) {
		if (variant_name[0] != 0)
			LoadVoice(variant_name, 2);

		DoVoiceChange(voice);
		voice_selector.languages = voice->language_name;
		SetVoiceStack(&voice_selector, variant_name);
		return ENS_OK;
	}

	return ENS_VOICE_NOT_FOUND;
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SetVoiceByName(const char *name)
{
	espeak_VOICE *v;
	int ix;
	espeak_VOICE voice_selector;
	char *variant_name;
	char buf[60];

	strncpy0(buf, name, sizeof(buf));

	variant_name = ExtractVoiceVariantName(buf, 0, 1);

	for (ix = 0;; ix++) {
		// convert voice name to lower case  (ascii)
		if ((buf[ix] = tolower(buf[ix])) == 0)
			break;
	}

	memset(&voice_selector, 0, sizeof(voice_selector));
	voice_selector.name = (char *)name; // include variant name in voice stack ??

	// first check for a voice with this filename
	// This may avoid the need to call espeak_ListVoices().

	if (LoadVoice(buf, 1) != NULL) {
		if (variant_name[0] != 0)
			LoadVoice(variant_name, 2);

		DoVoiceChange(voice);
		voice_selector.languages = voice->language_name;
		SetVoiceStack(&voice_selector, variant_name);
		return ENS_OK;
	}

	if (n_voices_list == 0)
		espeak_ListVoices(NULL); // create the voices list

	if ((v = SelectVoiceByName(voices_list, buf)) != NULL) {
		if (LoadVoice(v->identifier, 0) != NULL) {
			if (variant_name[0] != 0)
				LoadVoice(variant_name, 2);
			DoVoiceChange(voice);
			voice_selector.languages = voice->language_name;
			SetVoiceStack(&voice_selector, variant_name);
			return ENS_OK;
		}
	}
	return ENS_VOICE_NOT_FOUND;
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SetVoiceByProperties(espeak_VOICE *voice_selector)
{
	const char *voice_id;
	int voice_found;

	voice_id = SelectVoice(voice_selector, &voice_found);
	if (voice_found == 0)
		return ENS_VOICE_NOT_FOUND;

	LoadVoiceVariant(voice_id, 0);
	DoVoiceChange(voice);
	SetVoiceStack(voice_selector, "");

	return ENS_OK;
}

#pragma GCC visibility pop

void FreeVoiceList(void)
{
	int ix;
	for (ix = 0; ix < n_voices_list; ix++) {
		if (voices_list[ix] != NULL) {
			free(voices_list[ix]);
			voices_list[ix] = NULL;
		}
	}
	n_voices_list = 0;
}

#pragma GCC visibility push(default)

ESPEAK_API const espeak_VOICE **espeak_ListVoices(espeak_VOICE *voice_spec)
{
	char path_voices[sizeof(path_home)+12];

	espeak_VOICE *v;
	static espeak_VOICE **voices = NULL;

	// free previous voice list data
	FreeVoiceList();

	sprintf(path_voices, "%s%cvoices", path_home, PATHSEP);
	GetVoices(path_voices, strlen(path_voices)+1, 0);

	sprintf(path_voices, "%s%clang", path_home, PATHSEP);
	GetVoices(path_voices, strlen(path_voices)+1, 1);

	voices_list[n_voices_list] = NULL; // voices list terminator
	espeak_VOICE **new_voices = (espeak_VOICE **)realloc(voices, sizeof(espeak_VOICE *)*(n_voices_list+1));
	if (new_voices == NULL)
		return (const espeak_VOICE **)voices;
	voices = new_voices;

	// sort the voices list
	qsort(voices_list, n_voices_list, sizeof(espeak_VOICE *),
	      (int(__cdecl *)(const void *, const void *))VoiceNameSorter);

	if (voice_spec) {
		// select the voices which match the voice_spec, and sort them by preference
		SetVoiceScores(voice_spec, voices, 1);
	} else {
		// list all: omit variant and mbrola voices
		int ix;
		int j;

		j = 0;
		for (ix = 0; (v = voices_list[ix]) != NULL; ix++) {
			if ((v->languages[0] != 0) && (strcmp(&v->languages[1], "variant") != 0)
			    && (memcmp(v->identifier, "mb/", 3) != 0))
				voices[j++] = v;
		}
		voices[j] = NULL;
	}
	return (const espeak_VOICE **)voices;
}

ESPEAK_API espeak_VOICE *espeak_GetCurrentVoice(void)
{
	return &current_voice_selected;
}

#pragma GCC visibility pop

static int AddToVoicesList(const char *fname, int len_path_voices, int is_language_file) {
	int ftype = GetFileLength(fname);

	if (ftype == -EISDIR) {
		// a sub-directory
		GetVoices(fname, len_path_voices, is_language_file);
	} else if (ftype > 0) {
		// a regular file, add it to the voices list
		FILE *f_voice;
		if ((f_voice = fopen(fname, "r")) == NULL)
			return 1;

		// pass voice file name within the voices directory
		espeak_VOICE *voice_data;
		voice_data = ReadVoiceFile(f_voice, fname+len_path_voices, is_language_file);
		fclose(f_voice);

		if (voice_data != NULL)
			voices_list[n_voices_list++] = voice_data;
	}
	return 0;
}
