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

#ifndef ESPEAK_NG_SYNTHESIZE_H
#define ESPEAK_NG_SYNTHESIZE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdbool.h>
#include <espeak-ng/espeak_ng.h>
#include "phoneme.h"              // for PHONEME_TAB, N_PHONEME_TAB

#define espeakINITIALIZE_PHONEME_IPA 0x0002 // move this to speak_lib.h, after eSpeak version 1.46.02

#define N_PHONEME_LIST 1000 // enough for source[N_TR_SOURCE] full of text, else it will truncate

#define N_SEQ_FRAMES  25 // max frames in a spectrum sequence (real max is ablut 8)
#define STEPSIZE      64 // 2.9mS at 22 kHz sample rate

// flags set for frames within a spectrum sequence
#define FRFLAG_KLATT           0x01 // this frame includes extra data for Klatt synthesizer
#define FRFLAG_VOWEL_CENTRE    0x02 // centre point of vowel
#define FRFLAG_LEN_MOD         0x04 // reduce effect of length adjustment
#define FRFLAG_BREAK_LF        0x08 // but keep f3 upwards
#define FRFLAG_BREAK           0x10 // don't merge with next frame
#define FRFLAG_FORMANT_RATE    0x20 // Flag5 allow increased rate of change of formant freq
#define FRFLAG_MODULATE        0x40 // Flag6 modulate amplitude of some cycles to give trill
#define FRFLAG_DEFER_WAV       0x80 // Flag7 defer mixing WAV until the next frame
#define FRFLAG_LEN_MOD2      0x4000 // reduce effect of length adjustment, used for the start of a vowel
#define FRFLAG_COPIED        0x8000 // This frame has been copied into temporary rw memory

#define SFLAG_SEQCONTINUE      0x01 // a liquid or nasal after a vowel, but not followed by a vowel
#define SFLAG_EMBEDDED         0x02 // there are embedded commands before this phoneme
#define SFLAG_SYLLABLE         0x04 // vowel or syllabic consonant
#define SFLAG_LENGTHEN         0x08 // lengthen symbol : included after this phoneme
#define SFLAG_DICTIONARY       0x10 // the pronunciation of this word was listed in the xx_list dictionary
#define SFLAG_SWITCHED_LANG    0x20 // this word uses phonemes from a different language
#define SFLAG_PROMOTE_STRESS   0x40 // this unstressed word can be promoted to stressed

#define SFLAG_NEXT_PAUSE     0x2000 // consider next phoneme as pause

// embedded command numbers
#define EMBED_P     1 // pitch
#define EMBED_S     2 // speed (used in setlengths)
#define EMBED_A     3 // amplitude/volume
#define EMBED_R     4 // pitch range/expression
#define EMBED_H     5 // echo/reverberation
#define EMBED_T     6 // different tone for announcing punctuation (not used)
#define EMBED_I     7 // sound icon
#define EMBED_S2    8 // speed (used in synthesize)
#define EMBED_Y     9 // say-as commands
#define EMBED_M    10 // mark name
#define EMBED_U    11 // audio uri
#define EMBED_B    12 // break
#define EMBED_F    13 // emphasis

#define N_EMBEDDED_VALUES    15
extern int embedded_value[N_EMBEDDED_VALUES];
extern const int embedded_default[N_EMBEDDED_VALUES];

#define N_KLATTP   10 // this affects the phoneme data file format
#define N_KLATTP2  14 // used in vowel files, with extra parameters for future extensions

#define KLATT_AV      0
#define KLATT_FNZ     1 // nasal zero freq
#define KLATT_Tilt    2
#define KLATT_Aspr    3
#define KLATT_Skew    4

#define KLATT_Kopen   5
#define KLATT_AVp     6
#define KLATT_Fric    7
#define KLATT_FricBP  8
#define KLATT_Turb    9

typedef struct { // 64 bytes
	short frflags;
	short ffreq[7];
	unsigned char length;
	unsigned char rms;
	unsigned char fheight[8];
	unsigned char fwidth[6];   // width/4  f0-5
	unsigned char fright[3];   // width/4  f0-2
	unsigned char bw[4];       // Klatt bandwidth BNZ /2, f1,f2,f3
	unsigned char klattp[5];   // AV, FNZ, Tilt, Aspr, Skew
	unsigned char klattp2[5];  // continuation of klattp[],  Avp, Fric, FricBP, Turb
	unsigned char klatt_ap[7]; // Klatt parallel amplitude
	unsigned char klatt_bp[7]; // Klatt parallel bandwidth  /2
	unsigned char spare;       // pad to multiple of 4 bytes
} frame_t; // with extra Klatt parameters for parallel resonators

typedef struct { // 44 bytes
	short frflags;
	short ffreq[7];
	unsigned char length;
	unsigned char rms;
	unsigned char fheight[8];
	unsigned char fwidth[6];  // width/4  f0-5
	unsigned char fright[3];  // width/4  f0-2
	unsigned char bw[4];      // Klatt bandwidth BNZ /2, f1,f2,f3
	unsigned char klattp[5];  // AV, FNZ, Tilt, Aspr, Skew
} frame_t2; // without the extra Klatt parameters

typedef struct {
	const unsigned char *pitch_env;
	int pitch;      // pitch Hz*256
	int pitch_ix;   // index into pitch envelope (*256)
	int pitch_inc;  // increment to pitch_ix
	int pitch_base; // Hz*256 low, before modified by envelope
	int pitch_range; // Hz*256 range of envelope

	unsigned char *mix_wavefile; // wave file to be added to synthesis
	int n_mix_wavefile; // length in bytes
	int mix_wave_scale; // 0=2 byte samples
	int mix_wave_amp;
	int mix_wavefile_ix;
	int mix_wavefile_max; // length of available WAV data (in bytes)
	int mix_wavefile_offset;

	int amplitude;
	int amplitude_v;
	int amplitude_fmt; // percentage amplitude adjustment for formant synthesis
} WGEN_DATA;

typedef struct {
	double a;
	double b;
	double c;
	double x1;
	double x2;
} RESONATOR;

typedef struct {
	short length_total; // not used
	unsigned char n_frames;
	unsigned char sqflags;
	frame_t2 frame[N_SEQ_FRAMES]; // max. frames in a spectrum sequence
} SPECT_SEQ; // sequence of espeak formant frames

typedef struct {
	short length_total; // not used
	unsigned char n_frames;
	unsigned char sqflags;
	frame_t frame[N_SEQ_FRAMES]; // max. frames in a spectrum sequence
} SPECT_SEQK; // sequence of klatt formants frames

typedef struct {
	short length;
	short frflags;
	frame_t *frame;
} frameref_t;

// a clause translated into phoneme codes (first stage)
typedef struct {
	unsigned short synthflags; // NOTE Put shorts on 32bit boundaries, because of RISC OS compiler bug?
	unsigned char phcode;
	unsigned char stresslevel;
	unsigned short sourceix;  // ix into the original source text string, only set at the start of a word
	unsigned char wordstress; // the highest level stress in this word
	unsigned char tone_ph;    // tone phoneme to use with this vowel
} PHONEME_LIST2;

#define PHLIST_START_OF_WORD     1
#define PHLIST_END_OF_CLAUSE     2
#define PHLIST_START_OF_SENTENCE 4
#define PHLIST_START_OF_CLAUSE   8

typedef struct {
	// The first section is a copy of PHONEME_LIST2
	unsigned short synthflags;
	unsigned char phcode;
	unsigned char stresslevel;
	unsigned short sourceix;  // ix into the original source text string, only set at the start of a word
	unsigned char wordstress; // the highest level stress in this word
	unsigned char tone_ph;    // tone phoneme to use with this vowel

	PHONEME_TAB *ph;
	unsigned int length;  // length_mod
	unsigned char env;    // pitch envelope number
	unsigned char type;
	unsigned char prepause;
	unsigned char amp;
	unsigned char newword;   // bit flags, see PHLIST_(START|END)_OF_*
	unsigned char pitch1;
	unsigned char pitch2;
	unsigned char std_length;
	unsigned int phontab_addr;
	int sound_param;
} PHONEME_LIST;

#define pd_FMT    0
#define pd_WAV    1
#define pd_VWLSTART 2
#define pd_VWLEND 3
#define pd_ADDWAV 4

#define N_PHONEME_DATA_PARAM 16
#define pd_INSERTPHONEME   i_INSERT_PHONEME
#define pd_APPENDPHONEME   i_APPEND_PHONEME
#define pd_CHANGEPHONEME   i_CHANGE_PHONEME
#define pd_CHANGE_NEXTPHONEME  i_REPLACE_NEXT_PHONEME
#define pd_LENGTHMOD       i_SET_LENGTH

#define pd_FORNEXTPH     0x2
#define pd_DONTLENGTHEN  0x4

typedef struct {
	int pd_control;
	int pd_param[N_PHONEME_DATA_PARAM];  // set from group 0 instructions
	int sound_addr[5];
	int sound_param[5];
	int vowel_transition[4];
	int pitch_env;
	int amp_env;
	char ipa_string[18];
} PHONEME_DATA;

typedef struct {
	int fmt_control;
	int use_vowelin;
	int fmt_addr;
	int fmt_length;
	int fmt_amp;
	int fmt2_addr;
	int fmt2_lenadj;
	int wav_addr;
	int wav_amp;
	int transition0;
	int transition1;
	int std_length;
} FMT_PARAMS;

typedef struct {
	PHONEME_LIST prev_vowel;
} WORD_PH_DATA;

// instructions

#define INSTN_RETURN         0x0001
#define INSTN_CONTINUE       0x0002

// Group 0 instructions with 8 bit operand.  These values go into bits 8-15 of the instruction
#define i_CHANGE_PHONEME 0x01
#define i_REPLACE_NEXT_PHONEME 0x02
#define i_INSERT_PHONEME 0x03
#define i_APPEND_PHONEME 0x04
#define i_APPEND_IFNEXTVOWEL 0x05
#define i_VOICING_SWITCH 0x06
#define i_PAUSE_BEFORE   0x07
#define i_PAUSE_AFTER    0x08
#define i_LENGTH_MOD     0x09
#define i_SET_LENGTH     0x0a
#define i_LONG_LENGTH    0x0b
#define i_ADD_LENGTH     0x0c
#define i_IPA_NAME       0x0d

#define i_CHANGE_IF      0x10  // 0x10 to 0x14

// conditions and jumps
#define i_CONDITION  0x2000
#define i_OR         0x1000  // added to i_CONDITION
#define i_NOT        0x0003

#define i_JUMP       0x6000
#define i_JUMP_FALSE 0x6800
#define i_SWITCH_NEXTVOWEL 0x6a00
#define i_SWITCH_PREVVOWEL 0x6c00
#define MAX_JUMP     255  // max jump distance

// multi-word instructions
#define i_CALLPH     0x9100
#define i_PITCHENV   0x9200
#define i_AMPENV     0x9300
#define i_VOWELIN    0xa100
#define i_VOWELOUT   0xa200
#define i_FMT        0xb000
#define i_WAV        0xc000
#define i_VWLSTART   0xd000
#define i_VWLENDING  0xe000
#define i_WAVADD     0xf000

// conditions
#define CONDITION_IS_PHONEME_TYPE 0x00
#define CONDITION_IS_PLACE_OF_ARTICULATION 0x20
#define CONDITION_IS_PHFLAG_SET 0x40
#define CONDITION_IS_OTHER 0x80

// other conditions (stress)
#define STRESS_IS_DIMINISHED    0       // diminished, unstressed within a word
#define STRESS_IS_UNSTRESSED    1       // unstressed, weak
#define STRESS_IS_NOT_STRESSED  2       // default, not stressed
#define STRESS_IS_SECONDARY     3       // secondary stress
#define STRESS_IS_PRIMARY       4       // primary (main) stress
#define STRESS_IS_PRIORITY      5       // replaces primary markers

// other conditions
#define isAfterStress  9
#define isNotVowel    10
#define isFinalVowel  11
#define isVoiced      12 // voiced consonant, or vowel
#define isFirstVowel  13
#define isSecondVowel 14
#define isTranslationGiven 16 // phoneme translation given in **_list or as [[...]]
#define isBreak        17 // pause phoneme or (stop/vstop/vfric not followed by vowel or (liquid in same word))
#define isWordStart    18
#define isWordEnd      19

#define i_StressLevel  0x800

#define LENGTH_MOD_LIMIT 10

typedef struct {
	int pause_factor;
	int clause_pause_factor;
	unsigned int min_pause;
	int wav_factor;
	int lenmod_factor;
	int lenmod2_factor;
	int min_sample_len;
	int fast_settings;	// TODO: rename this variable to better explain the purpose, or delete if there is none
} SPEED_FACTORS;

typedef struct {
	char name[12];
	unsigned char flags[4];
	signed char head_extend[8];

	unsigned char prehead_start;
	unsigned char prehead_end;
	unsigned char stressed_env;
	unsigned char stressed_drop;
	unsigned char secondary_drop;
	unsigned char unstressed_shape;

	unsigned char onset;
	unsigned char head_start;
	unsigned char head_end;
	unsigned char head_last;

	unsigned char head_max_steps;
	unsigned char n_head_extend;

	signed char unstr_start[3]; // for: onset, head, last
	signed char unstr_end[3];

	unsigned char nucleus0_env; // pitch envelope, tonic syllable is at end, no tail
	unsigned char nucleus0_max;
	unsigned char nucleus0_min;

	unsigned char nucleus1_env; // when followed by a tail
	unsigned char nucleus1_max;
	unsigned char nucleus1_min;
	unsigned char tail_start;
	unsigned char tail_end;

	unsigned char split_nucleus_env;
	unsigned char split_nucleus_max;
	unsigned char split_nucleus_min;
	unsigned char split_tail_start;
	unsigned char split_tail_end;
	unsigned char split_tune;

	unsigned char spare[8];
	int spare2; // the struct length should be a multiple of 4 bytes
} TUNE;



// phoneme table
extern PHONEME_TAB *phoneme_tab[N_PHONEME_TAB];

// list of phonemes in a clause
extern int n_phoneme_list;
extern PHONEME_LIST phoneme_list[N_PHONEME_LIST+1];
extern unsigned int embedded_list[];

extern const unsigned char env_fall[128];

// queue of commands for wavegen
#define WCMD_KLATT  1
#define WCMD_KLATT2 2
#define WCMD_SPECT  3
#define WCMD_SPECT2 4
#define WCMD_PAUSE  5
#define WCMD_WAVE    6
#define WCMD_WAVE2   7
#define WCMD_AMPLITUDE 8
#define WCMD_PITCH  9
#define WCMD_MARKER 10
#define WCMD_VOICE   11
#define WCMD_EMBEDDED 12
#define WCMD_MBROLA_DATA 13
#define WCMD_FMT_AMPLITUDE 14
#define WCMD_SONIC_SPEED 15
#define WCMD_PHONEME_ALIGNMENT 16

#define N_WCMDQ   170
#define MIN_WCMDQ  25   // need this many free entries before adding new phoneme

extern intptr_t wcmdq[N_WCMDQ][4];
extern int wcmdq_head;
extern int wcmdq_tail;

void MarkerEvent(int type, unsigned int char_position, int value, int value2, unsigned char *out_ptr);

extern unsigned char *wavefile_data;
extern int samplerate;

#define N_ECHO_BUF 5500   // max of 250mS at 22050 Hz
extern int echo_head;
extern int echo_tail;
extern int echo_amp;
extern short echo_buf[N_ECHO_BUF];

void SynthesizeInit(void);
int  Generate(PHONEME_LIST *phoneme_list, int *n_ph, bool resume);
int  SpeakNextClause(int control);
void SetSpeed(int control);
void SetEmbedded(int control, int value);
int FormantTransition2(frameref_t *seq, int *n_frames, unsigned int data1, unsigned int data2, PHONEME_TAB *other_ph, int which);

void Write4Bytes(FILE *f, int value);

#if USE_LIBSONIC
void DoSonicSpeed(int value);
#endif

#define ENV_LEN  128    // length of pitch envelopes
#define PITCHfall   0  // standard pitch envelopes
#define PITCHrise   2
#define N_ENVELOPE_DATA   20
extern const unsigned char *const envelope_data[N_ENVELOPE_DATA];

extern int formant_rate[];         // max rate of change of each formant
extern SPEED_FACTORS speed;

extern unsigned char *out_ptr;
extern unsigned char *out_end;
extern espeak_EVENT *event_list;
extern const int version_phdata;

void DoEmbedded(int *embix, int sourceix);
void DoMarker(int type, int char_posn, int length, int value);
void DoPhonemeMarker(int type, int char_posn, int length, char *name);
int DoSample3(PHONEME_DATA *phdata, int length_mod, int amp);
int DoSpect2(PHONEME_TAB *this_ph, int which, FMT_PARAMS *fmt_params,  PHONEME_LIST *plist, int modulation);
int PauseLength(int pause, int control);
const char *WordToString(char buf[5], unsigned int word);

#ifdef __cplusplus
}
#endif

#endif
