/*
 * Copyright (C) 2005 to 2010 by Jonathan Duddington
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

#ifndef ESPEAK_NG_PHONEME_H
#define ESPEAK_NG_PHONEME_H

#include <espeak-ng/espeak_ng.h>

#ifdef __cplusplus
extern "C"
{
#endif

// See docs/phonemes.md for the list of supported features.
typedef enum {
#	define FEATURE_T(a, b, c) ((a << 16) | (b << 8) | (c))
	// invalid phoneme feature name
	inv = 0,
	// manner of articulation
	nas = FEATURE_T('n', 'a', 's'),
	stp = FEATURE_T('s', 't', 'p'),
	afr = FEATURE_T('a', 'f', 'r'),
	frc = FEATURE_T('f', 'r', 'c'),
	flp = FEATURE_T('f', 'l', 'p'),
	trl = FEATURE_T('t', 'r', 'l'),
	apr = FEATURE_T('a', 'p', 'r'),
	clk = FEATURE_T('c', 'l', 'k'),
	ejc = FEATURE_T('e', 'j', 'c'),
	imp = FEATURE_T('i', 'm', 'p'),
	vwl = FEATURE_T('v', 'w', 'l'),
	lat = FEATURE_T('l', 'a', 't'),
	sib = FEATURE_T('s', 'i', 'b'),
	// place of articulation
	blb = FEATURE_T('b', 'l', 'b'),
	lbd = FEATURE_T('l', 'b', 'd'),
	bld = FEATURE_T('b', 'l', 'd'),
	dnt = FEATURE_T('d', 'n', 't'),
	alv = FEATURE_T('a', 'l', 'v'),
	pla = FEATURE_T('p', 'l', 'a'),
	rfx = FEATURE_T('r', 'f', 'x'),
	alp = FEATURE_T('a', 'l', 'p'),
	pal = FEATURE_T('p', 'a', 'l'),
	vel = FEATURE_T('v', 'e', 'l'),
	lbv = FEATURE_T('l', 'b', 'v'),
	uvl = FEATURE_T('u', 'v', 'l'),
	phr = FEATURE_T('p', 'h', 'r'),
	glt = FEATURE_T('g', 'l', 't'),
	// voice
	vcd = FEATURE_T('v', 'c', 'd'),
	vls = FEATURE_T('v', 'l', 's'),
	// vowel height
	hgh = FEATURE_T('h', 'g', 'h'),
	smh = FEATURE_T('s', 'm', 'h'),
	umd = FEATURE_T('u', 'm', 'd'),
	mid = FEATURE_T('m', 'i', 'd'),
	lmd = FEATURE_T('l', 'm', 'd'),
	sml = FEATURE_T('s', 'm', 'l'),
	low = FEATURE_T('l', 'o', 'w'),
	// vowel backness
	fnt = FEATURE_T('f', 'n', 't'),
	cnt = FEATURE_T('c', 'n', 't'),
	bck = FEATURE_T('b', 'c', 'k'),
	// rounding
	unr = FEATURE_T('u', 'n', 'r'),
	rnd = FEATURE_T('r', 'n', 'd'),
	// articulation
	lgl = FEATURE_T('l', 'g', 'l'),
	idt = FEATURE_T('i', 'd', 't'),
	apc = FEATURE_T('a', 'p', 'c'),
	lmn = FEATURE_T('l', 'm', 'n'),
	// air flow
	egs = FEATURE_T('e', 'g', 's'),
	igs = FEATURE_T('i', 'g', 's'),
	// phonation
	brv = FEATURE_T('b', 'r', 'v'),
	slv = FEATURE_T('s', 'l', 'v'),
	stv = FEATURE_T('s', 't', 'v'),
	crv = FEATURE_T('c', 'r', 'v'),
	glc = FEATURE_T('g', 'l', 'c'),
	// rounding and labialization
	ptr = FEATURE_T('p', 't', 'r'),
	cmp = FEATURE_T('c', 'm', 'p'),
	mrd = FEATURE_T('m', 'r', 'd'),
	lrd = FEATURE_T('l', 'r', 'd'),
	// syllabicity
	syl = FEATURE_T('s', 'y', 'l'),
	nsy = FEATURE_T('n', 's', 'y'),
	// consonant release
	asp = FEATURE_T('a', 's', 'p'),
	nrs = FEATURE_T('n', 'r', 's'),
	lrs = FEATURE_T('l', 'r', 's'),
	unx = FEATURE_T('u', 'n', 'x'),
	// coarticulation
	pzd = FEATURE_T('p', 'z', 'd'),
	vzd = FEATURE_T('v', 'z', 'd'),
	fzd = FEATURE_T('f', 'z', 'd'),
	nzd = FEATURE_T('n', 'z', 'd'),
	rzd = FEATURE_T('r', 'z', 'd'),
	// tongue root
	atr = FEATURE_T('a', 't', 'r'),
	rtr = FEATURE_T('r', 't', 'r'),
	// fortis and lenis
	fts = FEATURE_T('f', 't', 's'),
	lns = FEATURE_T('l', 'n', 's'),
	// length
	est = FEATURE_T('e', 's', 't'),
	hlg = FEATURE_T('h', 'l', 'g'),
	lng = FEATURE_T('l', 'n', 'g'),
	elg = FEATURE_T('e', 'l', 'g'),
#	undef FEATURE_T
} phoneme_feature_t;

phoneme_feature_t phoneme_feature_from_string(const char *feature);

// phoneme types
#define phPAUSE   0
#define phSTRESS  1
#define phVOWEL   2
#define phLIQUID  3
#define phSTOP    4
#define phVSTOP   5
#define phFRICATIVE 6
#define phVFRICATIVE 7
#define phNASAL   8
#define phVIRTUAL 9
#define phDELETED 14
#define phINVALID 15

// places of articulation (phARTICULATION)
#define phPLACE_BILABIAL 1
#define phPLACE_LABIODENTAL 2
#define phPLACE_DENTAL 3
#define phPLACE_ALVEOLAR 4
#define phPLACE_RETROFLEX 5
#define phPLACE_PALATO_ALVEOLAR 6
#define phPLACE_PALATAL 7
#define phPLACE_VELAR 8
#define phPLACE_LABIO_VELAR 9
#define phPLACE_UVULAR 10
#define phPLACE_PHARYNGEAL 11
#define phPLACE_GLOTTAL 12

// phflags
#define phFLAGBIT_UNSTRESSED 1
#define phFLAGBIT_VOICELESS 3
#define phFLAGBIT_VOICED 4
#define phFLAGBIT_SIBILANT 5
#define phFLAGBIT_NOLINK 6
#define phFLAGBIT_TRILL 7
#define phFLAGBIT_PALATAL 9
#define phFLAGBIT_BRKAFTER 14 // [*] add a post-pause
#define phARTICULATION 0xf0000 // bits 16-19
#define phFLAGBIT_NONSYLLABIC 20 // don't count this vowel as a syllable when finding the stress position
#define phFLAGBIT_LONG 21
#define phFLAGBIT_LENGTHENSTOP 22 // make the pre-pause slightly longer
#define phFLAGBIT_RHOTIC 23
#define phFLAGBIT_NOPAUSE 24
#define phFLAGBIT_PREVOICE 25 // for voiced stops
#define phFLAGBIT_FLAG1 28
#define phFLAGBIT_FLAG2 29
#define phFLAGBIT_LOCAL 31 // used during compilation

// phoneme properties
#define phUNSTRESSED   (1U << phFLAGBIT_UNSTRESSED)
#define phVOICELESS    (1U << phFLAGBIT_VOICELESS)
#define phVOICED       (1U << phFLAGBIT_VOICED)
#define phSIBILANT     (1U << phFLAGBIT_SIBILANT)
#define phNOLINK       (1U << phFLAGBIT_NOLINK)
#define phTRILL        (1U << phFLAGBIT_TRILL)
#define phPALATAL      (1U << phFLAGBIT_PALATAL)
#define phBRKAFTER     (1U << phFLAGBIT_BRKAFTER)
#define phNONSYLLABIC  (1U << phFLAGBIT_NONSYLLABIC)
#define phLONG         (1U << phFLAGBIT_LONG)
#define phLENGTHENSTOP (1U << phFLAGBIT_LENGTHENSTOP)
#define phRHOTIC       (1U << phFLAGBIT_RHOTIC)
#define phNOPAUSE      (1U << phFLAGBIT_NOPAUSE)
#define phPREVOICE     (1U << phFLAGBIT_PREVOICE)
#define phFLAG1        (1U << phFLAGBIT_FLAG1)
#define phFLAG2        (1U << phFLAGBIT_FLAG2)
#define phLOCAL        (1U << phFLAGBIT_LOCAL)

// fixed phoneme code numbers, these can be used from the program code
#define phonCONTROL     1
#define phonSTRESS_U    2
#define phonSTRESS_D    3
#define phonSTRESS_2    4
#define phonSTRESS_3    5
#define phonSTRESS_P    6
#define phonSTRESS_P2   7    // priority stress within a word
#define phonSTRESS_PREV 8
#define phonPAUSE       9
#define phonPAUSE_SHORT 10
#define phonPAUSE_NOLINK 11
#define phonLENGTHEN    12
#define phonSCHWA       13
#define phonSCHWA_SHORT 14
#define phonEND_WORD    15
#define phonDEFAULTTONE 17
#define phonCAPITAL     18
#define phonGLOTTALSTOP 19
#define phonSYLLABIC    20
#define phonSWITCH      21
#define phonX1          22      // a language specific action
#define phonPAUSE_VSHORT 23
#define phonPAUSE_LONG  24
#define phonT_REDUCED   25
#define phonSTRESS_TONIC 26
#define phonPAUSE_CLAUSE 27
#define phonVOWELTYPES   28  // 28 to 33

#define N_PHONEME_TABS     150     // number of phoneme tables
#define N_PHONEME_TAB      256     // max phonemes in a phoneme table
#define N_PHONEME_TAB_NAME  32     // must be multiple of 4

// main table of phonemes, index by phoneme number (1-254)

typedef struct {
	unsigned int mnemonic;       // Up to 4 characters.  The first char is in the l.s.byte
	unsigned int phflags;        // bits 16-19 place of articulation
	unsigned short program;      // index into phondata file
	unsigned char code;          // the phoneme number
	unsigned char type;          // phVOWEL, phPAUSE, phSTOP etc
	unsigned char start_type;
	unsigned char end_type;      // vowels: endtype; consonant: voicing switch
	unsigned char std_length;    // for vowels, in mS/2;  for phSTRESS phonemes, this is the stress/tone type
	unsigned char length_mod;    // a length_mod group number, used to access length_mod_tab
} PHONEME_TAB;

espeak_ng_STATUS
phoneme_add_feature(PHONEME_TAB *phoneme,
                    phoneme_feature_t feature);

// Several phoneme tables may be loaded into memory. phoneme_tab points to
// one for the current voice
extern int n_phoneme_tab;
extern PHONEME_TAB *phoneme_tab[N_PHONEME_TAB];

typedef struct {
	char name[N_PHONEME_TAB_NAME];
	PHONEME_TAB *phoneme_tab_ptr;
	int n_phonemes;
	int includes;            // also include the phonemes from this other phoneme table
} PHONEME_TAB_LIST;

// table of phonemes to be replaced with different phonemes, for the current voice
#define N_REPLACE_PHONEMES   60
typedef struct {
	unsigned char old_ph;
	unsigned char new_ph;
	char type;   // 0=always replace, 1=only at end of word
} REPLACE_PHONEMES;

extern int n_replace_phonemes;
extern REPLACE_PHONEMES replace_phonemes[N_REPLACE_PHONEMES];

// Table of phoneme programs and lengths.  Used by MakeVowelLists
typedef struct {
	unsigned int addr;
	unsigned int length;
} PHONEME_PROG_LOG;

#define PhonemeCode2(c1, c2) PhonemeCode((c2<<8)+c1)

extern PHONEME_TAB_LIST phoneme_tab_list[N_PHONEME_TABS];
extern int phoneme_tab_number;

#ifdef __cplusplus
}
#endif

#endif
