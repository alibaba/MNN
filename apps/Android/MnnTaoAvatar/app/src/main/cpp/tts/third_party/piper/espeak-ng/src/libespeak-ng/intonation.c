/*
 * Copyright (C) 2005 to 2007 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2016 Reece H. Dunn
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "intonation.h"
#include "phoneme.h"     // for PHONEME_TAB, PhonemeCode2, phonPAUSE, phPAUSE
#include "synthdata.h"   // for PhonemeCode
#include "synthesize.h"  // for PHONEME_LIST, TUNE, phoneme_list, phoneme_tab
#include "translate.h"   // for Translator, LANGUAGE_OPTIONS, L, OPTION_EMPH...

/* Note this module is mostly old code that needs to be rewritten to
   provide a more flexible intonation system.
 */

// bits in SYLLABLE.flags
#define SYL_RISE        1
#define SYL_EMPHASIS    2
#define SYL_END_CLAUSE   4

typedef struct {
	char stress;
	char env;
	char flags; // bit 0=pitch rising, bit1=emnphasized, bit2=end of clause
	char nextph_type;
	unsigned char pitch1;
	unsigned char pitch2;
} SYLLABLE;

static int tone_pitch_env; // used to return pitch envelope

/* Pitch data for tone types */
/*****************************/

#define PITCHfall     0
#define PITCHrise     2
#define PITCHfrise    4 // and 3 must be for the variant preceded by 'r'
#define PITCHfrise2   6 // and 5 must be the 'r' variant

const unsigned char env_fall[128] = {
	0xff, 0xfd, 0xfa, 0xf8, 0xf6, 0xf4, 0xf2, 0xf0, 0xee, 0xec, 0xea, 0xe8, 0xe6, 0xe4, 0xe2, 0xe0,
	0xde, 0xdc, 0xda, 0xd8, 0xd6, 0xd4, 0xd2, 0xd0, 0xce, 0xcc, 0xca, 0xc8, 0xc6, 0xc4, 0xc2, 0xc0,
	0xbe, 0xbc, 0xba, 0xb8, 0xb6, 0xb4, 0xb2, 0xb0, 0xae, 0xac, 0xaa, 0xa8, 0xa6, 0xa4, 0xa2, 0xa0,
	0x9e, 0x9c, 0x9a, 0x98, 0x96, 0x94, 0x92, 0x90, 0x8e, 0x8c, 0x8a, 0x88, 0x86, 0x84, 0x82, 0x80,
	0x7e, 0x7c, 0x7a, 0x78, 0x76, 0x74, 0x72, 0x70, 0x6e, 0x6c, 0x6a, 0x68, 0x66, 0x64, 0x62, 0x60,
	0x5e, 0x5c, 0x5a, 0x58, 0x56, 0x54, 0x52, 0x50, 0x4e, 0x4c, 0x4a, 0x48, 0x46, 0x44, 0x42, 0x40,
	0x3e, 0x3c, 0x3a, 0x38, 0x36, 0x34, 0x32, 0x30, 0x2e, 0x2c, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20,
	0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
};

static const unsigned char env_rise[128] = {
	0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e,
	0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
	0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e,
	0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e,
	0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e,
	0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe,
	0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde,
	0xe0, 0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfd, 0xff
};

static const unsigned char env_frise[128] = {
	0xff, 0xf4, 0xea, 0xe0, 0xd6, 0xcc, 0xc3, 0xba, 0xb1, 0xa8, 0x9f, 0x97, 0x8f, 0x87, 0x7f, 0x78,
	0x71, 0x6a, 0x63, 0x5c, 0x56, 0x50, 0x4a, 0x44, 0x3f, 0x39, 0x34, 0x2f, 0x2b, 0x26, 0x22, 0x1e,
	0x1a, 0x17, 0x13, 0x10, 0x0d, 0x0b, 0x08, 0x06, 0x04, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x13, 0x15, 0x17,
	0x1a, 0x1d, 0x1f, 0x22, 0x25, 0x28, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x39, 0x3b, 0x3d, 0x40,
	0x42, 0x45, 0x47, 0x4a, 0x4c, 0x4f, 0x51, 0x54, 0x57, 0x5a, 0x5d, 0x5f, 0x62, 0x65, 0x68, 0x6b,
	0x6e, 0x71, 0x74, 0x78, 0x7b, 0x7e, 0x81, 0x85, 0x88, 0x8b, 0x8f, 0x92, 0x96, 0x99, 0x9d, 0xa0,
	0xa4, 0xa8, 0xac, 0xaf, 0xb3, 0xb7, 0xbb, 0xbf, 0xc3, 0xc7, 0xcb, 0xcf, 0xd3, 0xd7, 0xdb, 0xe0
};

static const unsigned char env_r_frise[128] = {
	0xcf, 0xcc, 0xc9, 0xc6, 0xc3, 0xc0, 0xbd, 0xb9, 0xb4, 0xb0, 0xab, 0xa7, 0xa2, 0x9c, 0x97, 0x92,
	0x8c, 0x86, 0x81, 0x7b, 0x75, 0x6f, 0x69, 0x63, 0x5d, 0x57, 0x50, 0x4a, 0x44, 0x3e, 0x38, 0x33,
	0x2d, 0x27, 0x22, 0x1c, 0x17, 0x12, 0x0d, 0x08, 0x04, 0x02, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08, 0x0a, 0x0c, 0x0d, 0x0f, 0x12, 0x14, 0x16,
	0x19, 0x1b, 0x1e, 0x21, 0x24, 0x27, 0x2a, 0x2d, 0x30, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3f, 0x41,
	0x43, 0x46, 0x48, 0x4b, 0x4d, 0x50, 0x52, 0x55, 0x58, 0x5a, 0x5d, 0x60, 0x63, 0x66, 0x69, 0x6c,
	0x6f, 0x72, 0x75, 0x78, 0x7b, 0x7e, 0x81, 0x85, 0x88, 0x8b, 0x8f, 0x92, 0x96, 0x99, 0x9d, 0xa0,
	0xa4, 0xa8, 0xac, 0xaf, 0xb3, 0xb7, 0xbb, 0xbf, 0xc3, 0xc7, 0xcb, 0xcf, 0xd3, 0xd7, 0xdb, 0xe0
};

static const unsigned char env_frise2[128] = {
	0xff, 0xf9, 0xf4, 0xee, 0xe9, 0xe4, 0xdf, 0xda, 0xd5, 0xd0, 0xcb, 0xc6, 0xc1, 0xbd, 0xb8, 0xb3,
	0xaf, 0xaa, 0xa6, 0xa1, 0x9d, 0x99, 0x95, 0x90, 0x8c, 0x88, 0x84, 0x80, 0x7d, 0x79, 0x75, 0x71,
	0x6e, 0x6a, 0x67, 0x63, 0x60, 0x5d, 0x59, 0x56, 0x53, 0x50, 0x4d, 0x4a, 0x47, 0x44, 0x41, 0x3e,
	0x3c, 0x39, 0x37, 0x34, 0x32, 0x2f, 0x2d, 0x2b, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a,
	0x19, 0x17, 0x15, 0x14, 0x12, 0x11, 0x0f, 0x0e, 0x0d, 0x0c, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05,
	0x05, 0x04, 0x03, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x04, 0x04, 0x05, 0x06, 0x07, 0x08,
	0x09, 0x0a, 0x0b, 0x0c, 0x0e, 0x0f, 0x10, 0x12, 0x13, 0x15, 0x17, 0x18, 0x1a, 0x1c, 0x1e, 0x20
};

static const unsigned char env_r_frise2[128] = {
	0xd0, 0xce, 0xcd, 0xcc, 0xca, 0xc8, 0xc7, 0xc5, 0xc3, 0xc1, 0xc0, 0xbd, 0xbb, 0xb8, 0xb5, 0xb3,
	0xb0, 0xad, 0xaa, 0xa7, 0xa3, 0xa0, 0x9d, 0x99, 0x96, 0x92, 0x8f, 0x8b, 0x87, 0x84, 0x80, 0x7c,
	0x78, 0x74, 0x70, 0x6d, 0x69, 0x65, 0x61, 0x5d, 0x59, 0x55, 0x51, 0x4d, 0x4a, 0x46, 0x42, 0x3e,
	0x3b, 0x37, 0x34, 0x31, 0x2f, 0x2d, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x19,
	0x17, 0x15, 0x14, 0x12, 0x11, 0x0f, 0x0e, 0x0d, 0x0c, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x05,
	0x04, 0x03, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x04, 0x04, 0x05, 0x06, 0x07, 0x08,
	0x09, 0x0a, 0x0b, 0x0c, 0x0e, 0x0f, 0x10, 0x12, 0x13, 0x15, 0x17, 0x18, 0x1a, 0x1c, 0x1e, 0x20
};

static const unsigned char env_risefall[128] = {
	0x98, 0x99, 0x99, 0x9a, 0x9c, 0x9d, 0x9f, 0xa1, 0xa4, 0xa7, 0xa9, 0xac, 0xb0, 0xb3, 0xb6, 0xba,
	0xbe, 0xc1, 0xc5, 0xc9, 0xcd, 0xd1, 0xd4, 0xd8, 0xdc, 0xdf, 0xe3, 0xe6, 0xea, 0xed, 0xf0, 0xf2,
	0xf5, 0xf7, 0xf9, 0xfb, 0xfc, 0xfd, 0xfe, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xfd,
	0xfb, 0xfa, 0xf8, 0xf6, 0xf3, 0xf1, 0xee, 0xec, 0xe9, 0xe6, 0xe4, 0xe0, 0xdd, 0xda, 0xd7, 0xd3,
	0xd0, 0xcc, 0xc8, 0xc4, 0xc0, 0xbc, 0xb8, 0xb4, 0xb0, 0xac, 0xa7, 0xa3, 0x9f, 0x9a, 0x96, 0x91,
	0x8d, 0x88, 0x84, 0x7f, 0x7b, 0x76, 0x72, 0x6d, 0x69, 0x65, 0x60, 0x5c, 0x58, 0x54, 0x50, 0x4c,
	0x48, 0x44, 0x40, 0x3c, 0x39, 0x35, 0x32, 0x2f, 0x2b, 0x28, 0x26, 0x23, 0x20, 0x1d, 0x1a, 0x17,
	0x15, 0x12, 0x0f, 0x0d, 0x0a, 0x08, 0x07, 0x05, 0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00
};

static const unsigned char env_rise2[128] = {
	0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05, 0x06, 0x06,
	0x07, 0x08, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
	0x16, 0x17, 0x18, 0x19, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x22, 0x23, 0x25, 0x26, 0x28, 0x29, 0x2b,
	0x2d, 0x2f, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x47, 0x49, 0x4b,
	0x4e, 0x50, 0x52, 0x55, 0x57, 0x5a, 0x5d, 0x5f, 0x62, 0x65, 0x67, 0x6a, 0x6d, 0x70, 0x73, 0x76,
	0x79, 0x7c, 0x7f, 0x82, 0x86, 0x89, 0x8c, 0x90, 0x93, 0x96, 0x9a, 0x9d, 0xa0, 0xa3, 0xa6, 0xa9,
	0xac, 0xaf, 0xb2, 0xb5, 0xb8, 0xbb, 0xbe, 0xc1, 0xc4, 0xc7, 0xca, 0xcd, 0xd0, 0xd3, 0xd6, 0xd9,
	0xdc, 0xdf, 0xe2, 0xe4, 0xe7, 0xe9, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfb, 0xfd
};

static const unsigned char env_fall2[128] = {
	0xfe, 0xfe, 0xfd, 0xfd, 0xfc, 0xfb, 0xfb, 0xfa, 0xfa, 0xf9, 0xf8, 0xf8, 0xf7, 0xf7, 0xf6, 0xf6,
	0xf5, 0xf4, 0xf4, 0xf3, 0xf3, 0xf2, 0xf2, 0xf1, 0xf0, 0xf0, 0xef, 0xee, 0xee, 0xed, 0xec, 0xeb,
	0xea, 0xea, 0xe9, 0xe8, 0xe7, 0xe6, 0xe5, 0xe4, 0xe3, 0xe2, 0xe1, 0xe0, 0xde, 0xdd, 0xdc, 0xdb,
	0xd9, 0xd8, 0xd6, 0xd5, 0xd3, 0xd2, 0xd0, 0xce, 0xcc, 0xcb, 0xc9, 0xc7, 0xc5, 0xc3, 0xc0, 0xbe,
	0xbc, 0xb9, 0xb7, 0xb5, 0xb2, 0xaf, 0xad, 0xaa, 0xa7, 0xa4, 0xa1, 0x9e, 0x9a, 0x97, 0x94, 0x90,
	0x8d, 0x89, 0x85, 0x81, 0x7d, 0x79, 0x75, 0x71, 0x6d, 0x68, 0x64, 0x61, 0x5e, 0x5b, 0x57, 0x54,
	0x51, 0x4d, 0x4a, 0x46, 0x43, 0x40, 0x3c, 0x39, 0x35, 0x32, 0x2e, 0x2a, 0x27, 0x23, 0x1f, 0x1c,
	0x18, 0x14, 0x11, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00
};

static const unsigned char env_fallrise3[128] = {
	0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xfd, 0xfc, 0xfa, 0xf8, 0xf6, 0xf4, 0xf1, 0xee, 0xeb,
	0xe8, 0xe5, 0xe1, 0xde, 0xda, 0xd6, 0xd2, 0xcd, 0xc9, 0xc4, 0xbf, 0xba, 0xb6, 0xb0, 0xab, 0xa6,
	0xa1, 0x9c, 0x96, 0x91, 0x8b, 0x86, 0x80, 0x7b, 0x75, 0x6f, 0x6a, 0x64, 0x5f, 0x59, 0x54, 0x4f,
	0x49, 0x44, 0x3f, 0x3a, 0x35, 0x30, 0x2b, 0x26, 0x22, 0x1d, 0x19, 0x15, 0x11, 0x0d, 0x0a, 0x07,
	0x04, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x02, 0x04, 0x05,
	0x07, 0x09, 0x0b, 0x0d, 0x10, 0x12, 0x15, 0x18, 0x1b, 0x1e, 0x22, 0x25, 0x29, 0x2d, 0x31, 0x35,
	0x3a, 0x3e, 0x43, 0x48, 0x4c, 0x51, 0x57, 0x5b, 0x5e, 0x62, 0x65, 0x68, 0x6b, 0x6e, 0x71, 0x74,
	0x76, 0x78, 0x7b, 0x7c, 0x7e, 0x80, 0x81, 0x82, 0x83, 0x83, 0x84, 0x84, 0x83, 0x83, 0x82, 0x81
};

static const unsigned char env_fallrise4[128] = {
	0x72, 0x72, 0x71, 0x71, 0x70, 0x6f, 0x6d, 0x6c, 0x6a, 0x68, 0x66, 0x64, 0x61, 0x5f, 0x5c, 0x5a,
	0x57, 0x54, 0x51, 0x4e, 0x4b, 0x48, 0x45, 0x42, 0x3f, 0x3b, 0x38, 0x35, 0x32, 0x2f, 0x2c, 0x29,
	0x26, 0x23, 0x20, 0x1d, 0x1b, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
	0x07, 0x07, 0x07, 0x07, 0x07, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x06,
	0x07, 0x07, 0x08, 0x09, 0x0a, 0x0c, 0x0d, 0x0f, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1b, 0x1d, 0x20,
	0x23, 0x26, 0x29, 0x2c, 0x2f, 0x33, 0x37, 0x3b, 0x3f, 0x43, 0x47, 0x4c, 0x51, 0x56, 0x5b, 0x60,
	0x65, 0x6a, 0x6f, 0x74, 0x79, 0x7f, 0x84, 0x89, 0x8f, 0x95, 0x9b, 0xa1, 0xa7, 0xad, 0xb3, 0xba,
	0xc0, 0xc7, 0xce, 0xd5, 0xdc, 0xe3, 0xea, 0xf1, 0xf5, 0xf7, 0xfa, 0xfc, 0xfd, 0xfe, 0xff, 0xff
};

static const unsigned char env_risefallrise[128] = {
	0x7f, 0x7f, 0x7f, 0x80, 0x81, 0x83, 0x84, 0x87, 0x89, 0x8c, 0x8f, 0x92, 0x96, 0x99, 0x9d, 0xa1,
	0xa5, 0xaa, 0xae, 0xb2, 0xb7, 0xbb, 0xc0, 0xc5, 0xc9, 0xcd, 0xd2, 0xd6, 0xda, 0xde, 0xe2, 0xe6,
	0xea, 0xed, 0xf0, 0xf3, 0xf5, 0xf8, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xfe, 0xfd, 0xfc, 0xfb, 0xf9,
	0xf7, 0xf4, 0xf0, 0xec, 0xe7, 0xe2, 0xdc, 0xd5, 0xce, 0xc6, 0xbd, 0xb4, 0xa9, 0x9e, 0x92, 0x88,
	0x82, 0x7d, 0x77, 0x72, 0x6c, 0x66, 0x60, 0x5a, 0x54, 0x4e, 0x49, 0x42, 0x3c, 0x37, 0x32, 0x2d,
	0x28, 0x24, 0x1f, 0x1b, 0x18, 0x14, 0x11, 0x0e, 0x0c, 0x09, 0x07, 0x06, 0x05, 0x04, 0x04, 0x04,
	0x04, 0x05, 0x06, 0x08, 0x0a, 0x0d, 0x10, 0x14, 0x18, 0x1d, 0x23, 0x29, 0x2f, 0x37, 0x3e, 0x47,
	0x50, 0x5a, 0x64, 0x70, 0x7c, 0x83, 0x85, 0x88, 0x8a, 0x8c, 0x8e, 0x8f, 0x91, 0x92, 0x93, 0x93
};

const unsigned char *const envelope_data[N_ENVELOPE_DATA] = {
	env_fall,  env_fall,
	env_rise,  env_rise,
	env_frise,  env_r_frise,
	env_frise2, env_r_frise2,
	env_risefall, env_risefall,

	env_fallrise3, env_fallrise3,
	env_fallrise4, env_fallrise4,
	env_fall2, env_fall2,
	env_rise2, env_rise2,
	env_risefallrise, env_risefallrise
};

// indexed by stress
static const int min_drop[] =  { 6, 7, 9, 9, 20, 20, 20, 25 };

// pitch change during the main part of the clause
static const int drops_0[8] = { 9, 9, 16, 16, 16, 23, 55, 32 };

// overflow table values are 64ths of the body pitch range (between body_start and body_end)
static const signed char oflow[] = { 0, 40, 24, 8, 0 };
static const signed char oflow_emf[] = { 10, 52, 32, 20, 10 };
static const signed char oflow_less[] = { 6, 38, 24, 14, 4 };

#define N_TONE_HEAD_TABLE    13
#define N_TONE_NUCLEUS_TABLE 13

typedef struct {
	unsigned char pre_start;
	unsigned char pre_end;

	unsigned char body_start;
	unsigned char body_end;

	const int  *body_drops;
	unsigned char body_max_steps;
	char body_lower_u;

	unsigned char n_overflow;
	const signed char *overflow;
} TONE_HEAD;

typedef struct {
	unsigned char pitch_env0; // pitch envelope, tonic syllable at end
	unsigned char tonic_max0;
	unsigned char tonic_min0;

	unsigned char pitch_env1; // followed by unstressed
	unsigned char tonic_max1;
	unsigned char tonic_min1;

	short *backwards;

	unsigned char tail_start;
	unsigned char tail_end;
	unsigned char flags;
} TONE_NUCLEUS;

#define T_EMPH  1

static const TONE_HEAD tone_head_table[N_TONE_HEAD_TABLE] = {
	{ 46, 57,   78, 50,  drops_0, 3, 7,   5, oflow },      // 0 statement
	{ 46, 57,   78, 46,  drops_0, 3, 7,   5, oflow },      // 1 comma
	{ 46, 57,   78, 46,  drops_0, 3, 7,   5, oflow },      // 2 question
	{ 46, 57,   90, 50,  drops_0, 3, 9,   5, oflow_emf },  // 3 exclamation
	{ 46, 57,   78, 50,  drops_0, 3, 7,   5, oflow },      // 4 statement, emphatic
	{ 46, 57,   74, 55,  drops_0, 4, 7,   5, oflow_less }, // 5 statement, less intonation
	{ 46, 57,   74, 55,  drops_0, 4, 7,   5, oflow_less }, // 6 comma, less intonation
	{ 46, 57,   74, 55,  drops_0, 4, 7,   5, oflow_less }, // 7 comma, less intonation, less rise
	{ 46, 57,   78, 50,  drops_0, 3, 7,   5, oflow },      // 8 pitch raises at end of sentence
	{ 46, 57,   78, 46,  drops_0, 3, 7,   5, oflow },      // 9 comma
	{ 46, 57,   78, 50,  drops_0, 3, 7,   5, oflow },      // 10  question
	{ 34, 41,   41, 32,  drops_0, 3, 7,   5, oflow_less }, // 11 test
	{ 46, 57,   55, 50,  drops_0, 3, 7,   5, oflow_less }, // 12 test
};

static const TONE_NUCLEUS tone_nucleus_table[N_TONE_NUCLEUS_TABLE] = {
	{ PITCHfall,   64,  8, PITCHfall,   70, 18, NULL, 24, 12, 0 },      //  0 statement
	{ PITCHfrise,  80, 18, PITCHfrise2, 78, 22, NULL, 34, 52, 0 },      //  1 comma
	{ PITCHfrise,  88, 22, PITCHfrise2, 82, 22, NULL, 34, 64, 0 },      //  2 question
	{ PITCHfall,   92,  8, PITCHfall,   92, 80, NULL, 76,  8, T_EMPH }, //  3 exclamation
	{ PITCHfall,   86,  4, PITCHfall,   94, 66, NULL, 34, 10, 0 },      //  4 statement, emphatic
	{ PITCHfall,   62, 10, PITCHfall,   62, 20, NULL, 28, 16, 0 },      //  5 statement, less intonation
	{ PITCHfrise,  68, 18, PITCHfrise2, 68, 22, NULL, 30, 44, 0 },      //  6 comma, less intonation
	{ PITCHfrise2, 64, 16, PITCHfall,   66, 32, NULL, 32, 18, 0 },      //  7 comma, less intonation, less rise
	{ PITCHrise,   68, 46, PITCHfall,   42, 32, NULL, 46, 58, 0 },      //  8 pitch raises at end of sentence
	{ PITCHfrise,  78, 24, PITCHfrise2, 72, 22, NULL, 42, 52, 0 },      //  9 comma
	{ PITCHfrise,  88, 34, PITCHfall,   64, 32, NULL, 46, 82, 0 },      // 10 question
	{ PITCHfall,   56, 12, PITCHfall,   56, 20, NULL, 24, 12, 0 },      // 11 test
	{ PITCHfall,   70, 18, PITCHfall,   70, 24, NULL, 32, 20, 0 },      // 12 test
};

#define SECONDARY        3
#define PRIMARY          4
#define PRIMARY_STRESSED 6
#define PRIMARY_LAST     7

static int number_pre;
static int number_tail;
static int last_primary;
static int tone_posn;
static int tone_posn2;
static int no_tonic;

static void count_pitch_vowels(SYLLABLE *syllable_tab, int start, int end, int clause_end)
{
	int ix;
	int stress;
	int max_stress = 0;
	int max_stress_posn = 0;  // last syllable ot the highest stress
	int max_stress_posn2 = 0; // penuntimate syllable of the highest stress

	number_pre = -1; // number of vowels before 1st primary stress
	number_tail = 0; // number between tonic syllable and next primary
	last_primary = -1;

	for (ix = start; ix < end; ix++) {
		stress = syllable_tab[ix].stress; // marked stress level

		if (stress >= max_stress) {
			if (stress > max_stress)
				max_stress_posn2 = ix;
			else
				max_stress_posn2 = max_stress_posn;
			max_stress_posn = ix;
			max_stress = stress;
		}
		if (stress >= PRIMARY) {
			if (number_pre < 0)
				number_pre = ix - start;

			last_primary = ix;
		}
	}

	if (number_pre < 0)
		number_pre = end;

	number_tail = end - max_stress_posn - 1;
	tone_posn = max_stress_posn;
	tone_posn2 = max_stress_posn2;

	if (no_tonic)
		tone_posn = tone_posn2 = end; // next position after the end of the truncated clause
	else if (last_primary >= 0) {
		if (end == clause_end)
			syllable_tab[last_primary].stress = PRIMARY_LAST;
	} else {
		// no primary stress. Use the highest stress
		syllable_tab[tone_posn].stress = PRIMARY_LAST;
	}
}

// Count number of primary stresses up to tonic syllable or body_reset
static int count_increments(SYLLABLE *syllable_tab, int ix, int end_ix, int min_stress)
{
	int count = 0;
	int stress;

	while (ix < end_ix) {
		stress = syllable_tab[ix++].stress;
		if (stress >= PRIMARY_LAST)
			break;

		if (stress >= min_stress)
			count++;
	}
	return count;
}

// Set the pitch of a vowel in syllable_tab
static void set_pitch(SYLLABLE *syl, int base, int drop)
{
	int pitch1, pitch2;
	int flags = 0;

	if (base < 0)  base = 0;

	pitch2 = base;

	if (drop < 0) {
		flags = SYL_RISE;
		drop = -drop;
	}

	pitch1 = pitch2 + drop;
	if (pitch1 < 0)
		pitch1 = 0;

	if (pitch1 > 254) pitch1 = 254;
	if (pitch2 > 254) pitch2 = 254;

	syl->pitch1 = pitch1;
	syl->pitch2 = pitch2;
	syl->flags |= flags;
}

static int CountUnstressed(SYLLABLE *syllable_tab, int start, int end, int limit)
{
	int ix;

	for (ix = start; ix <= end; ix++) {
		if (syllable_tab[ix].stress >= limit)
			break;
	}
	return ix - start;
}

static int SetHeadIntonation(SYLLABLE *syllable_tab, TUNE *tune, int syl_ix, int end_ix)
{
	int stress;
	SYLLABLE *syl;
	int ix;
	int pitch = 0;
	int increment = 0;
	int n_steps = 0;
	int stage; // onset, head, last
	bool initial;
	int overflow_ix = 0;
	int pitch_range;
	int pitch_range_abs;
	const int *drops;
	int n_unstressed = 0;
	int unstressed_ix = 0;
	int unstressed_inc;
	bool used_onset = false;
	int head_final = end_ix;
	int secondary = 2;

	pitch_range = (tune->head_end - tune->head_start) * 256;
	pitch_range_abs = abs(pitch_range);
	drops = drops_0; // this should be controlled by tune->head_drops
	initial = true;

	stage = 0;
	if (tune->onset == 255)
		stage = 1; // no onset specified

	if (tune->head_last != 255) {
		// find the last primary stress in the body
		for (ix = end_ix-1; ix >= syl_ix; ix--) {
			if (syllable_tab[ix].stress >= 4) {
				head_final = ix;
				break;
			}
		}
	}

	while (syl_ix < end_ix) {
		syl = &syllable_tab[syl_ix];
		stress = syl->stress;

		if (initial || (stress >= 4)) {
			// a primary stress

			if ((initial) || (stress == 5)) {
				initial = false;
				overflow_ix = 0;

				if (tune->onset == 255) {
					n_steps = count_increments(syllable_tab, syl_ix, head_final, 4);
					pitch = tune->head_start * 256;
				} else {
					// a pitch has been specified for the onset syllable, don't include it in the pitch incrementing
					n_steps = count_increments(syllable_tab, syl_ix+1, head_final, 4);
					pitch = tune->onset * 256;
					used_onset = true;
				}

				if (n_steps > tune->head_max_steps)
					n_steps = tune->head_max_steps;

				if (n_steps > 1)
					increment = pitch_range / (n_steps -1);
				else
					increment = 0;
			} else if (syl_ix == head_final) {
				// a pitch has been specified for the last primary stress before the nucleus
				pitch = tune->head_last * 256;
				stage = 2;
			} else {
				if (used_onset) {
					stage = 1;
					used_onset = false;
					pitch = tune->head_start * 256;
					n_steps++;
				} else if (n_steps > 0)
					pitch += increment;
				else {
					pitch = (tune->head_end * 256) + (pitch_range_abs * tune->head_extend[overflow_ix++])/64;
					if (overflow_ix >= tune->n_head_extend)
						overflow_ix = 0;
				}
			}

			n_steps--;
		}

		if (stress >= PRIMARY) {
			n_unstressed = CountUnstressed(syllable_tab, syl_ix+1, end_ix, secondary);
			unstressed_ix = 0;
			syl->stress = PRIMARY_STRESSED;
			syl->env = tune->stressed_env;
			set_pitch(syl, (pitch / 256), tune->stressed_drop);
		} else if (stress >= secondary) {
			n_unstressed = CountUnstressed(syllable_tab, syl_ix+1, end_ix, secondary);
			unstressed_ix = 0;
			set_pitch(syl, (pitch / 256), drops[stress]);
		} else {
			if (n_unstressed > 1)
				unstressed_inc = (tune->unstr_end[stage] - tune->unstr_start[stage]) / (n_unstressed - 1);
			else
				unstressed_inc = 0;

			set_pitch(syl, (pitch / 256) + tune->unstr_start[stage] + (unstressed_inc * unstressed_ix), drops[stress]);
			unstressed_ix++;
		}

		syl_ix++;
	}
	return syl_ix;
}

/* Calculate pitches until next RESET or tonic syllable, or end.
    Increment pitch if stress is >= min_stress.
    Used for tonic segment */
static int calc_pitch_segment(SYLLABLE *syllable_tab, int ix, int end_ix, const TONE_HEAD *th, const TONE_NUCLEUS *tn, int min_stress, bool continuing)
{
	int stress;
	int pitch = 0;
	int increment = 0;
	int n_primary = 0;
	int n_steps = 0;
	bool initial;
	int overflow = 0;
	int n_overflow;
	int pitch_range;
	int pitch_range_abs;
	const int *drops;
	const signed char *overflow_tab;
	SYLLABLE *syl;

	static const signed char continue_tab[5] = { -26, 32, 20, 8, 0 };

	drops = th->body_drops;
	pitch_range = (th->body_end - th->body_start) * 256;
	pitch_range_abs = abs(pitch_range);

	if (continuing) {
		initial = false;
		overflow = 0;
		n_overflow = 5;
		overflow_tab = continue_tab;
		increment = pitch_range / (th->body_max_steps -1);
	} else {
		n_overflow = th->n_overflow;
		overflow_tab = th->overflow;
		initial = true;
	}

	while (ix < end_ix) {
		syl = &syllable_tab[ix];
		stress = syl->stress;

		if (initial || (stress >= min_stress)) {
			// a primary stress

			if ((initial) || (stress == 5)) {
				initial = false;
				overflow = 0;
				n_steps = n_primary = count_increments(syllable_tab, ix, end_ix, min_stress);

				if (n_steps > th->body_max_steps)
					n_steps = th->body_max_steps;

				if (n_steps > 1)
					increment = pitch_range / (n_steps -1);
				else
					increment = 0;

				pitch = th->body_start * 256;
			} else {
				if (n_steps > 0)
					pitch += increment;
				else {
					pitch = (th->body_end * 256) + (pitch_range_abs * overflow_tab[overflow++])/64;
					if (overflow >= n_overflow) {
						overflow = 0;
						overflow_tab = th->overflow;
					}
				}
			}

			n_steps--;

			n_primary--;
			if ((tn->backwards) && (n_primary < 2))
				pitch = tn->backwards[n_primary] * 256;
		}

		if (stress >= PRIMARY) {
			syl->stress = PRIMARY_STRESSED;
			set_pitch(syl, (pitch / 256), drops[stress]);
		} else if (stress >= SECONDARY)
			set_pitch(syl, (pitch / 256), drops[stress]);
		else {
			// unstressed, drop pitch if preceded by PRIMARY
			if ((syllable_tab[ix-1].stress & 0x3f) >= SECONDARY)
				set_pitch(syl, (pitch / 256) - th->body_lower_u, drops[stress]);
			else
				set_pitch(syl, (pitch / 256), drops[stress]);
		}

		ix++;
	}
	return ix;
}

static void SetPitchGradient(SYLLABLE *syllable_tab, int start_ix, int end_ix, int start_pitch, int end_pitch)
{
	// Set a linear pitch change over a number of syllables.
	// Used for pre-head, unstressed syllables in the body, and the tail

	int ix;
	int stress;
	int pitch;
	int increment;
	int n_increments;
	int drop;
	SYLLABLE *syl;

	increment = (end_pitch - start_pitch) * 256;
	n_increments = end_ix - start_ix;

	if (n_increments <= 0)
		return;

	if (n_increments > 1)
		increment = increment / n_increments;

	pitch = start_pitch * 256;

	for (ix = start_ix; ix < end_ix; ix++) {
		syl = &syllable_tab[ix];
		stress = syl->stress;

		if (increment > 0) {
			set_pitch(syl, (pitch / 256), -(increment / 256));
			pitch += increment;
		} else {
			drop = -(increment / 256);
			if (drop < min_drop[stress])
				drop = min_drop[stress];

			pitch += increment;

			if (drop > 18)
				drop = 18;
			set_pitch(syl, (pitch / 256), drop);
		}
	}
}

// Calculate pitch values for the vowels in this tone group
static int calc_pitches2(SYLLABLE *syllable_tab, int start, int end,  int tune_number)
{
	int ix;
	TUNE *tune;
	int drop;

	tune = &tunes[tune_number];
	ix = start;

	// vowels before the first primary stress

	SetPitchGradient(syllable_tab, ix, ix+number_pre, tune->prehead_start, tune->prehead_end);
	ix += number_pre;

	// body of tonic segment

	if (option_tone_flags & OPTION_EMPHASIZE_PENULTIMATE)
		tone_posn = tone_posn2; // put tone on the penultimate stressed word
	ix = SetHeadIntonation(syllable_tab, tune, ix, tone_posn);

	if (no_tonic)
		return 0;

	// tonic syllable

	if (number_tail == 0) {
		tone_pitch_env = tune->nucleus0_env;
		drop = tune->nucleus0_max - tune->nucleus0_min;
		set_pitch(&syllable_tab[ix++], tune->nucleus0_min, drop);
	} else {
		tone_pitch_env = tune->nucleus1_env;
		drop = tune->nucleus1_max - tune->nucleus1_min;
		set_pitch(&syllable_tab[ix++], tune->nucleus1_min, drop);
	}

	syllable_tab[tone_posn].env = tone_pitch_env;
	if (syllable_tab[tone_posn].stress == PRIMARY)
		syllable_tab[tone_posn].stress = PRIMARY_STRESSED;

	// tail, after the tonic syllable

	SetPitchGradient(syllable_tab, ix, end, tune->tail_start, tune->tail_end);

	return tone_pitch_env;
}

// Calculate pitch values for the vowels in this tone group
static int calc_pitches(SYLLABLE *syllable_tab, int control, int start, int end,  int tune_number)
{
	int ix;
	const TONE_HEAD *th;
	const TONE_NUCLEUS *tn;
	int drop;
	bool continuing = false;

	if (control == 0)
		return calc_pitches2(syllable_tab, start, end, tune_number);

	if (start > 0)
		continuing = true;

	th = &tone_head_table[tune_number];
	tn = &tone_nucleus_table[tune_number];
	ix = start;

	// vowels before the first primary stress

	SetPitchGradient(syllable_tab, ix, ix+number_pre, th->pre_start, th->pre_end);
	ix += number_pre;

	// body of tonic segment

	if (option_tone_flags & OPTION_EMPHASIZE_PENULTIMATE)
		tone_posn = tone_posn2; // put tone on the penultimate stressed word
	ix = calc_pitch_segment(syllable_tab, ix, tone_posn, th, tn, PRIMARY, continuing);

	if (no_tonic)
		return 0;

	// tonic syllable

	if (tn->flags & T_EMPH)
		syllable_tab[ix].flags |= SYL_EMPHASIS;

	if (number_tail == 0) {
		tone_pitch_env = tn->pitch_env0;
		drop = tn->tonic_max0 - tn->tonic_min0;
		set_pitch(&syllable_tab[ix++], tn->tonic_min0, drop);
	} else {
		tone_pitch_env = tn->pitch_env1;
		drop = tn->tonic_max1 - tn->tonic_min1;
		set_pitch(&syllable_tab[ix++], tn->tonic_min1, drop);
	}

	syllable_tab[tone_posn].env = tone_pitch_env;
	if (syllable_tab[tone_posn].stress == PRIMARY)
		syllable_tab[tone_posn].stress = PRIMARY_STRESSED;

	// tail, after the tonic syllable

	SetPitchGradient(syllable_tab, ix, end, tn->tail_start, tn->tail_end);

	return tone_pitch_env;
}

static void CalcPitches_Tone(Translator *tr)
{
	PHONEME_LIST *p;
	int ix;
	int final_stressed = 0;

	int tone_ph;
	bool pause;
	bool tone_promoted;
	PHONEME_TAB *tph;
	PHONEME_TAB *prev_tph; // forget across word boundary
	PHONEME_TAB *prevw_tph; // remember across word boundary
	PHONEME_LIST *prev_p;

	int pitch_adjust = 0;    // pitch gradient through the clause - initial value
	int pitch_decrement = 0; // decrease by this for each stressed syllable
	int pitch_low = 0;       // until it drops to this
	int pitch_high = 0;      // then reset to this

	// count number of stressed syllables
	p = &phoneme_list[0];
	for (ix = 0; ix < n_phoneme_list; ix++, p++) {
		if ((p->type == phVOWEL) && (p->stresslevel >= 4)) {
			final_stressed = ix;
		}
	}

	phoneme_list[final_stressed].stresslevel = 7;

	// language specific, changes to tones
	if (tr->translator_name == L('v', 'i')) {
		// LANG=vi
		p = &phoneme_list[final_stressed];
		if (p->tone_ph == 0)
			p->tone_ph = PhonemeCode('7'); // change default tone (tone 1) to falling tone at end of clause
	}

	pause = true;
	tone_promoted = false;

	prev_p = p = &phoneme_list[0];
	prev_tph = prevw_tph = phoneme_tab[phonPAUSE];

	// perform tone sandhi
	for (ix = 0; ix < n_phoneme_list; ix++, p++) {
		if ((p->type == phPAUSE) && (p->ph->std_length > 50)) {
			pause = true; // there is a pause since the previous vowel
			prevw_tph = phoneme_tab[phonPAUSE]; // forget previous tone
		}

		if (p->newword)
			prev_tph = phoneme_tab[phonPAUSE]; // forget across word boundaries

		if (p->synthflags & SFLAG_SYLLABLE) {
			tone_ph = p->tone_ph;
			tph = phoneme_tab[tone_ph];
			
			/* Hakka
			ref.:https://en.wikipedia.org/wiki/Sixian_dialect#Tone_sandhi */
			if (tr->translator_name == L3('h','a','k')){
				if (prev_tph->mnemonic == 0x31){ // [previous one is 1st tone]
				  // [this one is 1st, 4th, or 6th tone]
				  if (tph->mnemonic == 0x31 || tph->mnemonic == 0x34 ||
					  tph->mnemonic == 0x36){
					/* trigger the tone sandhi of the prev. syllable
					   from 1st tone ->2nd tone */
					prev_p->tone_ph = PhonemeCode('2'); 
				  }
				}
			  }
			// Mandarin
			if (tr->translator_name == L('z', 'h') || tr->translator_name == L3('c', 'm', 'n')) {
				if (tone_ph == 0) {
					if (pause || tone_promoted) {
						tone_ph = PhonemeCode2('5', '5'); // no previous vowel, use tone 1
						tone_promoted = true;
					} else
						tone_ph = PhonemeCode2('1', '1'); // default tone 5

					p->tone_ph = tone_ph;
					tph = phoneme_tab[tone_ph];
				} else
					tone_promoted = false;

				if (ix == final_stressed) {
					if ((tph->mnemonic == 0x3535 ) || (tph->mnemonic == 0x3135)) {
						// change sentence final tone 1 or 4 to stress 6, not 7
						phoneme_list[final_stressed].stresslevel = 6;
					}
				}

				if (prevw_tph->mnemonic == 0x343132) { // [214]
					if (tph->mnemonic == 0x343132) // [214]
						prev_p->tone_ph = PhonemeCode2('3', '5');
					else
						prev_p->tone_ph = PhonemeCode2('2', '1');
				}
				if ((prev_tph->mnemonic == 0x3135)  && (tph->mnemonic == 0x3135)) // [51] + [51]
					prev_p->tone_ph = PhonemeCode2('5', '3');

				if (tph->mnemonic == 0x3131) { // [11] Tone 5
					// tone 5, change its level depending on the previous tone (across word boundaries)
					if (prevw_tph->mnemonic == 0x3535)
						p->tone_ph = PhonemeCode2('2', '2');
					if (prevw_tph->mnemonic == 0x3533)
						p->tone_ph = PhonemeCode2('3', '3');
					if (prevw_tph->mnemonic == 0x343132)
						p->tone_ph = PhonemeCode2('4', '4');

					// tone 5 is unstressed (shorter)
					p->stresslevel = 0; // diminished stress
				}
			}

			prev_p = p;
			prevw_tph = prev_tph = tph;
			pause = false;
		}
	}

	// convert tone numbers to pitch
	p = &phoneme_list[0];
	for (ix = 0; ix < n_phoneme_list; ix++, p++) {
		if (p->synthflags & SFLAG_SYLLABLE) {
			tone_ph = p->tone_ph;

			if (p->stresslevel != 0) { // TEST, consider all syllables as stressed
				if (ix == final_stressed) {
					// the last stressed syllable
					pitch_adjust = pitch_low;
				} else {
					pitch_adjust -= pitch_decrement;
					if (pitch_adjust <= pitch_low)
						pitch_adjust = pitch_high;
				}
			}

			if (tone_ph == 0) {
				tone_ph = phonDEFAULTTONE; // no tone specified, use default tone 1
				p->tone_ph = tone_ph;
			}
			p->pitch1 = pitch_adjust + phoneme_tab[tone_ph]->start_type;
			p->pitch2 = pitch_adjust + phoneme_tab[tone_ph]->end_type;
		}
	}
}

void CalcPitches(Translator *tr, int clause_type)
{
	// clause_type: 0=. 1=, 2=?, 3=! 4=none

	PHONEME_LIST *p;
	SYLLABLE *syl;
	int ix;
	int x;
	int st_ix;
	int n_st;
	int option;
	int group_tone;
	int group_tone_comma;
	int ph_start = 0;
	int st_start;
	int st_clause_end;
	int count;
	int n_primary;
	int count_primary;
	PHONEME_TAB *ph;
	int ph_end = n_phoneme_list;

	SYLLABLE syllable_tab[N_PHONEME_LIST];
	n_st = 0;
	n_primary = 0;
	for (ix = 0; ix < (n_phoneme_list-1); ix++) {
		p = &phoneme_list[ix];
		syllable_tab[ix].flags = 0;
		if (p->synthflags & SFLAG_SYLLABLE) {
			syllable_tab[n_st].env = PITCHfall;
			syllable_tab[n_st].nextph_type = phoneme_list[ix+1].type;
			syllable_tab[n_st++].stress = p->stresslevel;

			if (p->stresslevel >= 4)
				n_primary++;
		} else if ((p->ph->code == phonPAUSE_CLAUSE) && (n_st > 0))
			syllable_tab[n_st-1].flags |= SYL_END_CLAUSE;
	}
	syllable_tab[n_st].stress = 0; // extra 0 entry at the end

	if (n_st == 0)
		return; // nothing to do

	if (tr->langopts.tone_language == 1) {
		CalcPitches_Tone(tr);
		return;
	}

	option = tr->langopts.intonation_group;
	if (option >= INTONATION_TYPES)
		option = 1;

	if (option == 0) {
		group_tone = tr->langopts.tunes[clause_type];
		group_tone_comma = tr->langopts.tunes[1];
	} else {
		group_tone = tr->punct_to_tone[option][clause_type];
		group_tone_comma = tr->punct_to_tone[option][1]; // emphatic form of statement
	}

	if (clause_type == 4)
		no_tonic = 1; // incomplete clause, used for abbreviations such as Mr. Dr. Mrs.
	else
		no_tonic = 0;

	st_start = 0;
	count_primary = 0;
	for (st_ix = 0; st_ix < n_st; st_ix++) {
		syl = &syllable_tab[st_ix];

		if (syl->stress >= 4)
			count_primary++;

		if (syl->stress == 6) {
			// reduce the stress of the previous stressed syllable (review only the previous few syllables)
			for (ix = st_ix-1; ix >= st_start && ix >= (st_ix-3); ix--) {
				if (syllable_tab[ix].stress == 6)
					break;
				if (syllable_tab[ix].stress == 4) {
					syllable_tab[ix].stress = 3;
					break;
				}
			}

			// are the next primary syllables also emphasized ?
			for (ix = st_ix+1; ix < n_st; ix++) {
				if (syllable_tab[ix].stress == 4)
					break;
				if (syllable_tab[ix].stress == 6) {
					// emphasize this syllable, but don't end the current tone group
					syllable_tab[st_ix].flags = SYL_EMPHASIS;
					syl->stress = 5;
					break;
				}
			}
		}

		if (syl->stress == 6) {
			// an emphasized syllable, end the tone group after the next primary stress
			syllable_tab[st_ix].flags = SYL_EMPHASIS;

			count = 0;
			if ((n_primary - count_primary) > 1)
				count = 1;

			for (ix = st_ix+1; ix < n_st; ix++) {
				if (syllable_tab[ix].stress > 4)
					break;
				if (syllable_tab[ix].stress == 4) {
					count++;
					if (count > 1)
						break;
				}
			}

			count_pitch_vowels(syllable_tab, st_start, ix, n_st);
			if ((ix < n_st) || (clause_type == 0)) {
				calc_pitches(syllable_tab, option, st_start, ix, group_tone); // split into > 1 tone groups

				if ((clause_type == 1) || (clause_type == 2))
					group_tone = tr->langopts.tunes[1]; // , or ?  remainder has comma-tone
				else
					group_tone = tr->langopts.tunes[0]; // . or !  remainder has statement tone
			} else
				calc_pitches(syllable_tab, option, st_start, ix, group_tone);

			st_start = ix;
		}
		if ((st_start < st_ix) && (syl->flags & SYL_END_CLAUSE)) {
			// end of clause after this syllable, indicated by a phonPAUSE_CLAUSE phoneme
			st_clause_end = st_ix+1;
			count_pitch_vowels(syllable_tab, st_start, st_clause_end, st_clause_end);
			calc_pitches(syllable_tab, option, st_start, st_clause_end, group_tone_comma);
			st_start = st_clause_end;
		}
	}

	if (st_start < st_ix) {
		count_pitch_vowels(syllable_tab, st_start, st_ix, n_st);
		calc_pitches(syllable_tab, option, st_start, st_ix, group_tone);
	}

	// unpack pitch data
	st_ix = 0;
	for (ix = ph_start; ix < ph_end; ix++) {
		p = &phoneme_list[ix];
		p->stresslevel = syllable_tab[st_ix].stress;

		if (p->synthflags & SFLAG_SYLLABLE) {
			syl = &syllable_tab[st_ix];

			p->pitch1 = syl->pitch1;
			p->pitch2 = syl->pitch2;

			p->env = PITCHfall;
			if (syl->flags & SYL_RISE)
				p->env = PITCHrise;
			else if (p->stresslevel > 5)
				p->env = syl->env;

			if (p->pitch1 > p->pitch2) {
				// swap so that pitch2 is the higher
				x = p->pitch1;
				p->pitch1 = p->pitch2;
				p->pitch2 = x;
			}

			if (p->tone_ph) {
				ph = phoneme_tab[p->tone_ph];
				x = (p->pitch1 + p->pitch2)/2;
				p->pitch2 = x + ph->end_type;
				p->pitch1 = x + ph->start_type;
			}

			if (syl->flags & SYL_EMPHASIS)
				p->stresslevel |= 8; // emphasized

			st_ix++;
		}
	}
}
