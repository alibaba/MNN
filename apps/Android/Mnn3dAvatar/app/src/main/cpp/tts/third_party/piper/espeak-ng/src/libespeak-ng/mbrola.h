/*
 * Copyright (C) 2005 to 2015 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2018 Reece H. Dunn
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

// declarations for compilembrola.c and synth_mbrola.c

#ifndef ESPEAK_NG_MBROLA_H
#define ESPEAK_NG_MBROLA_H

#include <stdbool.h>

#include "synthesize.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
        int name;
        unsigned int next_phoneme;
        int mbr_name;
        int mbr_name2;
        int percent; // percentage length of first component
        int control;
} MBROLA_TAB;

extern int mbrola_delay;
extern char mbrola_name[20];

espeak_ng_STATUS LoadMbrolaTable(const char *mbrola_voice,
		const char *phtrans, 
		int *srate);

int MbrolaGenerate(PHONEME_LIST *phoneme_list,
		int *n_ph, bool resume);

int MbrolaFill(int length,
		bool resume,
		int amplitude);

void MbrolaReset(void);
int MbrolaTranslate(PHONEME_LIST *plist, int n_phonemes, bool resume, FILE *f_mbrola);

#ifdef __cplusplus
}
#endif

#endif

