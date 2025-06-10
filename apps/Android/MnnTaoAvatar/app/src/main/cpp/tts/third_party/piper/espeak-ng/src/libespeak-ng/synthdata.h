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

#ifndef ESPEAK_NG_SYNTHDATA_H
#define ESPEAK_NG_SYNTHDATA_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "espeak-ng/espeak_ng.h"       // for espeak_ng_ERROR_CONTEXT, espea...
#include "phoneme.h"                   // for PHONEME_TAB
#include "synthesize.h"                // for PHONEME_DATA, PHONEME_LIST
#include "translate.h"                 // for Translator

void InterpretPhoneme(Translator *tr,
		int control,
		PHONEME_LIST *plist,
		PHONEME_LIST *plist_start,
		PHONEME_DATA *phdata,
		WORD_PH_DATA *worddata);

void InterpretPhoneme2(int phcode,
		PHONEME_DATA *phdata);

void FreePhData(void);
const unsigned char *GetEnvelope(int index);
espeak_ng_STATUS LoadPhData(int *srate, espeak_ng_ERROR_CONTEXT *context);
int LookupPhonemeString(const char *string);
int LookupPhonemeTable(const char *name);
frameref_t *LookupSpect(PHONEME_TAB *this_ph,
		int which,
		FMT_PARAMS *fmt_params,
		int *n_frames,
		PHONEME_LIST *plist);

int PhonemeCode(unsigned int mnem);
void SelectPhonemeTable(int number);
int  SelectPhonemeTableName(const char *name);

extern int n_tunes;
extern TUNE *tunes;

#ifdef __cplusplus
}
#endif

#endif

