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
#ifndef ESPEAK_NG_READCLAUSE_H
#define ESPEAK_NG_READCLAUSE_H

#include "translate.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	int type;
	int parameter[N_SPEECH_PARAM];
} PARAM_STACK;

extern PARAM_STACK param_stack[];

int clause_type_from_codepoint(uint32_t c);
int Eof(void);
const char *WordToString2(char buf[5], unsigned int word);
int AddNameData(const char *name,
                int wide);
int ReadClause(Translator *tr,
		char *buf,
		short *charix,
		int *charix_top,
		int n_buf,
		int *tone_type,
		char *voice_change);




#ifdef __cplusplus
}
#endif

#endif

