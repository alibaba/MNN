/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2017 Reece H. Dunn
 * Copyright (C) 2021 Juho Hiltunen
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

#ifndef ESPEAK_NG_SOUNDICON_H
#define ESPEAK_NG_SOUNDICON_H

#ifdef __cplusplus
extern "C"
{
#endif

int LookupSoundicon(int c);
int LoadSoundFile2(const char *fname);

typedef struct {
        int name; // used for detecting punctuation
        int length;
        char *data;
        char *filename;
} SOUND_ICON;

#define N_SOUNDICON_TAB  80   // total entries for dynamic loading of audio files

extern int n_soundicon_tab;
extern SOUND_ICON soundicon_tab[N_SOUNDICON_TAB];


#ifdef __cplusplus
}
#endif

#endif
