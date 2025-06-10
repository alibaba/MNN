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

#ifndef ESPEAK_NG_WAVEGEN_H
#define ESPEAK_NG_WAVEGEN_H

#include "voice.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	int freq;     // Hz<<16
	int height;   // height<<15
	int left;     // Hz<<16
	int right;    // Hz<<16
	double freq1; // floating point versions of the above
	double height1;
	double left1;
	double right1;
	double freq_inc; // increment by this every 64 samples
	double height_inc;
	double left_inc;
	double right_inc;
} wavegen_peaks_t;


int GetAmplitude(void);
void InitBreath(void);
int PeaksToHarmspect(wavegen_peaks_t *peaks,
		int pitch,
		int *htab,
		int control);

void SetPitch2(voice_t *voice,
		int pitch1,
		int pitch2,
		int *pitch_base,
		int *pitch_range);

void WavegenInit(int rate,
		int wavemult_fact);

void WavegenFini(void);


int WavegenFill(void);
void WavegenSetVoice(voice_t *v);
int WcmdqFree(void);
void WcmdqStop(void);
int WcmdqUsed(void);
void WcmdqInc(void);

#ifdef __cplusplus
}
#endif

#endif

