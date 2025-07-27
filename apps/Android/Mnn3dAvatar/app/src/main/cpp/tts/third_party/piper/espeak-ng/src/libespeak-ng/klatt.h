/*
 * Copyright (C) 2008 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2015-2017 Reece H. Dunn
 *
 * Based on a re-implementation by:
 * (c) 1993,94 Jon Iles and Nick Ing-Simmons
 * of the Klatt cascade-parallel formant synthesizer
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

#ifndef ESPEAK_NG_KLATT_H
#define ESPEAK_NG_KLATT_H

#include "voice.h"                    // for voice_t
#include "synthesize.h"              // for frame_t, WGEN_DATA

#ifdef __cplusplus
extern "C"
{
#endif

#define CASCADE_PARALLEL 1         /* Type of synthesis model */
#define ALL_PARALLEL     2

#define IMPULSIVE        1         /* Type of voicing source */
#define NATURAL          2
#define SAMPLED          3
#define SAMPLED2         4

/* typedef's that need to be exported */

typedef long flag;

/* Resonator Structure */

typedef struct {
	double a;
	double b;
	double c;
	double p1;
	double p2;
	double a_inc;
	double b_inc;
	double c_inc;
} resonator_t, *resonator_ptr;

/* Structure for Klatt Globals */

typedef struct {
	flag synthesis_model; /* cascade-parallel or all-parallel */
	flag outsl;     /* Output waveform selector                      */
	long samrate;   /* Number of output samples per second           */
	long FLPhz;     /* Frequeny of glottal downsample low-pass filter */
	long BLPhz;     /* Bandwidth of glottal downsample low-pass filter */
	flag glsource;  /* Type of glottal source */
	int f0_flutter; /* Percentage of f0 flutter 0-100 */
	long nspfr;     /* number of samples per frame */
	long nper;      /* Counter for number of samples in a pitch period */
	long ns;
	long T0;        /* Fundamental period in output samples times 4 */
	long nopen;     /* Number of samples in open phase of period    */
	long nmod;      /* Position in period to begin noise amp. modul */
	long nrand;     /* Variable used by random number generator      */
	double pulse_shape_a; /* Makes waveshape of glottal pulse when open   */
	double pulse_shape_b; /* Makes waveshape of glottal pulse when open   */
	double minus_pi_t;
	double two_pi_t;
	double onemd;
	double decay;
	double amp_bypas; /* AB converted to linear gain              */
	double amp_voice; /* AVdb converted to linear gain            */
	double par_amp_voice; /* AVpdb converted to linear gain       */
	double amp_aspir; /* AP converted to linear gain              */
	double amp_frica; /* AF converted to linear gain              */
	double amp_breth; /* ATURB converted to linear gain           */
	double amp_gain0; /* G0 converted to linear gain              */
	int num_samples; /* number of glottal samples */
	double sample_factor; /* multiplication factor for glottal samples */
	const short *natural_samples; /* pointer to an array of glottal samples */
	long original_f0; /* original value of f0 not modified by flutter */

	int fadein;
	int fadeout;       // set to 64 to cause fadeout over 64 samples
	int scale_wav;     // depends on the voicing source

#define N_RSN 20
#define Rnz  0   // nasal zero, anti-resonator
#define R1c  1
#define R2c  2
#define R3c  3
#define R4c  4
#define R5c  5
#define R6c  6
#define R7c  7
#define R8c  8
#define Rnpc 9   // nasal pole

#define Rparallel 10
#define Rnpp 10
#define R1p  11
#define R2p  12
#define R6p  16

#define RGL  17
#define RLP  18
#define Rout 19

	resonator_t rsn[N_RSN];  // internal storage for resonators
	resonator_t rsn_next[N_RSN];

} klatt_global_t, *klatt_global_ptr;

/* Structure for Klatt Parameters */

#define F_NZ   0  // nasal zero formant
#define F1     1
#define F2     2
#define F3     3
#define F4     4
#define F5     5
#define F6     6
#define F_NP   9  // nasal pole formant


typedef struct {
	int F0hz10; /* Voicing fund freq in Hz                          */
	int AVdb;   /* Amp of voicing in dB,            0 to   70       */
	int Fhz[10];  // formant Hz, F_NZ to F6 to F_NP
	int Bhz[10];
	int Ap[10];   /* Amp of parallel formants in dB,    0 to   80       */
	int Bphz[10]; /* Parallel formants bw in Hz,       40 to 1000      */

	int ASP;    /* Amp of aspiration in dB,         0 to   70       */
	int Kopen;  /* # of samples in open period,     10 to   65      */
	int Aturb;  /* Breathiness in voicing,          0 to   80       */
	int TLTdb;  /* Voicing spectral tilt in dB,     0 to   24       */
	int AF;     /* Amp of frication in dB,          0 to   80       */
	int Kskew;  /* Skewness of alternate periods,   0 to   40 in sample#/2  */

	int AB;     /* Amp of bypass fric. in dB,       0 to   80       */
	int AVpdb;  /* Amp of voicing,  par in dB,      0 to   70       */
	int Gain0;  /* Overall gain, 60 dB is unity,    0 to   60       */

	int AVdb_tmp;      // copy of AVdb, which is changed within parwave()
	int Fhz_next[10];    // Fhz for the next chunk, so we can do interpolation of resonator (a,b,c) parameters
	int Bhz_next[10];
} klatt_frame_t, *klatt_frame_ptr;


typedef struct {
	int freq;     // Hz
	int bw;   // klatt bandwidth
	int ap;   // parallel amplitude
	int bp;   // parallel bandwidth
	double freq1; // floating point versions of the above
	double bw1;
	double ap1;
	double bp1;
	double freq_inc;    // increment by this every 64 samples
	double bw_inc;
	double ap_inc;
	double bp_inc;
}  klatt_peaks_t;

void KlattInit(void);
void KlattFini(void);
void KlattReset(int control);
int Wavegen_Klatt(int length, int resume, frame_t *fr1, frame_t *fr2, WGEN_DATA *wdata, voice_t *wvoice);

#ifdef __cplusplus
}
#endif

#endif
