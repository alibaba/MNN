/*
 * Copyright (C) 2008 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2016 Reece H. Dunn
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

// See URL: ftp://svr-ftp.eng.cam.ac.uk/pub/comp.speech/synthesis/klatt.3.04.tar.gz

#include "config.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

#include "klatt.h"
#include "common.h"      // for espeak_rand
#include "synthesize.h"  // for frame_t, WGEN_DATA, STEPSIZE, N_KLATTP, echo...
#include "voice.h"       // for voice_t, N_PEAKS
#define USE_SPEECHPLAYER 0
#if USE_SPEECHPLAYER
#include "sPlayer.h"
#endif

extern unsigned char *out_ptr;
extern unsigned char *out_end;
static int nsamples;
static int sample_count;

#define getrandom(min, max) espeak_rand((min), (max))

// function prototypes for functions private to this file

static void flutter(klatt_frame_ptr);
static double sampled_source(int);
static double impulsive_source(void);
static double natural_source(void);
static void pitch_synch_par_reset(klatt_frame_ptr);
static double gen_noise(double);
static double DBtoLIN(long);
static void frame_init(klatt_frame_ptr);
static void setabc(long, long, resonator_ptr);
static void SetSynth_Klatt(int length, frame_t *fr1, frame_t *fr2, voice_t *v, int control);
static void setzeroabc(long, long, resonator_ptr);

static klatt_frame_t kt_frame;
static klatt_global_t kt_globals;

#define NUMBER_OF_SAMPLES 100

static const int scale_wav_tab[] = { 45, 38, 45, 45, 55, 45 }; // scale output from different voicing sources

// For testing, this can be overwritten in KlattInit()
static const short natural_samples2[256] = {
	 2583,  2516,  2450,  2384,  2319,  2254,  2191,  2127,
	 2067,  2005,  1946,  1890,  1832,  1779,  1726,  1675,
	 1626,  1579,  1533,  1491,  1449,  1409,  1372,  1336,
	 1302,  1271,  1239,  1211,  1184,  1158,  1134,  1111,
	 1089,  1069,  1049,  1031,  1013,   996,   980,   965,
	  950,   936,   921,   909,   895,   881,   869,   855,
	  843,   830,   818,   804,   792,   779,   766,   754,
	  740,   728,   715,   702,   689,   676,   663,   651,
	  637,   626,   612,   601,   588,   576,   564,   552,
	  540,   530,   517,   507,   496,   485,   475,   464,
	  454,   443,   434,   424,   414,   404,   394,   385,
	  375,   366,   355,   347,   336,   328,   317,   308,
	  299,   288,   280,   269,   260,   250,   240,   231,
	  220,   212,   200,   192,   181,   172,   161,   152,
	  142,   133,   123,   113,   105,    94,    86,    76,
	   67,    57,    49,    39,    30,    22,    11,     4,
	   -5,   -14,   -23,   -32,   -41,   -50,   -60,   -69,
	  -78,   -87,   -96,  -107,  -115,  -126,  -134,  -144,
	 -154,  -164,  -174,  -183,  -193,  -203,  -213,  -222,
	 -233,  -242,  -252,  -262,  -271,  -281,  -291,  -301,
	 -310,  -320,  -330,  -339,  -349,  -357,  -368,  -377,
	 -387,  -397,  -406,  -417,  -426,  -436,  -446,  -456,
	 -467,  -477,  -487,  -499,  -509,  -521,  -532,  -543,
	 -555,  -567,  -579,  -591,  -603,  -616,  -628,  -641,
	 -653,  -666,  -679,  -692,  -705,  -717,  -732,  -743,
	 -758,  -769,  -783,  -795,  -808,  -820,  -834,  -845,
	 -860,  -872,  -885,  -898,  -911,  -926,  -939,  -955,
	 -968,  -986,  -999, -1018, -1034, -1054, -1072, -1094,
	-1115, -1138, -1162, -1188, -1215, -1244, -1274, -1307,
	-1340, -1377, -1415, -1453, -1496, -1538, -1584, -1631,
	-1680, -1732, -1783, -1839, -1894, -1952, -2010, -2072,
	-2133, -2196, -2260, -2325, -2390, -2456, -2522, -2589,
};
static const short natural_samples[100] = {
	 -310,  -400,   530,   356,   224,    89,   23,  -10, -58, -16, 461,  599,  536,   701,   770,
	  605,   497,   461,   560,   404,   110,  224,  131, 104, -97, 155,  278, -154, -1165,
	 -598,   737,   125,  -592,    41,    11, -247,  -10,  65,  92,  80, -304,   71,   167,    -1, 122,
	  233,   161,   -43,   278,   479,   485,  407,  266, 650, 134,  80,  236,   68,   260,   269, 179,
	   53,   140,   275,   293,   296,   104,  257,  152, 311, 182, 263,  245,  125,   314,   140, 44,
	  203,   230,  -235,  -286,    23,   107,   92,  -91,  38, 464, 443,  176,   98,  -784, -2449,
	-1891, -1045, -1600, -1462, -1384, -1261, -949, -730
};

/*
   function RESONATOR

   This is a generic resonator function. Internal memory for the resonator
   is stored in the globals structure.
 */

static double resonator(resonator_ptr r, double input)
{
	double x;

	x = (double)((double)r->a * (double)input + (double)r->b * (double)r->p1 + (double)r->c * (double)r->p2);
	r->p2 = (double)r->p1;
	r->p1 = (double)x;

	return (double)x;
}

/*
function ANTIRESONATOR

This is a generic anti-resonator function. The code is the same as resonator
except that a,b,c need to be set with setzeroabc() and we save inputs in
p1/p2 rather than outputs. There is currently only one of these - "rnz"
Output = (rnz.a * input) + (rnz.b * oldin1) + (rnz.c * oldin2)
*/

static double antiresonator(resonator_ptr r, double input)
{
	register double x = (double)r->a * (double)input + (double)r->b * (double)r->p1 + (double)r->c * (double)r->p2;
	r->p2 = (double)r->p1;
	r->p1 = (double)input;
	return (double)x;
}

/*
   function FLUTTER

   This function adds F0 flutter, as specified in:

   "Analysis, synthesis and perception of voice quality variations among
   female and male talkers" D.H. Klatt and L.C. Klatt JASA 87(2) February 1990.

   Flutter is added by applying a quasi-random element constructed from three
   slowly varying sine waves.
 */

static void flutter(klatt_frame_ptr frame)
{
	static int time_count;
	double delta_f0;
	double fla, flb, flc, fld, fle;

	fla = (double)kt_globals.f0_flutter / 50;
	flb = (double)kt_globals.original_f0 / 100;
	flc = sin(M_PI*12.7*time_count); // because we are calling flutter() more frequently, every 2.9mS
	fld = sin(M_PI*7.1*time_count);
	fle = sin(M_PI*4.7*time_count);
	delta_f0 =  fla * flb * (flc + fld + fle) * 10;
	frame->F0hz10 = frame->F0hz10 + (long)delta_f0;
	time_count++;
}

/*
   function SAMPLED_SOURCE

   Allows the use of a glottal excitation waveform sampled from a real
   voice.
 */

static double sampled_source(int source_num)
{
	int itemp;
	double ftemp;
	double result;
	double diff_value;
	int current_value;
	int next_value;
	double temp_diff;
	const short *samples;

	if (source_num == 0) {
		samples = natural_samples;
		kt_globals.num_samples = 100;
	} else {
		samples = natural_samples2;
		kt_globals.num_samples = 256;
	}

	if (kt_globals.T0 != 0) {
		ftemp = (double)kt_globals.nper;
		ftemp = ftemp / kt_globals.T0;
		ftemp = ftemp * kt_globals.num_samples;
		itemp = (int)ftemp;

		temp_diff = ftemp - (double)itemp;

		current_value = samples[(itemp) % kt_globals.num_samples];
		next_value = samples[(itemp+1) % kt_globals.num_samples];

		diff_value = (double)next_value - (double)current_value;
		diff_value = diff_value * temp_diff;

		result = samples[(itemp) % kt_globals.num_samples] + diff_value;
		result = result * kt_globals.sample_factor;
	} else
		result = 0;
	return result;
}

/*
   function PARWAVE

   Converts synthesis parameters to a waveform.
 */

static int parwave(klatt_frame_ptr frame, WGEN_DATA *wdata)
{
	double temp;
	int value;
	double outbypas;
	double out;
	long n4;
	double frics;
	double glotout;
	double aspiration;
	double casc_next_in;
	double par_glotout;
	static double noise;
	static double voice;
	static double vlast;
	static double glotlast;
	static double sourc;
	int ix;

	flutter(frame); // add f0 flutter

	// MAIN LOOP, for each output sample of current frame:

	for (kt_globals.ns = 0; kt_globals.ns < kt_globals.nspfr; kt_globals.ns++) {
		// Get low-passed random number for aspiration and frication noise
		noise = gen_noise(noise);

		// Amplitude modulate noise (reduce noise amplitude during
		// second half of glottal period) if voicing simultaneously present.

		if (kt_globals.nper > kt_globals.nmod)
			noise *= (double)0.5;

		// Compute frication noise
		frics = kt_globals.amp_frica * noise;

		// Compute voicing waveform. Run glottal source simulation at 4
		// times normal sample rate to minimize quantization noise in
		// period of female voice.

		for (n4 = 0; n4 < 4; n4++) {
			switch (kt_globals.glsource)
			{
			case IMPULSIVE:
				voice = impulsive_source();
				break;
			case NATURAL:
				voice = natural_source();
				break;
			case SAMPLED:
				voice = sampled_source(0);
				break;
			case SAMPLED2:
				voice = sampled_source(1);
				break;
			}

			// Reset period when counter 'nper' reaches T0
			if (kt_globals.nper >= kt_globals.T0) {
				kt_globals.nper = 0;
				pitch_synch_par_reset(frame);
			}

			// Low-pass filter voicing waveform before downsampling from 4*samrate
			// to samrate samples/sec.  Resonator f=.09*samrate, bw=.06*samrate

			voice = resonator(&(kt_globals.rsn[RLP]), voice);

			// Increment counter that keeps track of 4*samrate samples per sec
			kt_globals.nper++;
		}

		if(kt_globals.glsource==5) {
			double v=(kt_globals.nper/(double)kt_globals.T0);
			v=(v*2)-1;
			voice=v*6000;
		}

		// Tilt spectrum of voicing source down by soft low-pass filtering, amount
		// of tilt determined by TLTdb

		voice = (voice * kt_globals.onemd) + (vlast * kt_globals.decay);
		vlast = voice;

		// Add breathiness during glottal open phase. Amount of breathiness
		// determined by parameter Aturb Use nrand rather than noise because
		// noise is low-passed.

		if (kt_globals.nper < kt_globals.nopen)
			voice += kt_globals.amp_breth * kt_globals.nrand;

		// Set amplitude of voicing
		glotout = kt_globals.amp_voice * voice;
		par_glotout = kt_globals.par_amp_voice * voice;

		// Compute aspiration amplitude and add to voicing source
		aspiration = kt_globals.amp_aspir * noise;
		glotout += aspiration;

		par_glotout += aspiration;

		// Cascade vocal tract, excited by laryngeal sources.
		// Nasal antiresonator, then formants FNP, F5, F4, F3, F2, F1

		out = 0;
		if (kt_globals.synthesis_model != ALL_PARALLEL) {
			casc_next_in = antiresonator(&(kt_globals.rsn[Rnz]), glotout);
			casc_next_in = resonator(&(kt_globals.rsn[Rnpc]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R8c]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R7c]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R6c]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R5c]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R4c]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R3c]), casc_next_in);
			casc_next_in = resonator(&(kt_globals.rsn[R2c]), casc_next_in);
			out = resonator(&(kt_globals.rsn[R1c]), casc_next_in);
		}

		// Excite parallel F1 and FNP by voicing waveform
		sourc = par_glotout; // Source is voicing plus aspiration

		// Standard parallel vocal tract Formants F6,F5,F4,F3,F2,
		// outputs added with alternating sign. Sound source for other
		// parallel resonators is frication plus first difference of
		// voicing waveform.

		out += resonator(&(kt_globals.rsn[R1p]), sourc);
		out += resonator(&(kt_globals.rsn[Rnpp]), sourc);

		sourc = frics + par_glotout - glotlast;
		glotlast = par_glotout;

		for (ix = R2p; ix <= R6p; ix++)
			out = resonator(&(kt_globals.rsn[ix]), sourc) - out;

		outbypas = kt_globals.amp_bypas * sourc;

		out = outbypas - out;

		out = resonator(&(kt_globals.rsn[Rout]), out);
		temp = (int)(out * wdata->amplitude * kt_globals.amp_gain0); // Convert back to integer

		// mix with a recorded WAV if required for this phoneme
		signed char c;
		int sample;

		if (wdata->mix_wavefile_ix < wdata->n_mix_wavefile) {
			if (wdata->mix_wave_scale == 0) {
				// a 16 bit sample
				c = wdata->mix_wavefile[wdata->mix_wavefile_ix+wdata->mix_wavefile_offset+1];
				sample = wdata->mix_wavefile[wdata->mix_wavefile_ix+wdata->mix_wavefile_offset] + (c * 256);
				wdata->mix_wavefile_ix += 2;
			} else {
				// a 8 bit sample, scaled
				sample = (signed char)wdata->mix_wavefile[wdata->mix_wavefile_offset+wdata->mix_wavefile_ix++] * wdata->mix_wave_scale;
			}
			int z2 = sample * wdata->amplitude_v / 1024;
			z2 = (z2 * wdata->mix_wave_amp)/40;
			temp += z2;

			if ((wdata->mix_wavefile_ix + wdata->mix_wavefile_offset) >= wdata->mix_wavefile_max)  // reached the end of available WAV data
				wdata->mix_wavefile_offset -= (wdata->mix_wavefile_max*3)/4;
		}

		if (kt_globals.fadein < 64) {
			temp = (temp * kt_globals.fadein) / 64;
			++kt_globals.fadein;
		}

		// if fadeout is set, fade to zero over 64 samples, to avoid clicks at end of synthesis
		if (kt_globals.fadeout > 0) {
			kt_globals.fadeout--;
			temp = (temp * kt_globals.fadeout) / 64;
			if (kt_globals.fadeout == 0)
				kt_globals.fadein = 0;
		}

		value = (int)temp + ((echo_buf[echo_tail++]*echo_amp) >> 8);
		if (echo_tail >= N_ECHO_BUF)
			echo_tail = 0;

		if (value < -32768)
			value = -32768;

		if (value > 32767)
			value =  32767;

		*out_ptr++ = value;
		*out_ptr++ = value >> 8;

		echo_buf[echo_head++] = value;
		if (echo_head >= N_ECHO_BUF)
			echo_head = 0;

		sample_count++;
		if (out_ptr + 2 > out_end)
			return 1;
	}
	return 0;
}

void KlattReset(int control)
{
	int r_ix;

#if USE_SPEECHPLAYER
	KlattResetSP();
#endif

	if (control == 2) {
		// Full reset
		kt_globals.FLPhz = (950 * kt_globals.samrate) / 10000;
		kt_globals.BLPhz = (630 * kt_globals.samrate) / 10000;
		kt_globals.minus_pi_t = -M_PI / kt_globals.samrate;
		kt_globals.two_pi_t = -2.0 * kt_globals.minus_pi_t;
		setabc(kt_globals.FLPhz, kt_globals.BLPhz, &(kt_globals.rsn[RLP]));
	}

	if (control > 0) {
		kt_globals.nper = 0;
		kt_globals.T0 = 0;
		kt_globals.nopen = 0;
		kt_globals.nmod = 0;

		for (r_ix = RGL; r_ix < N_RSN; r_ix++) {
			kt_globals.rsn[r_ix].p1 = 0;
			kt_globals.rsn[r_ix].p2 = 0;
		}
	}

	for (r_ix = 0; r_ix <= R6p; r_ix++) {
		kt_globals.rsn[r_ix].p1 = 0;
		kt_globals.rsn[r_ix].p2 = 0;
	}
}

void KlattFini(void)
{
#if USE_SPEECHPLAYER
	KlattFiniSP();
#endif
}

/*
   function FRAME_INIT

   Use parameters from the input frame to set up resonator coefficients.
 */

static void frame_init(klatt_frame_ptr frame)
{
	double amp_par[7];
	static const double amp_par_factor[7] = { 0.6, 0.4, 0.15, 0.06, 0.04, 0.022, 0.03 };
	long Gain0_tmp;
	int ix;

	kt_globals.original_f0 = frame->F0hz10 / 10;

	frame->AVdb_tmp  = frame->AVdb - 7;
	if (frame->AVdb_tmp < 0)
		frame->AVdb_tmp = 0;

	kt_globals.amp_aspir = DBtoLIN(frame->ASP) * 0.05;
	kt_globals.amp_frica = DBtoLIN(frame->AF) * 0.25;
	kt_globals.par_amp_voice = DBtoLIN(frame->AVpdb);
	kt_globals.amp_bypas = DBtoLIN(frame->AB) * 0.05;

	for (ix = 0; ix <= 6; ix++) {
		// parallel amplitudes F1 to F6, and parallel nasal pole
		amp_par[ix] = DBtoLIN(frame->Ap[ix]) * amp_par_factor[ix];
	}

	Gain0_tmp = frame->Gain0 - 3;
	if (Gain0_tmp <= 0)
		Gain0_tmp = 57;
	kt_globals.amp_gain0 = DBtoLIN(Gain0_tmp) / kt_globals.scale_wav;

	// Set coefficients of variable cascade resonators
	for (ix = 1; ix <= 9; ix++) {
		// formants 1 to 8, plus nasal pole
		setabc(frame->Fhz[ix], frame->Bhz[ix], &(kt_globals.rsn[ix]));

		if (ix <= 5) {
			setabc(frame->Fhz_next[ix], frame->Bhz_next[ix], &(kt_globals.rsn_next[ix]));

			kt_globals.rsn[ix].a_inc = (kt_globals.rsn_next[ix].a - kt_globals.rsn[ix].a) / 64.0;
			kt_globals.rsn[ix].b_inc = (kt_globals.rsn_next[ix].b - kt_globals.rsn[ix].b) / 64.0;
			kt_globals.rsn[ix].c_inc = (kt_globals.rsn_next[ix].c - kt_globals.rsn[ix].c) / 64.0;
		}
	}

	// nasal zero anti-resonator
	setzeroabc(frame->Fhz[F_NZ], frame->Bhz[F_NZ], &(kt_globals.rsn[Rnz]));
	setzeroabc(frame->Fhz_next[F_NZ], frame->Bhz_next[F_NZ], &(kt_globals.rsn_next[Rnz]));
	kt_globals.rsn[F_NZ].a_inc = (kt_globals.rsn_next[F_NZ].a - kt_globals.rsn[F_NZ].a) / 64.0;
	kt_globals.rsn[F_NZ].b_inc = (kt_globals.rsn_next[F_NZ].b - kt_globals.rsn[F_NZ].b) / 64.0;
	kt_globals.rsn[F_NZ].c_inc = (kt_globals.rsn_next[F_NZ].c - kt_globals.rsn[F_NZ].c) / 64.0;

	// Set coefficients of parallel resonators, and amplitude of outputs

	for (ix = 0; ix <= 6; ix++) {
		setabc(frame->Fhz[ix], frame->Bphz[ix], &(kt_globals.rsn[Rparallel+ix]));
		kt_globals.rsn[Rparallel+ix].a *= amp_par[ix];
	}

	// output low-pass filter

	setabc((long)0.0, (long)(kt_globals.samrate/2), &(kt_globals.rsn[Rout]));
}

/*
   function IMPULSIVE_SOURCE

   Generate a low pass filtered train of impulses as an approximation of
   a natural excitation waveform. Low-pass filter the differentiated impulse
   with a critically-damped second-order filter, time constant proportional
   to Kopen.
 */

static double impulsive_source(void)
{
	static const double doublet[] = { 0.0, 13000000.0, -13000000.0 };
	static double vwave;

	if (kt_globals.nper < 3)
		vwave = doublet[kt_globals.nper];
	else
		vwave = 0.0;

	return resonator(&(kt_globals.rsn[RGL]), vwave);
}

/*
   function NATURAL_SOURCE

   Vwave is the differentiated glottal flow waveform, there is a weak
   spectral zero around 800 Hz, magic constants a,b reset pitch synchronously.
 */

static double natural_source(void)
{
	double lgtemp;
	static double vwave;

	if (kt_globals.nper < kt_globals.nopen) {
		kt_globals.pulse_shape_a -= kt_globals.pulse_shape_b;
		vwave += kt_globals.pulse_shape_a;
		lgtemp = vwave * 0.028;

		return lgtemp;
	}
	vwave = 0.0;
	return 0.0;
}

/*
   function PITCH_SYNC_PAR_RESET

   Reset selected parameters pitch-synchronously.


   Constant B0 controls shape of glottal pulse as a function
   of desired duration of open phase N0
   (Note that N0 is specified in terms of 40,000 samples/sec of speech)

   Assume voicing waveform V(t) has form: k1 t**2 - k2 t**3

   If the radiation characterivative, a temporal derivative
   is folded in, and we go from continuous time to discrete
   integers n:  dV/dt = vwave[n]
                        = sum over i=1,2,...,n of { a - (i * b) }
                        = a n  -  b/2 n**2

   where the  constants a and b control the detailed shape
   and amplitude of the voicing waveform over the open
   potion of the voicing cycle "nopen".

   Let integral of dV/dt have no net dc flow --> a = (b * nopen) / 3

   Let maximum of dUg(n)/dn be constant --> b = gain / (nopen * nopen)
   meaning as nopen gets bigger, V has bigger peak proportional to n

   Thus, to generate the table below for 40 <= nopen <= 263:

   B0[nopen - 40] = 1920000 / (nopen * nopen)
 */

static void pitch_synch_par_reset(klatt_frame_ptr frame)
{
	long temp;
	double temp1;
	static long skew;
	static const short B0[224] = {
		1200, 1142, 1088, 1038, 991, 948, 907, 869, 833, 799, 768, 738, 710, 683, 658,
		 634,  612,  590,  570, 551, 533, 515, 499, 483, 468, 454, 440, 427, 415, 403,
		 391,  380,  370,  360, 350, 341, 332, 323, 315, 307, 300, 292, 285, 278, 272,
		 265,  259,  253,  247, 242, 237, 231, 226, 221, 217, 212, 208, 204, 199, 195,
		 192,  188,  184,  180, 177, 174, 170, 167, 164, 161, 158, 155, 153, 150, 147,
		 145,  142,  140,  137, 135, 133, 131, 128, 126, 124, 122, 120, 119, 117, 115,
		 113,  111,  110,  108, 106, 105, 103, 102, 100,  99,  97,  96,  95,  93,  92, 91, 90,
		  88,   87,   86,   85,  84,  83,  82,  80,  79,  78,  77,  76,  75,  75,  74, 73, 72, 71,
		  70,   69,   68,   68,  67,  66,  65,  64,  64,  63,  62,  61,  61,  60,  59, 59, 58, 57,
		  57,   56,   56,   55,  55,  54,  54,  53,  53,  52,  52,  51,  51,  50,  50, 49, 49, 48, 48,
		  47,   47,   46,   46,  45,  45,  44,  44,  43,  43,  42,  42,  41,  41,  41, 41, 40, 40,
		  39,   39,   38,   38,  38,  38,  37,  37,  36,  36,  36,  36,  35,  35,  35, 35, 34, 34, 33,
		  33,   33,   33,   32,  32,  32,  32,  31,  31,  31,  31,  30,  30,  30,  30, 29, 29, 29, 29,
		  28,   28,   28,   28,  27,  27
	};

	if (frame->F0hz10 > 0) {
		// T0 is 4* the number of samples in one pitch period

		kt_globals.T0 = (40 * kt_globals.samrate) / frame->F0hz10;

		kt_globals.amp_voice = DBtoLIN(frame->AVdb_tmp);

		// Duration of period before amplitude modulation

		kt_globals.nmod = kt_globals.T0;
		if (frame->AVdb_tmp > 0)
			kt_globals.nmod >>= 1;

		// Breathiness of voicing waveform

		kt_globals.amp_breth = DBtoLIN(frame->Aturb) * 0.1;

		// Set open phase of glottal period where  40 <= open phase <= 263

		kt_globals.nopen = 4 * frame->Kopen;

		if ((kt_globals.glsource == IMPULSIVE) && (kt_globals.nopen > 263))
			kt_globals.nopen = 263;

		if (kt_globals.nopen >= (kt_globals.T0-1))
			kt_globals.nopen = kt_globals.T0 - 2;

		if (kt_globals.nopen < 40) {
			// F0 max = 1000 Hz
			kt_globals.nopen = 40;
		}

		// Reset a & b, which determine shape of "natural" glottal waveform

		kt_globals.pulse_shape_b = B0[kt_globals.nopen-40];
		kt_globals.pulse_shape_a = (kt_globals.pulse_shape_b * kt_globals.nopen) * 0.333;

		// Reset width of "impulsive" glottal pulse

		temp = kt_globals.samrate / kt_globals.nopen;

		setabc((long)0, temp, &(kt_globals.rsn[RGL]));

		// Make gain at F1 about constant

		temp1 = kt_globals.nopen *.00833;
		kt_globals.rsn[RGL].a *= temp1 * temp1;

		// Truncate skewness so as not to exceed duration of closed phase
		// of glottal period.

		temp = kt_globals.T0 - kt_globals.nopen;
		if (frame->Kskew > temp)
			frame->Kskew = temp;
		if (skew >= 0)
			skew = frame->Kskew;
		else
			skew = -frame->Kskew;

		// Add skewness to closed portion of voicing period
		kt_globals.T0 = kt_globals.T0 + skew;
		skew = -skew;
	} else {
		kt_globals.T0 = 4; // Default for f0 undefined
		kt_globals.amp_voice = 0.0;
		kt_globals.nmod = kt_globals.T0;
		kt_globals.amp_breth = 0.0;
		kt_globals.pulse_shape_a = 0.0;
		kt_globals.pulse_shape_b = 0.0;
	}

	// Reset these pars pitch synchronously or at update rate if f0=0

	if ((kt_globals.T0 != 4) || (kt_globals.ns == 0)) {
		// Set one-pole low-pass filter that tilts glottal source

		kt_globals.decay = (0.033 * frame->TLTdb);

		if (kt_globals.decay > 0.0)
			kt_globals.onemd = 1.0 - kt_globals.decay;
		else
			kt_globals.onemd = 1.0;
	}
}

/*
   function SETABC

   Convert formant frequencies and bandwidth into resonator difference
   equation constants.
 */

static void setabc(long int f, long int bw, resonator_ptr rp)
{
	double r;
	double arg;

	// Let r  =  exp(-pi bw t)
	arg = kt_globals.minus_pi_t * bw;
	r = exp(arg);

	// Let c  =  -r**2
	rp->c = -(r * r);

	// Let b = r * 2*cos(2 pi f t)
	arg = kt_globals.two_pi_t * f;
	rp->b = r * cos(arg) * 2.0;

	// Let a = 1.0 - b - c
	rp->a = 1.0 - rp->b - rp->c;
}

/*
   function SETZEROABC

   Convert formant frequencies and bandwidth into anti-resonator difference
   equation constants.
 */

static void setzeroabc(long int f, long int bw, resonator_ptr rp)
{
	double r;
	double arg;

	f = -f;

	// First compute ordinary resonator coefficients
	// Let r  =  exp(-pi bw t)
	arg = kt_globals.minus_pi_t * bw;
	r = exp(arg);

	// Let c  =  -r**2
	rp->c = -(r * r);

	// Let b = r * 2*cos(2 pi f t)
	arg = kt_globals.two_pi_t * f;
	rp->b = r * cos(arg) * 2.;

	// Let a = 1.0 - b - c
	rp->a = 1.0 - rp->b - rp->c;

	// Now convert to antiresonator coefficients (a'=1/a, b'=b/a, c'=c/a)

	// If f == 0 then rp->a gets set to 0 which makes a'=1/a set a', b' and c' to
	// INF, causing an audible sound spike when triggered (e.g. apiration with the
	// nasal register set to f=0, bw=0).
	if (rp->a != 0) {
		// Now convert to antiresonator coefficients (a'=1/a, b'=b/a, c'=c/a)
		rp->a = 1.0 / rp->a;
		rp->c *= -rp->a;
		rp->b *= -rp->a;
	}
}

/*
   function GEN_NOISE

   Random number generator (return a number between -8191 and +8191)
   Noise spectrum is tilted down by soft low-pass filter having a pole near
   the origin in the z-plane, i.e. output = input + (0.75 * lastoutput)
 */

static double gen_noise(double noise)
{
	long temp;
	static double nlast;

	temp = (long)getrandom(-8191, 8191);
	kt_globals.nrand = (long)temp;

	noise = kt_globals.nrand + (0.75 * nlast);
	nlast = noise;

	return noise;
}

/*
   function DBTOLIN

   Convert from decibels to a linear scale factor


   Conversion table, db to linear, 87 dB --> 32767
                                86 dB --> 29491 (1 dB down = 0.5**1/6)
                                 ...
                                81 dB --> 16384 (6 dB down = 0.5)
                                 ...
                                 0 dB -->     0

   The just noticeable difference for a change in intensity of a vowel
   is approximately 1 dB.  Thus all amplitudes are quantized to 1 dB
   steps.
 */

static double DBtoLIN(long dB)
{
	static const short amptable[88] = {
		   0,      0,     0,     0,     0,     0,     0,    0,     0,    0,   0,   0,  0, 6, 7,
		   8,      9,    10,    11,    13,    14,    16,   18,    20,   22,  25,  28, 32,
		   35,    40,    45,    51,    57,    64,    71,   80,    90,  101, 114, 128,
		  142,   159,   179,   202,   227,   256,   284,  318,   359,  405,
		  455,   512,   568,   638,   719,   881,   911, 1024,  1137, 1276,
		 1438,  1622,  1823,  2048,  2273,  2552,  2875, 3244,  3645,
		 4096,  4547,  5104,  5751,  6488,  7291,  8192, 9093, 10207,
		11502, 12976, 14582, 16384, 18350, 20644, 23429,
		26214, 29491, 32767
	};

	if ((dB < 0) || (dB > 87))
		return 0;

	return (double)(amptable[dB]) * 0.001;
}

static klatt_peaks_t peaks[N_PEAKS];
static int end_wave;
static int klattp[N_KLATTP];
static double klattp1[N_KLATTP];
static double klattp_inc[N_KLATTP];

int Wavegen_Klatt(int length, int resume, frame_t *fr1, frame_t *fr2, WGEN_DATA *wdata, voice_t *wvoice)
{
#if USE_SPEECHPLAYER
	if(wvoice->klattv[0] == 6)
	return Wavegen_KlattSP(wdata, wvoice, length, resume, fr1, fr2);
#endif

	if (resume == 0)
		SetSynth_Klatt(length, fr1, fr2, wvoice, 1);

	int pk;
	int x;
	int ix;
	int fade;

	if (resume == 0)
		sample_count = 0;

	while (sample_count < nsamples) {
		kt_frame.F0hz10 = (wdata->pitch * 10) / 4096;

		// formants F6,F7,F8 are fixed values for cascade resonators, set in KlattInit()
		// but F6 is used for parallel resonator
		// F0 is used for the nasal zero
		for (ix = 0; ix < 6; ix++) {
			kt_frame.Fhz[ix] = peaks[ix].freq;
			if (ix < 4)
				kt_frame.Bhz[ix] = peaks[ix].bw;
		}
		for (ix = 1; ix < 7; ix++)
			kt_frame.Ap[ix] = peaks[ix].ap;

		kt_frame.AVdb = klattp[KLATT_AV];
		kt_frame.AVpdb = klattp[KLATT_AVp];
		kt_frame.AF = klattp[KLATT_Fric];
		kt_frame.AB = klattp[KLATT_FricBP];
		kt_frame.ASP = klattp[KLATT_Aspr];
		kt_frame.Aturb = klattp[KLATT_Turb];
		kt_frame.Kskew = klattp[KLATT_Skew];
		kt_frame.TLTdb = klattp[KLATT_Tilt];
		kt_frame.Kopen = klattp[KLATT_Kopen];

		// advance formants
		for (pk = 0; pk < N_PEAKS; pk++) {
			peaks[pk].freq1 += peaks[pk].freq_inc;
			peaks[pk].freq = (int)peaks[pk].freq1;
			peaks[pk].bw1 += peaks[pk].bw_inc;
			peaks[pk].bw = (int)peaks[pk].bw1;
			peaks[pk].bp1 += peaks[pk].bp_inc;
			peaks[pk].bp = (int)peaks[pk].bp1;
			peaks[pk].ap1 += peaks[pk].ap_inc;
			peaks[pk].ap = (int)peaks[pk].ap1;
		}

		// advance other parameters
		for (ix = 0; ix < N_KLATTP; ix++) {
			klattp1[ix] += klattp_inc[ix];
			klattp[ix] = (int)klattp1[ix];
		}

		for (ix = 0; ix <= 6; ix++) {
			kt_frame.Fhz_next[ix] = peaks[ix].freq;
			if (ix < 4)
				kt_frame.Bhz_next[ix] = peaks[ix].bw;
		}

		// advance the pitch
		wdata->pitch_ix += wdata->pitch_inc;
		if ((ix = wdata->pitch_ix>>8) > 127) ix = 127;
		x = wdata->pitch_env[ix] * wdata->pitch_range;
		wdata->pitch = (x>>8) + wdata->pitch_base;

		kt_globals.nspfr = (nsamples - sample_count);
		if (kt_globals.nspfr > STEPSIZE)
			kt_globals.nspfr = STEPSIZE;

		frame_init(&kt_frame); // get parameters for next frame of speech

		if (parwave(&kt_frame, wdata) == 1)
			return 1; // output buffer is full
	}

	if (end_wave > 0) {
		fade = 64; // not followed by formant synthesis

		// fade out to avoid a click
		kt_globals.fadeout = fade;
		end_wave = 0;
		sample_count -= fade;
		kt_globals.nspfr = fade;
		if (parwave(&kt_frame, wdata) == 1)
			return 1; // output buffer is full
	}

	return 0;
}

static void SetSynth_Klatt(int length, frame_t *fr1, frame_t *fr2, voice_t *wvoice, int control)
{
	int ix;
	double next;
	int qix;
	int cmd;
	frame_t *fr3;
	static frame_t prev_fr;

	if (wvoice != NULL) {
		if ((wvoice->klattv[0] > 0) && (wvoice->klattv[0] <= 5 )) {
			kt_globals.glsource = wvoice->klattv[0];
			kt_globals.scale_wav = scale_wav_tab[kt_globals.glsource];
		}
		kt_globals.f0_flutter = wvoice->flutter/32;
	}

	end_wave = 0;
	if (control & 2)
		end_wave = 1; // fadeout at the end
	if (control & 1) {
		end_wave = 1;
		for (qix = wcmdq_head+1;; qix++) {
			if (qix >= N_WCMDQ) qix = 0;
			if (qix == wcmdq_tail) break;

			cmd = wcmdq[qix][0];
			if (cmd == WCMD_KLATT) {
				end_wave = 0; // next wave generation is from another spectrum

				fr3 = (frame_t *)wcmdq[qix][2];
				for (ix = 1; ix < 6; ix++) {
					if (fr3->ffreq[ix] != fr2->ffreq[ix]) {
						// there is a discontinuity in formants
						end_wave = 2;
						break;
					}
				}
				break;
			}
			if ((cmd == WCMD_WAVE) || (cmd == WCMD_PAUSE))
				break; // next is not from spectrum, so continue until end of wave cycle
		}

		for (ix = 1; ix < 6; ix++) {
			if (prev_fr.ffreq[ix] != fr1->ffreq[ix]) {
				// Discontinuity in formants.
				// end_wave was set in SetSynth_Klatt() to fade out the previous frame
				KlattReset(0);
				break;
			}
		}
		memcpy(&prev_fr, fr2, sizeof(prev_fr));
	}

	for (ix = 0; ix < N_KLATTP; ix++) {
		if ((ix >= 5) || ((fr1->frflags & FRFLAG_KLATT) == 0)) {
			klattp1[ix] = klattp[ix] = 0;
			klattp_inc[ix] = 0;
		} else {
			klattp1[ix] = klattp[ix] = fr1->klattp[ix];
			klattp_inc[ix] = (double)((fr2->klattp[ix] - klattp[ix]) * STEPSIZE)/length;
		}
	}

	nsamples = length;

	for (ix = 1; ix < 6; ix++) {
		peaks[ix].freq1 = (fr1->ffreq[ix] * wvoice->freq[ix] / 256.0) + wvoice->freqadd[ix];
		peaks[ix].freq = (int)peaks[ix].freq1;
		next = (fr2->ffreq[ix] * wvoice->freq[ix] / 256.0) + wvoice->freqadd[ix];
		peaks[ix].freq_inc =  ((next - peaks[ix].freq1) * STEPSIZE) / length;

		if (ix < 4) {
			// klatt bandwidth for f1, f2, f3 (others are fixed)
			peaks[ix].bw1 = fr1->bw[ix] * 2  * (wvoice->width[ix] / 256.0);
			peaks[ix].bw = (int)peaks[ix].bw1;
			next = fr2->bw[ix] * 2;
			peaks[ix].bw_inc =  ((next - peaks[ix].bw1) * STEPSIZE) / length;
		}
	}

	// nasal zero frequency
	peaks[0].freq1 = fr1->klattp[KLATT_FNZ] * 2;
	if (peaks[0].freq1 == 0)
		peaks[0].freq1 = kt_frame.Fhz[F_NP]; // if no nasal zero, set it to same freq as nasal pole

	peaks[0].freq = (int)peaks[0].freq1;
	next = fr2->klattp[KLATT_FNZ] * 2;
	if (next == 0)
		next = kt_frame.Fhz[F_NP];

	peaks[0].freq_inc = ((next - peaks[0].freq1) * STEPSIZE) / length;

	peaks[0].bw1 = 89;
	peaks[0].bw = 89;
	peaks[0].bw_inc = 0;

	if (fr1->frflags & FRFLAG_KLATT) {
		// the frame contains additional parameters for parallel resonators
		for (ix = 1; ix < 7; ix++) {
			peaks[ix].bp1 = fr1->klatt_bp[ix] * 4; // parallel bandwidth
			peaks[ix].bp = (int)peaks[ix].bp1;
			next = fr2->klatt_bp[ix] * 4;
			peaks[ix].bp_inc =  ((next - peaks[ix].bp1) * STEPSIZE) / length;

			peaks[ix].ap1 = fr1->klatt_ap[ix]; // parallal amplitude
			peaks[ix].ap = (int)peaks[ix].ap1;
			next = fr2->klatt_ap[ix];
			peaks[ix].ap_inc =  ((next - peaks[ix].ap1) * STEPSIZE) / length;
		}
	}
}

void KlattInit(void)
{

	static const short formant_hz[10] = { 280, 688, 1064, 2806, 3260, 3700, 6500, 7000, 8000, 280 };
	static const short bandwidth[10] = { 89, 160, 70, 160, 200, 200, 500, 500, 500, 89 };
	static const short parallel_amp[10] = { 0, 59, 59, 59, 59, 59, 59, 0, 0, 0 };
	static const short parallel_bw[10] = { 59, 59, 89, 149, 200, 200, 500, 0, 0, 0 };

	int ix;

#if USE_SPEECHPLAYER
	KlattInitSP();
#endif

	sample_count = 0;

	kt_globals.synthesis_model = CASCADE_PARALLEL;
	kt_globals.samrate = 22050;

	kt_globals.glsource = IMPULSIVE;
	kt_globals.scale_wav = scale_wav_tab[kt_globals.glsource];
	kt_globals.natural_samples = natural_samples;
	kt_globals.num_samples = NUMBER_OF_SAMPLES;
	kt_globals.sample_factor = 3.0;
	kt_globals.nspfr = (kt_globals.samrate * 10) / 1000;
	kt_globals.outsl = 0;
	kt_globals.f0_flutter = 20;

	KlattReset(2);

	// set default values for frame parameters
	for (ix = 0; ix <= 9; ix++) {
		kt_frame.Fhz[ix] = formant_hz[ix];
		kt_frame.Bhz[ix] = bandwidth[ix];
		kt_frame.Ap[ix] = parallel_amp[ix];
		kt_frame.Bphz[ix] = parallel_bw[ix];
	}
	kt_frame.Bhz_next[F_NZ] = bandwidth[F_NZ];

	kt_frame.F0hz10 = 1000;
	kt_frame.AVdb = 59;
	kt_frame.ASP = 0;
	kt_frame.Kopen = 40;
	kt_frame.Aturb = 0;
	kt_frame.TLTdb = 0;
	kt_frame.AF = 50;
	kt_frame.Kskew = 0;
	kt_frame.AB = 0;
	kt_frame.AVpdb = 0;
	kt_frame.Gain0 = 62;
}
