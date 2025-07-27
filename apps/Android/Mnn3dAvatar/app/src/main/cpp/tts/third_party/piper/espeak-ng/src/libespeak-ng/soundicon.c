/*
 * Copyright (C) 2005 to 2015 by Jonathan Duddington
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

#include "config.h"

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>
#include <ucd/ucd.h>

#include "soundicon.h" 
#include "common.h"                // for GetFileLength
#include "error.h"                // for create_file_error_context
#include "speech.h"                   // for path_home, PATHSEP
#include "synthesize.h"                   // for samplerate

int n_soundicon_tab = 0;
SOUND_ICON soundicon_tab[N_SOUNDICON_TAB];


static espeak_ng_STATUS LoadSoundFile(const char *fname, int index, espeak_ng_ERROR_CONTEXT *context)
{
	FILE *f;
	unsigned char *p;
	int length;
	char fname_temp[100];
	char fname2[sizeof(path_home)+13+40];

	if (fname == NULL) {
		// filename is already in the table
		fname = soundicon_tab[index].filename;
	}

	if (fname == NULL)
		return EINVAL;

	if (fname[0] != '/') {
		// a relative path, look in espeak-ng-data/soundicons
		sprintf(fname2, "%s%csoundicons%c%s", path_home, PATHSEP, PATHSEP, fname);
		fname = fname2;
	}

	fname_temp[0] = 0;

	f = NULL;
	if ((f = fopen(fname, "rb")) != NULL) {
		int ix;
		int header[3];

		if (fseek(f, 20, SEEK_SET) == -1) {
			int error = errno;
			fclose(f);
			return create_file_error_context(context, error, fname);
		}

		for (ix = 0; ix < 3; ix++)
			header[ix] = Read4Bytes(f);

		// if the sound file is not mono, 16 bit signed, at the correct sample rate, then convert it
		if ((header[0] != 0x10001) || (header[1] != samplerate) || (header[2] != samplerate*2)) {
			fclose(f);
			f = NULL;

#if HAVE_MKSTEMP
			strcpy(fname_temp, "/tmp/espeakXXXXXX");
			int fd_temp;
			if ((fd_temp = mkstemp(fname_temp)) >= 0)
				close(fd_temp);
#else
			strcpy(fname_temp, tmpnam(NULL));
#endif

//			sprintf(command, "sox \"%s\" -r %d -c1 -b 16 -t wav %s\n", fname, samplerate, fname_temp);
//			if (system(command) == 0)
				fname = fname_temp;
		}
	}

	if (f == NULL) {
		f = fopen(fname, "rb");
		if (f == NULL)
			return create_file_error_context(context, errno, fname);
	}

	length = GetFileLength(fname);
	if (length < 0) { // length == -errno
		fclose(f);
		return create_file_error_context(context, -length, fname);
	}
	if (fseek(f, 0, SEEK_SET) == -1) {
		int error = errno;
		fclose(f);
		return create_file_error_context(context, error, fname);
	}
	if ((p = realloc(soundicon_tab[index].data, length)) == NULL) {
		fclose(f);
		return ENOMEM;
	}
	if (fread(p, 1, length, f) != length) {
		int error = errno;
		fclose(f);
		if (fname_temp[0])
			remove(fname_temp);
		free(p);
		return create_file_error_context(context, error, fname);
	}
	fclose(f);
	if (fname_temp[0])
		remove(fname_temp);

	length = p[40] | (p[41] << 8) | (p[42] << 16) | (p[43] << 24);
	soundicon_tab[index].length = length / 2; // length in samples
	soundicon_tab[index].data = (char *) p;
	return ENS_OK;
}

int LookupSoundicon(int c)
{
	// Find the sound icon number for a punctuation character and load the audio file if it's not yet loaded
	int ix;

	for (ix = 0; ix < n_soundicon_tab; ix++) {
		if (soundicon_tab[ix].name == c) {
			if (soundicon_tab[ix].length == 0) { // not yet loaded, load now
				if (LoadSoundFile(NULL, ix, NULL) != ENS_OK) {
					return -1; // sound file is not available
				}
			}
			return ix;
		}
	}
	return -1;
}

int LoadSoundFile2(const char *fname)
{
	// Load a sound file into the sound icon table and memory
	// (if it's not already loaded)
	// returns -1 on error or the index of loaded file on success

	int ix;
	for (ix = 0; ix < n_soundicon_tab; ix++) {
		if (((soundicon_tab[ix].filename != NULL) && strcmp(fname, soundicon_tab[ix].filename) == 0)) {
			// the file information is found. If length = 0 it needs to be loaded to memory
			if (soundicon_tab[ix].length == 0) {
				if (LoadSoundFile(NULL, ix, NULL) != ENS_OK)
					return -1; // sound file is not available
			}
			return ix; // sound file already loaded to memory
		}
	}

	// load the file into the current slot and increase index
	if (LoadSoundFile(fname, n_soundicon_tab, NULL) != ENS_OK)
		return -1;

	soundicon_tab[n_soundicon_tab].filename = (char *)realloc(soundicon_tab[n_soundicon_tab].filename, strlen(fname)+1);
	strcpy(soundicon_tab[n_soundicon_tab].filename, fname);
	n_soundicon_tab++;
	return n_soundicon_tab - 1;
}
