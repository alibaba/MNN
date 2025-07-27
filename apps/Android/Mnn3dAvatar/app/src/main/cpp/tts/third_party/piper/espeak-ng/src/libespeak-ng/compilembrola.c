/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2016 Reece H. Dunn
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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

#include "mbrola.h"

#include "error.h"                // for create_file_error_context
#include "common.h"               // for StringToWord
#include "mbrola.h"               // for MBROLA_TAB
#include "phoneme.h"              // for N_PHONEME_TAB
#include "speech.h"               // for path_home
#include "synthesize.h"           // for Write4Bytes

static const char *basename(const char *filename)
{
	const char *current = filename + strlen(filename);
	while (current != filename && !(*current == '/' || *current == '\\'))
		--current;
	return current == filename ? current : current + 1;
}

#pragma GCC visibility push(default)
espeak_ng_STATUS espeak_ng_CompileMbrolaVoice(const char *filepath, FILE *log, espeak_ng_ERROR_CONTEXT *context)
{
	if (!log) log = stderr;

	char *p;
	FILE *f_in;
	FILE *f_out;
	int percent;
	int n;
	int *pw;
	int *pw_end;
	int count = 0;
	int control;
	char phoneme[40];
	char phoneme2[40];
	char name1[40];
	char name2[40];
	char mbrola_voice[40];
	char buf[sizeof(path_home)+30];
	int mbrola_ctrl = 20; // volume in 1/16 ths
	MBROLA_TAB data[N_PHONEME_TAB];

	if ((f_in = fopen(filepath, "r")) == NULL)
		return create_file_error_context(context, errno, filepath);

	while (fgets(buf, sizeof(phoneme), f_in) != NULL) {
		buf[sizeof(phoneme)-1] = 0;

		if ((p = strstr(buf, "//")) != NULL)
			*p = 0; // truncate line at comment

               if (strncmp(buf, "volume", 6) == 0) {
			mbrola_ctrl = atoi(&buf[6]);
			continue;
		}

		n = sscanf(buf, "%d %s %s %d %s %s", &control, phoneme, phoneme2, &percent, name1, name2);
		if (n >= 5) {
			data[count].name = StringToWord(phoneme);
			if (strcmp(phoneme2, "NULL") == 0)
				data[count].next_phoneme = 0;
			else if (strcmp(phoneme2, "VWL") == 0)
				data[count].next_phoneme = 2;
			else
				data[count].next_phoneme = StringToWord(phoneme2);
			data[count].mbr_name = 0;
			data[count].mbr_name2 = 0;
			data[count].percent = percent;
			data[count].control = control;
			if (strcmp(name1, "NULL") != 0)
				data[count].mbr_name = StringToWord(name1);
			if (n == 6)
				data[count].mbr_name2 = StringToWord(name2);

			count++;
		}
	}
	fclose(f_in);

	strcpy(mbrola_voice, basename(filepath));
	sprintf(buf, "%s/mbrola_ph/%s_phtrans", path_home, mbrola_voice);
	if ((f_out = fopen(buf, "wb")) == NULL)
		return create_file_error_context(context, errno, buf);

	memset(&data[count], 0, sizeof(data[count]));
	data[count].name = 0; // list terminator
	Write4Bytes(f_out, mbrola_ctrl);

	pw_end = (int *)(&data[count+1]);
	for (pw = (int *)data; pw < pw_end; pw++)
		Write4Bytes(f_out, *pw);
	fclose(f_out);
	fprintf(log, "Mbrola translation file: %s -- %d phonemes\n", buf, count);
	return ENS_OK;
}
#pragma GCC visibility pop
