/*
 * Copyright (C) 2005 to 2013 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2017 Reece H. Dunn
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

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>

#if USE_LIBPCAUDIO
#include <pcaudiolib/audio.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <winreg.h>
#endif

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "speech.h"
#include "common.h"               // for GetFileLength
#include "dictionary.h"           // for GetTranslatedPhonemeString, strncpy0
#include "espeak_command.h"       // for delete_espeak_command, SetParameter
#include "event.h"                // for event_declare, event_clear_all, eve...
#include "fifo.h"                 // for fifo_add_command, fifo_add_commands
#include "langopts.h"             // for LoadConfig
#include "mbrola.h"               // for mbrola_delay
#include "readclause.h"           // for PARAM_STACK, param_stack
#include "synthdata.h"            // for FreePhData, LoadPhData
#include "synthesize.h"           // for SpeakNextClause, Generate, Synthesi...
#include "translate.h"            // for p_decoder, InitText, translator
#include "voice.h"                // for FreeVoiceList, VoiceReset, current_...
#include "wavegen.h"              // for WavegenFill, WavegenInit, WcmdqUsed

static unsigned char *outbuf = NULL;
static int outbuf_size = 0;
static unsigned char *out_start;

espeak_EVENT *event_list = NULL;
static int event_list_ix = 0;
static int n_event_list;
static long count_samples;
#if USE_LIBPCAUDIO
static struct audio_object *my_audio = NULL;
#endif

static unsigned int my_unique_identifier = 0;
static void *my_user_data = NULL;
static espeak_ng_OUTPUT_MODE my_mode = ENOUTPUT_MODE_SYNCHRONOUS;
static int out_samplerate = 0;
static int voice_samplerate = 22050;
static const int min_buffer_length = 60; // minimum buffer length in ms
static espeak_ng_STATUS err = ENS_OK;

static t_espeak_callback *synth_callback = NULL;

char path_home[N_PATH_HOME]; // this is the espeak-ng-data directory
extern int saved_parameters[N_SPEECH_PARAM]; // Parameters saved on synthesis start

void cancel_audio(void)
{
#if USE_LIBPCAUDIO
	if ((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO) {
		audio_object_flush(my_audio);
	}
#endif
}

static int dispatch_audio(short *outbuf, int length, espeak_EVENT *event)
{
	int a_wave_can_be_played = 1;
#if USE_ASYNC
	if ((my_mode & ENOUTPUT_MODE_SYNCHRONOUS) == 0)
		a_wave_can_be_played = fifo_is_command_enabled();
#endif

	switch ((int)my_mode)
	{
	case ENOUTPUT_MODE_SPEAK_AUDIO:
	case ENOUTPUT_MODE_SPEAK_AUDIO | ENOUTPUT_MODE_SYNCHRONOUS:
	{
		int event_type = 0;
		if (event)
			event_type = event->type;

		if (event_type == espeakEVENT_SAMPLERATE) {
			voice_samplerate = event->id.number;

			if (out_samplerate != voice_samplerate) {
#if USE_LIBPCAUDIO
				if (out_samplerate != 0) {
					// sound was previously open with a different sample rate
					audio_object_close(my_audio);
					out_samplerate = 0;
				}
#endif
#if USE_LIBPCAUDIO
				int error = audio_object_open(my_audio, AUDIO_OBJECT_FORMAT_S16LE, voice_samplerate, 1);
				if (error != 0) {
					fprintf(stderr, "error: %s\n", audio_object_strerror(my_audio, error));
					err = ENS_AUDIO_ERROR;
					return -1;
				}
#endif
				out_samplerate = voice_samplerate;
#if USE_ASYNC
				if ((my_mode & ENOUTPUT_MODE_SYNCHRONOUS) == 0)
					event_init();
#endif
			}
		}

#if USE_LIBPCAUDIO
		if (out_samplerate == 0) {
			int error = audio_object_open(my_audio, AUDIO_OBJECT_FORMAT_S16LE, voice_samplerate, 1);
			if (error != 0) {
				fprintf(stderr, "error: %s\n", audio_object_strerror(my_audio, error));
				err = ENS_AUDIO_ERROR;
				return -1;
			}
			out_samplerate = voice_samplerate;
		}
#endif

#if USE_LIBPCAUDIO
		if (outbuf && length && a_wave_can_be_played) {
			int error = audio_object_write(my_audio, (char *)outbuf, 2*length);
			if (error != 0)
				fprintf(stderr, "error: %s\n", audio_object_strerror(my_audio, error));
		}
#endif

#if USE_ASYNC
		while (event && a_wave_can_be_played) {
			// TBD: some event are filtered here but some insight might be given
			// TBD: in synthesise.cpp for avoiding to create WORDs with size=0.
			// TBD: For example sentence "or ALT)." returns three words
			// "or", "ALT" and "".
			// TBD: the last one has its size=0.
			if ((event->type == espeakEVENT_WORD) && (event->length == 0))
				break;
			if ((my_mode & ENOUTPUT_MODE_SYNCHRONOUS) == 0) {
				err = event_declare(event);
				if (err != ENS_EVENT_BUFFER_FULL)
					break;
				usleep(10000);
				a_wave_can_be_played = fifo_is_command_enabled();
			} else
				break;
		}
#endif
	}
		break;
	case 0:
		if (synth_callback)
			synth_callback(outbuf, length, event);
		break;
	}

	return a_wave_can_be_played == 0; // 1 = stop synthesis, -1 = error
}

static int create_events(short *outbuf, int length, espeak_EVENT *event_list)
{
	int finished;
	int i = 0;

	// The audio data are written to the output device.
	// The list of events in event_list (index: event_list_ix) is read:
	// Each event is declared to the "event" object which stores them internally.
	// The event object is responsible of calling the external callback
	// as soon as the relevant audio sample is played.

	do { // for each event
		espeak_EVENT *event;
		if (event_list_ix == 0)
			event = NULL;
		else
			event = event_list + i;
		finished = dispatch_audio((short *)outbuf, length, event);
		length = 0; // the wave data are played once.
		i++;
	} while ((i < event_list_ix) && !finished);
	return finished;
}

#if USE_ASYNC

int sync_espeak_terminated_msg(uint32_t unique_identifier, void *user_data)
{
	int finished = 0;

	memset(event_list, 0, 2*sizeof(espeak_EVENT));

	event_list[0].type = espeakEVENT_MSG_TERMINATED;
	event_list[0].unique_identifier = unique_identifier;
	event_list[0].user_data = user_data;
	event_list[1].type = espeakEVENT_LIST_TERMINATED;
	event_list[1].unique_identifier = unique_identifier;
	event_list[1].user_data = user_data;

	if (my_mode == ENOUTPUT_MODE_SPEAK_AUDIO) {
		while (1) {
			err = event_declare(event_list);
			if (err != ENS_EVENT_BUFFER_FULL)
				break;
			usleep(10000);
		}
	} else if (synth_callback)
		finished = synth_callback(NULL, 0, event_list);
	return finished;
}

#endif

static int check_data_path(const char *path, int allow_directory)
{
	if (!path) return 0;

	snprintf(path_home, sizeof(path_home), "%s/espeak-ng-data", path);
	if (GetFileLength(path_home) == -EISDIR)
		return 1;

	if (!allow_directory)
		return 0;

	snprintf(path_home, sizeof(path_home), "%s", path);
	return GetFileLength(path_home) == -EISDIR;
}

#pragma GCC visibility push(default)

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_InitializeOutput(espeak_ng_OUTPUT_MODE output_mode, int buffer_length, const char *device)
{
	(void)device; // unused if  USE_LIBPCAUDIO is not defined

	my_mode = output_mode;
	out_samplerate = 0;

#if USE_LIBPCAUDIO
	if (((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO) && (my_audio == NULL))
		my_audio = create_audio_device_object(device, "eSpeak", "Text-to-Speech");
#endif

#if USE_ASYNC
	if ((my_mode & ENOUTPUT_MODE_SYNCHRONOUS) == 0) fifo_init();
#endif

	// Don't allow buffer be smaller than safe minimum
	if (buffer_length < min_buffer_length)
		buffer_length = min_buffer_length;

	// allocate 2 bytes per sample
	// Always round up to the nearest sample and the nearest byte.
	int millisamples = buffer_length * samplerate;
	outbuf_size = (millisamples + 1000 - millisamples % 1000) / 500;
	out_start = (unsigned char *)realloc(outbuf, outbuf_size);
	if (out_start == NULL)
		return ENOMEM;
	else
		outbuf = out_start;

	// allocate space for event list.  Allow 200 events per second.
	// Add a constant to allow for very small buffer_length
	n_event_list = (buffer_length*200)/1000 + 20;
	espeak_EVENT *new_event_list = (espeak_EVENT *)realloc(event_list, sizeof(espeak_EVENT) * n_event_list);
	if (new_event_list == NULL)
		return ENOMEM;
	event_list = new_event_list;

	return ENS_OK;
}


ESPEAK_NG_API void espeak_ng_InitializePath(const char *path)
{
	if (check_data_path(path, 1))
		return;

#if PLATFORM_WINDOWS
	HKEY RegKey;
	unsigned long size;
	unsigned long var_type;
	unsigned char buf[sizeof(path_home)-13];

	if (check_data_path(getenv("ESPEAK_DATA_PATH"), 1))
		return;

	buf[0] = 0;
	RegOpenKeyExA(HKEY_LOCAL_MACHINE, "Software\\eSpeak NG", 0, KEY_READ, &RegKey);
	if (RegKey == NULL)
		RegOpenKeyExA(HKEY_LOCAL_MACHINE, "Software\\WOW6432Node\\eSpeak NG", 0, KEY_READ, &RegKey);
	size = sizeof(buf);
	var_type = REG_SZ;
	RegQueryValueExA(RegKey, "Path", 0, &var_type, buf, &size);

	if (check_data_path(buf, 1))
		return;
#elif !defined(PLATFORM_DOS)
	if (check_data_path(getenv("ESPEAK_DATA_PATH"), 1))
		return;

	if (check_data_path(getenv("HOME"), 0))
		return;
#endif

	strcpy(path_home, PATH_ESPEAK_DATA);
}

const int param_defaults[N_SPEECH_PARAM] = {
	0,   // silence (internal use)
	espeakRATE_NORMAL, // rate wpm
	100, // volume
	50,  // pitch
	50,  // range
	0,   // punctuation
	0,   // capital letters
	0,   // wordgap
	0,   // options
	0,   // intonation
	100, // ssml break mul
	0,
	0,   // emphasis
	0,   // line length
	0,   // voice type
};


ESPEAK_NG_API espeak_ng_STATUS espeak_ng_Initialize(espeak_ng_ERROR_CONTEXT *context)
{
	int param;
	int srate = 22050; // default sample rate 22050 Hz

	// It seems that the wctype functions don't work until the locale has been set
	// to something other than the default "C".  Then, not only Latin1 but also the
	// other characters give the correct results with iswalpha() etc.
	if (setlocale(LC_CTYPE, "C.UTF-8") == NULL) {
		if (setlocale(LC_CTYPE, "UTF-8") == NULL) {
			if (setlocale(LC_CTYPE, "en_US.UTF-8") == NULL)
				setlocale(LC_CTYPE, "");
		}
	}

	espeak_ng_STATUS result = LoadPhData(&srate, context);
	if (result != ENS_OK)
		return result;

	WavegenInit(srate, 0);
	LoadConfig();

	espeak_VOICE *current_voice_selected = espeak_GetCurrentVoice();
	memset(current_voice_selected, 0, sizeof(espeak_VOICE));
	SetVoiceStack(NULL, "");
	SynthesizeInit();
	InitNamedata();

	VoiceReset(0);

	for (param = 0; param < N_SPEECH_PARAM; param++)
		param_stack[0].parameter[param] = saved_parameters[param] = param_defaults[param];

	SetParameter(espeakRATE, espeakRATE_NORMAL, 0);
	SetParameter(espeakVOLUME, 100, 0);
	SetParameter(espeakCAPITALS, option_capitals, 0);
	SetParameter(espeakPUNCTUATION, option_punctuation, 0);
	SetParameter(espeakWORDGAP, 0, 0);

	option_phonemes = 0;
	option_phoneme_events = 0;

	// Seed random generator
	espeak_srand(time(NULL));

	return ENS_OK;
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SetPhonemeEvents(int enable, int ipa) {
	option_phoneme_events = 0;
	if (enable) {
		option_phoneme_events |= espeakINITIALIZE_PHONEME_EVENTS;
		if (ipa) {
			option_phoneme_events |= espeakINITIALIZE_PHONEME_IPA;
		}
	}
	return ENS_OK;
}

ESPEAK_NG_API int espeak_ng_GetSampleRate(void)
{
	return samplerate;
}

#pragma GCC visibility pop

static espeak_ng_STATUS Synthesize(unsigned int unique_identifier, const void *text, int flags)
{
	// Fill the buffer with output sound
	int length;
	int finished = 0;

	if ((outbuf == NULL) || (event_list == NULL))
		return ENS_NOT_INITIALIZED;

	option_ssml = flags & espeakSSML;
	option_phoneme_input = flags & espeakPHONEMES;
	option_endpause = flags & espeakENDPAUSE;

	count_samples = 0;

	espeak_ng_STATUS status;
	if (translator == NULL) {
		status = espeak_ng_SetVoiceByName(ESPEAKNG_DEFAULT_VOICE);
		if (status != ENS_OK)
			return status;
	}

	if (p_decoder == NULL)
		p_decoder = create_text_decoder();

	status = text_decoder_decode_string_multibyte(p_decoder, text, translator->encoding, flags);
	if (status != ENS_OK)
		return status;

	SpeakNextClause(0);

	for (;;) {
		out_ptr = outbuf;
		out_end = &outbuf[outbuf_size];
		event_list_ix = 0;
		WavegenFill();

		length = (out_ptr - outbuf)/2;
		count_samples += length;
		event_list[event_list_ix].type = espeakEVENT_LIST_TERMINATED; // indicates end of event list
		event_list[event_list_ix].unique_identifier = unique_identifier;
		event_list[event_list_ix].user_data = my_user_data;

		if ((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO) {
			finished = create_events((short *)outbuf, length, event_list);
			if (finished < 0)
				return ENS_AUDIO_ERROR;
		} else if (synth_callback)
			finished = synth_callback((short *)outbuf, length, event_list);
		if (finished) {
			SpeakNextClause(2); // stop
			return ENS_SPEECH_STOPPED;
		}

		if (Generate(phoneme_list, &n_phoneme_list, 1) == 0) {
			if (WcmdqUsed() == 0) {
				// don't process the next clause until the previous clause has finished generating speech.
				// This ensures that <audio> tag (which causes end-of-clause) is at a sound buffer boundary

				event_list[0].type = espeakEVENT_LIST_TERMINATED;
				event_list[0].unique_identifier = my_unique_identifier;
				event_list[0].user_data = my_user_data;

				if (SpeakNextClause(1) == 0) {
					finished = 0;
					if ((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO) {
						if (dispatch_audio(NULL, 0, NULL) < 0)
							return ENS_AUDIO_ERROR;
					} else if (synth_callback)
						finished = synth_callback(NULL, 0, event_list); // NULL buffer ptr indicates end of data
					if (finished) {
						SpeakNextClause(2); // stop
						return ENS_SPEECH_STOPPED;
					}
					return ENS_OK;
				}
			}
		}
	}
}

void MarkerEvent(int type, unsigned int char_position, int value, int value2, unsigned char *out_ptr)
{
	// type: 1=word, 2=sentence, 3=named mark, 4=play audio, 5=end, 7=phoneme
	espeak_EVENT *ep;
	double time;

	if ((event_list == NULL) || (event_list_ix >= (n_event_list-2)))
		return;

	ep = &event_list[event_list_ix++];
	ep->type = (espeak_EVENT_TYPE)type;
	ep->unique_identifier = my_unique_identifier;
	ep->user_data = my_user_data;
	ep->text_position = char_position & 0xffffff;
	ep->length = char_position >> 24;

#if !USE_MBROLA
	static const int mbrola_delay = 0;
#endif

	time = ((double)(count_samples + mbrola_delay + (out_ptr - out_start)/2)*1000.0)/samplerate;
	ep->audio_position = (int)time;
	ep->sample = (count_samples + mbrola_delay + (out_ptr - out_start)/2);

	if ((type == espeakEVENT_MARK) || (type == espeakEVENT_PLAY))
		ep->id.name = &namedata[value];
	else if (type == espeakEVENT_PHONEME) {
		int *p;
		p = (int *)(ep->id.string);
		p[0] = value;
		p[1] = value2;
	} else
		ep->id.number = value;
}

espeak_ng_STATUS sync_espeak_Synth(unsigned int unique_identifier, const void *text,
                                   unsigned int position, espeak_POSITION_TYPE position_type,
                                   unsigned int end_position, unsigned int flags, void *user_data)
{
	InitText(flags);
	my_unique_identifier = unique_identifier;
	my_user_data = user_data;

	for (int i = 0; i < N_SPEECH_PARAM; i++)
		saved_parameters[i] = param_stack[0].parameter[i];

	switch (position_type)
	{
	case POS_CHARACTER:
		skip_characters = position;
		break;
	case POS_WORD:
		skip_words = position;
		break;
	case POS_SENTENCE:
		skip_sentences = position;
		break;

	}
	if (skip_characters || skip_words || skip_sentences)
		skipping_text = true;

	end_character_position = end_position;

	espeak_ng_STATUS aStatus = Synthesize(unique_identifier, text, flags);
#if USE_LIBPCAUDIO
	if ((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO) {
		int error = (aStatus == ENS_SPEECH_STOPPED)
		          ? audio_object_flush(my_audio)
		          : audio_object_drain(my_audio);
		if (error != 0)
			fprintf(stderr, "error: %s\n", audio_object_strerror(my_audio, error));
	}
#endif

	return aStatus;
}

espeak_ng_STATUS sync_espeak_Synth_Mark(unsigned int unique_identifier, const void *text,
                                        const char *index_mark, unsigned int end_position,
                                        unsigned int flags, void *user_data)
{
	InitText(flags);

	my_unique_identifier = unique_identifier;
	my_user_data = user_data;

	if (index_mark != NULL) {
		strncpy0(skip_marker, index_mark, sizeof(skip_marker));
		skipping_text = true;
	}

	end_character_position = end_position;

	return Synthesize(unique_identifier, text, flags | espeakSSML);
}

espeak_ng_STATUS sync_espeak_Key(const char *key)
{
	// symbolic name, symbolicname_character  - is there a system resource of symbolic names per language?
	int letter;
	int ix;

	ix = utf8_in(&letter, key);
	if (key[ix] == 0) // a single character
		return sync_espeak_Char(letter);

	my_unique_identifier = 0;
	my_user_data = NULL;
	return Synthesize(0, key, 0); // speak key as a text string
}

espeak_ng_STATUS sync_espeak_Char(wchar_t character)
{
	// is there a system resource of character names per language?
	char buf[80];
	my_unique_identifier = 0;
	my_user_data = NULL;

	sprintf(buf, "<say-as interpret-as=\"tts:char\">&#%d;</say-as>", character);
	return Synthesize(0, buf, espeakSSML);
}

void sync_espeak_SetPunctuationList(const wchar_t *punctlist)
{
	// Set the list of punctuation which are spoken for "some".
	my_unique_identifier = 0;
	my_user_data = NULL;

	option_punctlist[0] = 0;
	if (punctlist != NULL) {
		wcsncpy(option_punctlist, punctlist, N_PUNCTLIST);
		option_punctlist[N_PUNCTLIST-1] = 0;
	}
}

#pragma GCC visibility push(default)

ESPEAK_API void espeak_SetSynthCallback(t_espeak_callback *SynthCallback)
{
	synth_callback = SynthCallback;
#if USE_ASYNC
	event_set_callback(synth_callback);
#endif
}

ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_Synthesize(const void *text, size_t size,
                     unsigned int position,
                     espeak_POSITION_TYPE position_type,
                     unsigned int end_position, unsigned int flags,
                     unsigned int *unique_identifier, void *user_data)
{
	(void)size; // unused in non-async modes

	unsigned int temp_identifier;

	if (unique_identifier == NULL)
		unique_identifier = &temp_identifier;
	*unique_identifier = 0;

	if (my_mode & ENOUTPUT_MODE_SYNCHRONOUS)
		return sync_espeak_Synth(0, text, position, position_type, end_position, flags, user_data);

#if USE_ASYNC
	// Create the text command
	t_espeak_command *c1 = create_espeak_text(text, size, position, position_type, end_position, flags, user_data);
	if (c1) {
		// Retrieve the unique identifier
		*unique_identifier = c1->u.my_text.unique_identifier;
	}

	// Create the "terminated msg" command (same uid)
	t_espeak_command *c2 = create_espeak_terminated_msg(*unique_identifier, user_data);

	// Try to add these 2 commands (single transaction)
	if (c1 && c2) {
		espeak_ng_STATUS status = fifo_add_commands(c1, c2);
		if (status != ENS_OK) {
			delete_espeak_command(c1);
			delete_espeak_command(c2);
		}
		return status;
	}

	delete_espeak_command(c1);
	delete_espeak_command(c2);
	return ENOMEM;
#else
	return sync_espeak_Synth(0, text, position, position_type, end_position, flags, user_data);
#endif
}

ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_SynthesizeMark(const void *text,
                         size_t size,
                         const char *index_mark,
                         unsigned int end_position,
                         unsigned int flags,
                         unsigned int *unique_identifier,
                         void *user_data)
{
	(void)size; // unused in non-async modes

	unsigned int temp_identifier;

	if (unique_identifier == NULL)
		unique_identifier = &temp_identifier;
	*unique_identifier = 0;

	if (my_mode & ENOUTPUT_MODE_SYNCHRONOUS)
		return sync_espeak_Synth_Mark(0, text, index_mark, end_position, flags, user_data);

#if USE_ASYNC
	// Create the mark command
	t_espeak_command *c1 = create_espeak_mark(text, size, index_mark, end_position,
	                                          flags, user_data);
	if (c1) {
		// Retrieve the unique identifier
		*unique_identifier = c1->u.my_mark.unique_identifier;
	}

	// Create the "terminated msg" command (same uid)
	t_espeak_command *c2 = create_espeak_terminated_msg(*unique_identifier, user_data);

	// Try to add these 2 commands (single transaction)
	if (c1 && c2) {
		espeak_ng_STATUS status = fifo_add_commands(c1, c2);
		if (status != ENS_OK) {
			delete_espeak_command(c1);
			delete_espeak_command(c2);
		}
		return status;
	}

	delete_espeak_command(c1);
	delete_espeak_command(c2);
	return ENOMEM;
#else
	return sync_espeak_Synth_Mark(0, text, index_mark, end_position, flags, user_data);
#endif
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SpeakKeyName(const char *key_name)
{
	// symbolic name, symbolicname_character  - is there a system resource of symbolicnames per language

	if (my_mode & ENOUTPUT_MODE_SYNCHRONOUS)
		return sync_espeak_Key(key_name);

#if USE_ASYNC
	t_espeak_command *c = create_espeak_key(key_name, NULL);
	espeak_ng_STATUS status = fifo_add_command(c);
	if (status != ENS_OK)
		delete_espeak_command(c);
	return status;
#else
	return sync_espeak_Key(key_name);
#endif
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SpeakCharacter(wchar_t character)
{
	// is there a system resource of character names per language?

#if USE_ASYNC
	if (my_mode & ENOUTPUT_MODE_SYNCHRONOUS)
		return sync_espeak_Char(character);

	t_espeak_command *c = create_espeak_char(character, NULL);
	espeak_ng_STATUS status = fifo_add_command(c);
	if (status != ENS_OK)
		delete_espeak_command(c);
	return status;
#else
	return sync_espeak_Char(character);
#endif
}

ESPEAK_API int espeak_GetParameter(espeak_PARAMETER parameter, int current)
{
	// current: 0=default value, 1=current value
	if (current)
		return param_stack[0].parameter[parameter];
	return param_defaults[parameter];
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SetParameter(espeak_PARAMETER parameter, int value, int relative)
{
#if USE_ASYNC
	if (my_mode & ENOUTPUT_MODE_SYNCHRONOUS)
		return SetParameter(parameter, value, relative);

	t_espeak_command *c = create_espeak_parameter(parameter, value, relative);

	espeak_ng_STATUS status = fifo_add_command(c);
	if (status != ENS_OK)
		delete_espeak_command(c);
	return status;
#else
	return SetParameter(parameter, value, relative);
#endif
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_SetPunctuationList(const wchar_t *punctlist)
{
	// Set the list of punctuation which are spoken for "some".

#if USE_ASYNC
	if (my_mode & ENOUTPUT_MODE_SYNCHRONOUS) {
		sync_espeak_SetPunctuationList(punctlist);
		return ENS_OK;
	}

	t_espeak_command *c = create_espeak_punctuation_list(punctlist);
	espeak_ng_STATUS status = fifo_add_command(c);
	if (status != ENS_OK)
		delete_espeak_command(c);
	return status;
#else
	sync_espeak_SetPunctuationList(punctlist);
	return ENS_OK;
#endif
}

ESPEAK_API void espeak_SetPhonemeTrace(int phonememode, FILE *stream)
{
	/* phonememode:  Controls the output of phoneme symbols for the text
	      bits 0-2:
	         value=0  No phoneme output (default)
	         value=1  Output the translated phoneme symbols for the text
	         value=2  as (1), but produces IPA phoneme names rather than ascii
	      bit 3:   output a trace of how the translation was done (showing the matching rules and list entries)
	      bit 4:   produce pho data for mbrola
	      bit 7:   use (bits 8-23) as a tie within multi-letter phonemes names
	      bits 8-23:  separator character, between phoneme names

	   stream   output stream for the phoneme symbols (and trace).  If stream=NULL then it uses stdout.
	*/

	option_phonemes = phonememode;
	f_trans = stream;
	if (stream == NULL)
		f_trans = stderr;
}

ESPEAK_API const char* espeak_TextToPhonemesWithTerminator(const void** textptr, int textmode, int phonememode, int* terminator)
{
	/* phoneme_mode
	    bit 1:   0=eSpeak's ascii phoneme names, 1= International Phonetic Alphabet (as UTF-8 characters).
	    bit 7:   use (bits 8-23) as a tie within multi-letter phonemes names
	    bits 8-23:  separator character, between phoneme names
	 */

	if (p_decoder == NULL)
		p_decoder = create_text_decoder();


	if (text_decoder_decode_string_multibyte(p_decoder, *textptr, translator->encoding, textmode) != ENS_OK)
		return NULL;

	TranslateClauseWithTerminator(translator, NULL, NULL, terminator);
	*textptr = text_decoder_get_buffer(p_decoder);

	return GetTranslatedPhonemeString(phonememode);
}

ESPEAK_API const char *espeak_TextToPhonemes(const void **textptr, int textmode, int phonememode)
{
	return espeak_TextToPhonemesWithTerminator(textptr, textmode, phonememode, NULL);
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_Cancel(void)
{
#if USE_ASYNC
	fifo_stop();
	event_clear_all();
#endif

#if USE_LIBPCAUDIO
	if ((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO)
		audio_object_flush(my_audio);
#endif
	embedded_value[EMBED_T] = 0; // reset echo for pronunciation announcements

	for (int i = 0; i < N_SPEECH_PARAM; i++)
		SetParameter(i, saved_parameters[i], 0);

	return ENS_OK;
}

ESPEAK_API int espeak_IsPlaying(void)
{
#if USE_ASYNC
	return fifo_is_busy();
#else
	return 0;
#endif
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_Synchronize(void)
{
	espeak_ng_STATUS berr = err;
#if USE_ASYNC
	while (espeak_IsPlaying())
		usleep(20000);
#endif
	err = ENS_OK;
	return berr;
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_Terminate(void)
{
#if USE_ASYNC
	fifo_stop();
	fifo_terminate();
	event_terminate();
#endif

	if ((my_mode & ENOUTPUT_MODE_SPEAK_AUDIO) == ENOUTPUT_MODE_SPEAK_AUDIO) {
#if USE_LIBPCAUDIO
		audio_object_close(my_audio);
		audio_object_destroy(my_audio);
		my_audio = NULL;
#endif
		out_samplerate = 0;
	}

	free(event_list);
	event_list = NULL;

	free(outbuf);
	outbuf = NULL;

	FreePhData();
	FreeVoiceList();

	DeleteTranslator(translator);
	translator = NULL;

	if (p_decoder != NULL) {
		destroy_text_decoder(p_decoder);
		p_decoder = NULL;
	}

	WavegenFini();

	return ENS_OK;
}

static const char version_string[] = PACKAGE_VERSION;
ESPEAK_API const char *espeak_Info(const char **ptr)
{
	if (ptr != NULL)
		*ptr = path_home;
	return version_string;
}

#pragma GCC visibility pop
