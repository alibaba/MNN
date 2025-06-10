/* An implementation of the eSpeak API using the espeak-ng API.
 *
 * Copyright (C) 2005 to 2013 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2016 Reece H. Dunn
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

#include <stdint.h>
#include <stdlib.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "compiledict.h"

#include "synthesize.h"           // for espeakINITIALIZE_PHONEME_IPA
#include "translate.h"            // for dictionary_name, option_phoneme_events

static espeak_ERROR status_to_espeak_error(espeak_ng_STATUS status)
{
	switch (status)
	{
	case ENS_OK:                     return EE_OK;
	case ENS_SPEECH_STOPPED:         return EE_OK;
	case ENS_VOICE_NOT_FOUND:        return EE_NOT_FOUND;
	case ENS_MBROLA_NOT_FOUND:       return EE_NOT_FOUND;
	case ENS_MBROLA_VOICE_NOT_FOUND: return EE_NOT_FOUND;
	case ENS_FIFO_BUFFER_FULL:       return EE_BUFFER_FULL;
	default:                         return EE_INTERNAL_ERROR;
	}
}

#pragma GCC visibility push(default)

ESPEAK_API int espeak_Initialize(espeak_AUDIO_OUTPUT output_type, int buf_length, const char *path, int options)
{
	espeak_ng_InitializePath(path);
	espeak_ng_ERROR_CONTEXT context = NULL;
	espeak_ng_STATUS result = espeak_ng_Initialize(&context);
	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, context);
		espeak_ng_ClearErrorContext(&context);
		if ((options & espeakINITIALIZE_DONT_EXIT) == 0)
			exit(1);
	}

	switch (output_type)
	{
	case AUDIO_OUTPUT_PLAYBACK:
		espeak_ng_InitializeOutput(ENOUTPUT_MODE_SPEAK_AUDIO, buf_length, NULL);
		break;
	case AUDIO_OUTPUT_RETRIEVAL:
		espeak_ng_InitializeOutput(0, buf_length, NULL);
		break;
	case AUDIO_OUTPUT_SYNCHRONOUS:
		espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, buf_length, NULL);
		break;
	case AUDIO_OUTPUT_SYNCH_PLAYBACK:
		espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS | ENOUTPUT_MODE_SPEAK_AUDIO, buf_length, NULL);
		break;
	}

	option_phoneme_events = (options & (espeakINITIALIZE_PHONEME_EVENTS | espeakINITIALIZE_PHONEME_IPA));

	return espeak_ng_GetSampleRate();
}

ESPEAK_API espeak_ERROR espeak_Synth(const void *text, size_t size,
                                     unsigned int position,
                                     espeak_POSITION_TYPE position_type,
                                     unsigned int end_position, unsigned int flags,
                                     unsigned int *unique_identifier, void *user_data)
{
	return status_to_espeak_error(espeak_ng_Synthesize(text, size, position, position_type, end_position, flags, unique_identifier, user_data));
}

ESPEAK_API espeak_ERROR espeak_Synth_Mark(const void *text, size_t size,
                                          const char *index_mark,
                                          unsigned int end_position,
                                          unsigned int flags,
                                          unsigned int *unique_identifier,
                                          void *user_data)
{
	return status_to_espeak_error(espeak_ng_SynthesizeMark(text, size, index_mark, end_position, flags, unique_identifier, user_data));
}

ESPEAK_API espeak_ERROR espeak_Key(const char *key_name)
{
	return status_to_espeak_error(espeak_ng_SpeakKeyName(key_name));
}

ESPEAK_API espeak_ERROR espeak_Char(wchar_t character)
{
	return status_to_espeak_error(espeak_ng_SpeakCharacter(character));
}

ESPEAK_API espeak_ERROR espeak_SetParameter(espeak_PARAMETER parameter, int value, int relative)
{
	return status_to_espeak_error(espeak_ng_SetParameter(parameter, value, relative));
}

ESPEAK_API espeak_ERROR espeak_SetPunctuationList(const wchar_t *punctlist)
{
	return status_to_espeak_error(espeak_ng_SetPunctuationList(punctlist));
}

ESPEAK_API espeak_ERROR espeak_SetVoiceByName(const char *name)
{
	return status_to_espeak_error(espeak_ng_SetVoiceByName(name));
}

ESPEAK_API espeak_ERROR espeak_SetVoiceByFile(const char *filename)
{
	return status_to_espeak_error(espeak_ng_SetVoiceByFile(filename));
}

ESPEAK_API espeak_ERROR espeak_SetVoiceByProperties(espeak_VOICE *voice_selector)
{
	return status_to_espeak_error(espeak_ng_SetVoiceByProperties(voice_selector));
}

ESPEAK_API espeak_ERROR espeak_Cancel(void)
{
	return status_to_espeak_error(espeak_ng_Cancel());
}

ESPEAK_API espeak_ERROR espeak_Synchronize(void)
{
	return status_to_espeak_error(espeak_ng_Synchronize());
}

ESPEAK_API espeak_ERROR espeak_Terminate(void)
{
	return status_to_espeak_error(espeak_ng_Terminate());
}

ESPEAK_API void espeak_CompileDictionary(const char *path, FILE *log, int flags)
{
	espeak_ng_ERROR_CONTEXT context = NULL;
	espeak_ng_STATUS result = espeak_ng_CompileDictionary(path, dictionary_name, log, flags, &context);
	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, context);
		espeak_ng_ClearErrorContext(&context);
	}
}

#pragma GCC visibility pop
