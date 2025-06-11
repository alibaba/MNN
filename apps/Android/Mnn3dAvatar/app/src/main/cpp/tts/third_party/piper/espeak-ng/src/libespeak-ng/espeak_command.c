/*
 * Copyright (C) 2007, Gilles Casse <gcasse@oralux.org>
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

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include <espeak-ng/espeak_ng.h>

#include "espeak_command.h"

#if USE_ASYNC

static unsigned int my_current_text_id = 0;

t_espeak_command *create_espeak_text(const void *text, size_t size, unsigned int position, espeak_POSITION_TYPE position_type, unsigned int end_position, unsigned int flags, void *user_data)
{
	if (!text || !size)
		return NULL;

	void *a_text = NULL;
	t_espeak_text *data = NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_text = malloc(size+1);
	if (!a_text) {
		free(a_command);
		return NULL;
	}
	memcpy(a_text, text, size);

	a_command->type = ET_TEXT;
	a_command->state = CS_UNDEFINED;
	data = &(a_command->u.my_text);
	data->unique_identifier = ++my_current_text_id;
	data->text = a_text;
	data->position = position;
	data->position_type = position_type;
	data->end_position = end_position;
	data->flags = flags;
	data->user_data = user_data;

	return a_command;
}

t_espeak_command *create_espeak_terminated_msg(unsigned int unique_identifier, void *user_data)
{
	t_espeak_terminated_msg *data = NULL;
	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_TERMINATED_MSG;
	a_command->state = CS_UNDEFINED;
	data = &(a_command->u.my_terminated_msg);
	data->unique_identifier = unique_identifier;
	data->user_data = user_data;

	return a_command;
}

t_espeak_command *create_espeak_mark(const void *text, size_t size, const char *index_mark, unsigned int end_position, unsigned int flags, void *user_data)
{
	if (!text || !size || !index_mark)
		return NULL;

	void *a_text = NULL;
	char *a_index_mark = NULL;
	t_espeak_mark *data = NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_text = malloc(size);
	if (!a_text) {
		free(a_command);
		return NULL;
	}
	memcpy(a_text, text, size);

	a_index_mark = strdup(index_mark);

	a_command->type = ET_MARK;
	a_command->state = CS_UNDEFINED;
	data = &(a_command->u.my_mark);
	data->unique_identifier = ++my_current_text_id;
	data->text = a_text;
	data->index_mark = a_index_mark;
	data->end_position = end_position;
	data->flags = flags;
	data->user_data = user_data;

	return a_command;
}

t_espeak_command *create_espeak_key(const char *key_name, void *user_data)
{
	if (!key_name)
		return NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_KEY;
	a_command->state = CS_UNDEFINED;
	a_command->u.my_key.user_data = user_data;
	a_command->u.my_key.unique_identifier = ++my_current_text_id;
	a_command->u.my_key.key_name = strdup(key_name);

	return a_command;
}

t_espeak_command *create_espeak_char(wchar_t character, void *user_data)
{
	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_CHAR;
	a_command->state = CS_UNDEFINED;
	a_command->u.my_char.user_data = user_data;
	a_command->u.my_char.unique_identifier = ++my_current_text_id;
	a_command->u.my_char.character = character;

	return a_command;
}

t_espeak_command *create_espeak_parameter(espeak_PARAMETER parameter, int value, int relative)
{
	t_espeak_parameter *data = NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_PARAMETER;
	a_command->state = CS_UNDEFINED;
	data = &(a_command->u.my_param);
	data->parameter = parameter;
	data->value = value;
	data->relative = relative;

	return a_command;
}

t_espeak_command *create_espeak_punctuation_list(const wchar_t *punctlist)
{
	if (!punctlist)
		return NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_PUNCTUATION_LIST;
	a_command->state = CS_UNDEFINED;

	size_t len = (wcslen(punctlist) + 1)*sizeof(wchar_t);
	wchar_t *a_list = (wchar_t *)malloc(len);
	if (a_list == NULL) {
		free(a_command);
		return NULL;
	}
	memcpy(a_list, punctlist, len);
	a_command->u.my_punctuation_list = a_list;

	return a_command;
}

t_espeak_command *create_espeak_voice_name(const char *name)
{
	if (!name)
		return NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_VOICE_NAME;
	a_command->state = CS_UNDEFINED;
	a_command->u.my_voice_name = strdup(name);

	return a_command;
}

t_espeak_command *create_espeak_voice_spec(espeak_VOICE *voice)
{
	if (!voice)
		return NULL;

	t_espeak_command *a_command = (t_espeak_command *)malloc(sizeof(t_espeak_command));
	if (!a_command)
		return NULL;

	a_command->type = ET_VOICE_SPEC;
	a_command->state = CS_UNDEFINED;

	espeak_VOICE *data = &(a_command->u.my_voice_spec);
	memcpy(data, voice, sizeof(espeak_VOICE));

	if (voice->name)
		data->name = strdup(voice->name);

	if (voice->languages)
		data->languages = strdup(voice->languages);

	if (voice->identifier)
		data->identifier = strdup(voice->identifier);

	return a_command;
}

int delete_espeak_command(t_espeak_command *the_command)
{
	int a_status = 0;
	if (the_command) {
		switch (the_command->type)
		{
		case ET_TEXT:
			if (the_command->u.my_text.text)
				free(the_command->u.my_text.text);
			break;
		case ET_MARK:
			if (the_command->u.my_mark.text)
				free(the_command->u.my_mark.text);
			if (the_command->u.my_mark.index_mark)
				free((void *)(the_command->u.my_mark.index_mark));
			break;
		case ET_TERMINATED_MSG:
		{
			// if the terminated msg is pending,
			// it must be processed here for informing the calling program
			// that its message is finished.
			// This can be important for cleaning the related user data.
			t_espeak_terminated_msg *data = &(the_command->u.my_terminated_msg);
			if (the_command->state == CS_PENDING) {
				the_command->state = CS_PROCESSED;
				sync_espeak_terminated_msg(data->unique_identifier, data->user_data);
			}
		}
			break;
		case ET_KEY:
			if (the_command->u.my_key.key_name)
				free((void *)(the_command->u.my_key.key_name));
			break;
		case ET_CHAR:
		case ET_PARAMETER:
			// No allocation
			break;
		case ET_PUNCTUATION_LIST:
			if (the_command->u.my_punctuation_list)
				free((void *)(the_command->u.my_punctuation_list));
			break;
		case ET_VOICE_NAME:
			if (the_command->u.my_voice_name)
				free((void *)(the_command->u.my_voice_name));
			break;
		case ET_VOICE_SPEC:
		{
			espeak_VOICE *data = &(the_command->u.my_voice_spec);

			if (data->name)
				free((void *)data->name);

			if (data->languages)
				free((void *)data->languages);

			if (data->identifier)
				free((void *)data->identifier);
		}
			break;
		default:
			assert(0);
		}
		free(the_command);
		a_status = 1;
	}
	return a_status;
}

void process_espeak_command(t_espeak_command *the_command)
{
	if (the_command == NULL)
		return;

	the_command->state = CS_PROCESSED;

	switch (the_command->type)
	{
	case ET_TEXT:
	{
		t_espeak_text *data = &(the_command->u.my_text);
		sync_espeak_Synth(data->unique_identifier, data->text,
		                  data->position, data->position_type,
		                  data->end_position, data->flags, data->user_data);
	}
		break;
	case ET_MARK:
	{
		t_espeak_mark *data = &(the_command->u.my_mark);
		sync_espeak_Synth_Mark(data->unique_identifier, data->text,
		                       data->index_mark, data->end_position, data->flags,
		                       data->user_data);
	}
		break;
	case ET_TERMINATED_MSG:
	{
		t_espeak_terminated_msg *data = &(the_command->u.my_terminated_msg);
		sync_espeak_terminated_msg(data->unique_identifier, data->user_data);
	}
		break;
	case ET_KEY:
	{
		const char *data = the_command->u.my_key.key_name;
		sync_espeak_Key(data);
	}
		break;
	case ET_CHAR:
	{
		const wchar_t data = the_command->u.my_char.character;
		sync_espeak_Char(data);
	}
		break;
	case ET_PARAMETER:
	{
		t_espeak_parameter *data = &(the_command->u.my_param);
		SetParameter(data->parameter, data->value, data->relative);
	}
		break;
	case ET_PUNCTUATION_LIST:
	{
		const wchar_t *data = the_command->u.my_punctuation_list;
		sync_espeak_SetPunctuationList(data);
	}
		break;
	case ET_VOICE_NAME:
	{
		const char *data = the_command->u.my_voice_name;
		espeak_SetVoiceByName(data);
	}
		break;
	case ET_VOICE_SPEC:
	{
		espeak_VOICE *data = &(the_command->u.my_voice_spec);
		espeak_SetVoiceByProperties(data);
	}
		break;
	default:
		assert(0);
		break;
	}
}

#endif
