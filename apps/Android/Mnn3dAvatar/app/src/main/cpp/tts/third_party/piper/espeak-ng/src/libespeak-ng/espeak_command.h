/*
 * Copyright (C) 2007, Gilles Casse <gcasse@oralux.org>
 * Copyright (C) 2015 Reece H. Dunn
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

#ifndef ESPEAK_NG_COMMAND_H
#define ESPEAK_NG_COMMAND_H

#include <espeak-ng/espeak_ng.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
	ET_TEXT,
	ET_MARK,
	ET_KEY,
	ET_CHAR,
	ET_PARAMETER,
	ET_PUNCTUATION_LIST,
	ET_VOICE_NAME,
	ET_VOICE_SPEC,
	ET_TERMINATED_MSG
} t_espeak_type;

typedef struct {
	unsigned int unique_identifier;
	void *text;
	unsigned int position;
	espeak_POSITION_TYPE position_type;
	unsigned int end_position;
	unsigned int flags;
	void *user_data;
} t_espeak_text;

typedef struct {
	unsigned int unique_identifier;
	void *text;
	const char *index_mark;
	unsigned int end_position;
	unsigned int flags;
	void *user_data;
} t_espeak_mark;

typedef struct {
	unsigned int unique_identifier;
	void *user_data;
	wchar_t character;
} t_espeak_character;

typedef struct {
	unsigned int unique_identifier;
	void *user_data;
	const char *key_name;
} t_espeak_key;


typedef struct {
	unsigned int unique_identifier;
	void *user_data;
} t_espeak_terminated_msg;


typedef struct {
	espeak_PARAMETER parameter;
	int value;
	int relative;
} t_espeak_parameter;

typedef enum {
	CS_UNDEFINED, // The command has just been created
	CS_PENDING,   // stored in the fifo
	CS_PROCESSED  // processed
} t_command_state;

typedef struct {
	t_espeak_type type;
	t_command_state state;

	union command {
		t_espeak_text my_text;
		t_espeak_mark my_mark;
		t_espeak_key my_key;
		t_espeak_character my_char;
		t_espeak_parameter my_param;
		const wchar_t *my_punctuation_list;
		const char *my_voice_name;
		espeak_VOICE my_voice_spec;
		t_espeak_terminated_msg my_terminated_msg;
	} u;
} t_espeak_command;


t_espeak_command *create_espeak_text(const void *text, size_t size, unsigned int position, espeak_POSITION_TYPE position_type, unsigned int end_position, unsigned int flags, void *user_data);

t_espeak_command *create_espeak_mark(const void *text, size_t size, const char *index_mark, unsigned int end_position, unsigned int flags, void *user_data);

t_espeak_command *create_espeak_terminated_msg(unsigned int unique_identifier, void *user_data);

t_espeak_command *create_espeak_key(const char *key_name, void *user_data);

t_espeak_command *create_espeak_char(wchar_t character, void *user_data);

t_espeak_command *create_espeak_parameter(espeak_PARAMETER parameter, int value, int relative);

t_espeak_command *create_espeak_punctuation_list(const wchar_t *punctlist);

t_espeak_command *create_espeak_voice_name(const char *name);

t_espeak_command *create_espeak_voice_spec(espeak_VOICE *voice_spec);

void process_espeak_command(t_espeak_command *the_command);

int delete_espeak_command(t_espeak_command *the_command);

espeak_ng_STATUS sync_espeak_Synth(unsigned int unique_identifier, const void *text,
                                   unsigned int position, espeak_POSITION_TYPE position_type,
                                   unsigned int end_position, unsigned int flags, void *user_data);
espeak_ng_STATUS sync_espeak_Synth_Mark(unsigned int unique_identifier, const void *text,
                                        const char *index_mark, unsigned int end_position,
                                        unsigned int flags, void *user_data);
espeak_ng_STATUS sync_espeak_Key(const char *key);
espeak_ng_STATUS sync_espeak_Char(wchar_t character);
void sync_espeak_SetPunctuationList(const wchar_t *punctlist);
void sync_espeak_SetParameter(espeak_PARAMETER parameter, int value, int relative);
espeak_ng_STATUS SetParameter(int parameter, int value, int relative);

int sync_espeak_terminated_msg(unsigned int unique_identifier, void *user_data);

#ifdef __cplusplus
}
#endif

// >
#endif
