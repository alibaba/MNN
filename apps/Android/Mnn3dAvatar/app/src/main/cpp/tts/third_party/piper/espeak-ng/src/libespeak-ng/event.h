/*
 * Copyright (C) 2007, Gilles Casse <gcasse@oralux.org>
 * Copyright (C) 2015-2016 Reece H. Dunn
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

/*
   Manage events (sentence, word, mark, end,...), is responsible of calling the external
   callback as soon as the relevant audio sample is played.


   The audio stream is composed of samples from synthetised messages or audio icons.
   Each event is associated to a sample.

   Scenario:

   - event_declare is called for each expected event.

   - A timeout is started for the first pending event.

   - When the timeout happens, the synth_callback is called.

   Note: the timeout is checked against the real progress of the audio stream, which depends on pauses or underruns. If the real progress is lower than the expected one, a new timeout starts.

 */

#ifndef ESPEAK_NG_EVENT_H
#define ESPEAK_NG_EVENT_H

#include <espeak-ng/espeak_ng.h>

#ifdef __cplusplus
extern "C"
{
#endif

// Initialize the event component.
// First function to be called.
// the callback will be called when the event actually occurs.
// The callback is detailed in speak_lib.h .
void event_init(void);
void event_set_callback(t_espeak_callback *cb);

// Clear any pending event.
espeak_ng_STATUS event_clear_all(void);

// Declare a future event
espeak_ng_STATUS event_declare(espeak_EVENT *event);

// Terminate the event component.
// Last function to be called.
void event_terminate(void);

// general functions
struct timespec;
void clock_gettime2(struct timespec *ts);
void add_time_in_ms(struct timespec *ts, int time_in_ms);

#ifdef __cplusplus
}
#endif

#endif
