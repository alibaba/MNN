/*
 * mbrowrap -- A wrapper library around the mbrola binary
 * providing a subset of the API from the Windows mbrola DLL.
 *
 * Copyright (C) 2010 by Nicolas Pitre <nico@fluxnic.net>
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
 */

#ifndef MBROWRAP_H
#define MBROWRAP_H

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(_WIN32) && !defined(_WIN64)
#define WINAPI
typedef int BOOL;
#endif

/*
 * Initialize mbrola.  The 'voice_path' argument must contain the
 * path and file name to the mbrola voice database to be used. Returned
 * value is 0 on success, or an error code otherwise (currently only -1
 * is returned. If not successful, lastErrorStr_MBR() will provide the
 * error reason.  If this is successful, then close_MBR() must be called
 * before init_MBR() can be called again.
 */
extern int (WINAPI *init_MBR)(char *voice_path);

/*
 * Stop mbrola and release any resources.  It is necessary to call
 * this after a successful call to init_MBR() before init_MBR() can be
 * called again.
 */
extern void (WINAPI *close_MBR)(void);

/*
 * Stop any ongoing processing and flush all buffers.  After this call
 * any synthesis request will start afresh.
 */
extern void (WINAPI *reset_MBR)(void);

/*
 * Return at most 'nb_samples' audio samples into 'buffer'. The returned
 * value is the actual number of samples returned, or -1 on error.
 * If not successful, lastErrorStr_MBR() will provide the error reason.
 * Samples are always 16-bit little endian.
 */
extern int (WINAPI *read_MBR)(short *buffer, int nb_samples);

/*
 * Write a NULL terminated string of phoneme in the input buffer.
 * Return the number of chars actually written, or -1 on error.
 * If not successful, lastErrorStr_MBR() will provide the error reason.
 */
extern int (WINAPI *write_MBR)(char *data);

/*
 * Send a flush command to the mbrola input stream.
 * This is currently similar to write_MBR("#\n").  Return 1 on success
 * or 0 on failure. If not successful, lastErrorStr_MBR() will provide
 * the error reason.
 */
extern int (WINAPI *flush_MBR)(void);

/*
 * Return the audio sample frequency of the used voice database.
 */
extern int (WINAPI *getFreq_MBR)(void);

/*
 * Overall volume.
 */
extern void (WINAPI *setVolumeRatio_MBR)(float value);

/*
 * Copy into 'buffer' at most 'bufsize' bytes from the latest error
 * message.  This may also contain non-fatal errors from mbrola.  When
 * no error message is pending then an empty string is returned.
 * Consecutive calls to lastErrorStr_MBR() will return the same message.
 */
extern char * (WINAPI *lastErrorStr_MBR)(char *buffer, int bufsize);

/*
 * Tolerance to missing diphones.
 */
extern void (WINAPI *setNoError_MBR)(int no_error);

BOOL load_MBR(void);
void unload_MBR(void);

#ifdef __cplusplus
}
#endif

#endif
