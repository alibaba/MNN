/*
 * Copyright (C) 2006 to 2013 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
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

#include "config.h"

#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

#ifndef PROGRAM_NAME
#define PROGRAM_NAME "espeak-ng"
#endif

#ifndef PLAYBACK_MODE
#define PLAYBACK_MODE ENOUTPUT_MODE_SPEAK_AUDIO
#endif

extern ESPEAK_NG_API void strncpy0(char *to, const char *from, int size);
extern ESPEAK_NG_API int utf8_in(int *c, const char *buf);
extern ESPEAK_NG_API int GetFileLength(const char *filename);

static const char *help_text =
    "\n" PROGRAM_NAME " [options] [\"<words>\"]\n\n"
    "-f <text file>   Text file to speak\n"
    "--stdin    Read text input from stdin at once till to the end of a stream.\n\n"
    "If neither -f nor --stdin are provided, then <words> from arguments are spoken,\n"
	"or text is spoken from stdin, read separately one line by line at a time.\n\n"
    "-a <integer>\n"
    "\t   Amplitude, 0 to 200, default is 100\n"
    "-d <device>\n"
    "\t   Use the specified device to speak the audio on. If not specified, the\n"
    "\t   default audio device is used.\n"
    "-g <integer>\n"
    "\t   Word gap. Pause between words, units of 10mS at the default speed\n"
    "-k <integer>\n"
    "\t   Indicate capital letters with: 1=sound, 2=the word \"capitals\",\n"
    "\t   higher values indicate a pitch increase (try -k20).\n"
    "-l <integer>\n"
    "\t   Line length. If not zero (which is the default), consider\n"
    "\t   lines less than this length as end-of-clause\n"
    "-p <integer>\n"
    "\t   Pitch adjustment, 0 to 99, default is 50\n"
    "-P <integer>\n"
    "\t   Pitch range adjustment, 0 to 99, default is 50\n"
    "-s <integer>\n"
    "\t   Speed in approximate words per minute. The default is 175\n"
    "-v <voice name>\n"
    "\t   Use voice file of this name from espeak-ng-data/voices\n"
    "-w <wave file name>\n"
    "\t   Write speech to this WAV file, rather than speaking it directly\n"
    "-b\t   Input text encoding, 1=UTF8, 2=8 bit, 4=16 bit \n"
    "-m\t   Interpret SSML markup, and ignore other < > tags\n"
    "--ssml-break=<percentage>\n"
    "\t   Set SSML break time multiplier, default is 100\n"
    "-q\t   Quiet, don't produce any speech (may be useful with -x)\n"
    "-x\t   Write phoneme mnemonics to stdout\n"
    "-X\t   Write phonemes mnemonics and translation trace to stdout\n"
    "-z\t   No final sentence pause at the end of the text\n"
    "-D\t   Enable deterministic random mode\n"
    "--compile=<voice name>\n"
    "\t   Compile pronunciation rules and dictionary from the current\n"
    "\t   directory. <voice name> specifies the language\n"
    "--compile-debug=<voice name>\n"
    "\t   Compile pronunciation rules and dictionary from the current\n"
    "\t   directory, including line numbers for use with -X.\n"
    "\t   <voice name> specifies the language\n"
#if USE_MBROLA
    "--compile-mbrola=<voice name>\n"
    "\t   Compile an MBROLA voice\n"
#endif
    "--compile-intonations\n"
    "\t   Compile the intonation data\n"
    "--compile-phonemes=<phsource-dir>\n"
    "\t   Compile the phoneme data using <phsource-dir> or the default phsource directory\n"
    "--ipa      Write phonemes to stdout using International Phonetic Alphabet\n"
    "--path=\"<path>\"\n"
    "\t   Specifies the directory containing the espeak-ng-data directory\n"
    "--pho      Write mbrola phoneme data (.pho) to stdout or to the file in --phonout\n"
    "--phonout=\"<filename>\"\n"
    "\t   Write phoneme output from -x -X --ipa and --pho to this file\n"
    "--punct=\"<characters>\"\n"
    "\t   Speak the names of punctuation characters during speaking.  If\n"
    "\t   =<characters> is omitted, all punctuation is spoken.\n"
    "--sep=<character>\n"
    "\t   Separate phonemes (from -x --ipa) with <character>.\n"
    "\t   Default is space, z means ZWJN character.\n"
    "--split=<minutes>\n"
    "\t   Starts a new WAV file every <minutes>.  Used with -w\n"
    "--stdout   Write speech output to stdout\n"
    "--tie=<character>\n"
    "\t   Use a tie character within multi-letter phoneme names.\n"
    "\t   Default is U+361, z means ZWJ character.\n"
    "--version  Shows version number and date, and location of espeak-ng-data\n"
    "--voices=<language>\n"
    "\t   List the available voices for the specified language.\n"
    "\t   If <language> is omitted, then list all voices.\n"
    "--load     Load voice from a file in current directory by name.\n"
    "-h, --help Show this help.\n";

static int samplerate;
bool quiet = false;
unsigned int samples_total = 0;
unsigned int samples_split = 0;
unsigned int samples_split_seconds = 0;
unsigned int wavefile_count = 0;

FILE *f_wavfile = NULL;
char filetype[5];
char wavefile[200];

static void DisplayVoices(FILE *f_out, char *language)
{
	int ix;
	const char *p;
	int len;
	int count;
	int c;
	size_t j;
	const espeak_VOICE *v;
	const char *lang_name;
	char age_buf[12];
	char buf[80];
	const espeak_VOICE **voices;
	espeak_VOICE voice_select;

	static const char genders[4] = { '-', 'M', 'F', '-' };

	if ((language != NULL) && (language[0] != 0)) {
		// display only voices for the specified language, in order of priority
		voice_select.languages = language;
		voice_select.age = 0;
		voice_select.gender = 0;
		voice_select.name = NULL;
		voices = espeak_ListVoices(&voice_select);
	} else
		voices = espeak_ListVoices(NULL);

	fprintf(f_out, "Pty Language       Age/Gender VoiceName          File                 Other Languages\n");

	for (ix = 0; (v = voices[ix]) != NULL; ix++) {
		count = 0;
		p = v->languages;
		while (*p != 0) {
			len = strlen(p+1);
			lang_name = p+1;

			if (v->age == 0)
				strcpy(age_buf, " --");
			else
				sprintf(age_buf, "%3d", v->age);

			if (count == 0) {
				for (j = 0; j < sizeof(buf); j++) {
					// replace spaces in the name
					if ((c = v->name[j]) == ' ')
						c = '_';
					if ((buf[j] = c) == 0)
						break;
				}
				fprintf(f_out, "%2d  %-15s%s/%c      %-18s %-20s ",
				        p[0], lang_name, age_buf, genders[v->gender], buf, v->identifier);
			} else
				fprintf(f_out, "(%s %d)", lang_name, p[0]);
			count++;
			p += len+2;
		}
		fputc('\n', f_out);
	}
}

static void Write4Bytes(FILE *f, int value)
{
	// Write 4 bytes to a file, least significant first
	int ix;

	for (ix = 0; ix < 4; ix++) {
		fputc(value & 0xff, f);
		value = value >> 8;
	}
}

static int OpenWavFile(char *path, int rate)
{
	static const unsigned char wave_hdr[44] = {
		'R', 'I', 'F', 'F', 0x24, 0xf0, 0xff, 0x7f, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
		0x10, 0, 0, 0, 1, 0, 1, 0,  9, 0x3d, 0, 0, 0x12, 0x7a, 0, 0,
		2, 0, 0x10, 0, 'd', 'a', 't', 'a',  0x00, 0xf0, 0xff, 0x7f
	};

	if (path == NULL)
		return 2;

	while (isspace(*path)) path++;

	f_wavfile = NULL;
	if (path[0] != 0) {
		if (strcmp(path, "stdout") == 0) {
#ifdef _WIN32
			// prevent Windows adding 0x0d before 0x0a bytes
			_setmode(_fileno(stdout), _O_BINARY);
#endif
			f_wavfile = stdout;
		} else
			f_wavfile = fopen(path, "wb");
	}

	if (f_wavfile == NULL) {
		fprintf(stderr, "Can't write to: '%s'\n", path);
		return 1;
	}

	fwrite(wave_hdr, 1, 24, f_wavfile);
	Write4Bytes(f_wavfile, rate);
	Write4Bytes(f_wavfile, rate * 2);
	fwrite(&wave_hdr[32], 1, 12, f_wavfile);
	return 0;
}

static void CloseWavFile()
{
	unsigned int pos;

	if ((f_wavfile == NULL) || (f_wavfile == stdout))
		return;

	fflush(f_wavfile);
	pos = ftell(f_wavfile);

	if (fseek(f_wavfile, 4, SEEK_SET) != -1)
		Write4Bytes(f_wavfile, pos - 8);

	if (fseek(f_wavfile, 40, SEEK_SET) != -1)
		Write4Bytes(f_wavfile, pos - 44);

	fclose(f_wavfile);
	f_wavfile = NULL;
}

static int SynthCallback(short *wav, int numsamples, espeak_EVENT *events)
{
	char fname[210];

	if (quiet || wav == NULL) return 0;

	while (events->type != 0) {
		if (events->type == espeakEVENT_SAMPLERATE) {
			samplerate = events->id.number;
			samples_split = samples_split_seconds * samplerate;
		} else if (events->type == espeakEVENT_SENTENCE) {
			// start a new WAV file when the limit is reached, at this sentence boundary
			if ((samples_split > 0) && (samples_total > samples_split)) {
				CloseWavFile();
				samples_total = 0;
				wavefile_count++;
			}
		}
		events++;
	}

	if (f_wavfile == NULL) {
		if (samples_split > 0) {
			sprintf(fname, "%s_%.2d%s", wavefile, wavefile_count+1, filetype);
			if (OpenWavFile(fname, samplerate) != 0)
				return 1;
		} else if (OpenWavFile(wavefile, samplerate) != 0)
			return 1;
	}

	if (numsamples > 0) {
		samples_total += numsamples;
		fwrite(wav, numsamples*2, 1, f_wavfile);
	}
	return 0;
}

static void PrintVersion()
{
	const char *version;
	const char *path_data;
	espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, NULL, espeakINITIALIZE_DONT_EXIT);
	version = espeak_Info(&path_data);
	printf("eSpeak NG text-to-speech: %s  Data at: %s\n", version, path_data);
}

int main(int argc, char **argv)
{
	static const struct option long_options[] = {
		{ "help",    no_argument,       0, 'h' },
		{ "stdin",   no_argument,       0, 0x100 },
		{ "compile-debug", optional_argument, 0, 0x101 },
		{ "compile", optional_argument, 0, 0x102 },
		{ "punct",   optional_argument, 0, 0x103 },
		{ "voices",  optional_argument, 0, 0x104 },
		{ "stdout",  no_argument,       0, 0x105 },
		{ "split",   optional_argument, 0, 0x106 },
		{ "path",    required_argument, 0, 0x107 },
		{ "phonout", required_argument, 0, 0x108 },
		{ "pho",     no_argument,       0, 0x109 },
		{ "ipa",     optional_argument, 0, 0x10a },
		{ "version", no_argument,       0, 0x10b },
		{ "sep",     optional_argument, 0, 0x10c },
		{ "tie",     optional_argument, 0, 0x10d },
#if USE_MBROLA
		{ "compile-mbrola", optional_argument, 0, 0x10e },
#endif
		{ "compile-intonations", no_argument, 0, 0x10f },
		{ "compile-phonemes", optional_argument, 0, 0x110 },
		{ "load",    no_argument,       0, 0x111 },
		{ "ssml-break", required_argument, 0, 0x112 },
		{ 0, 0, 0, 0 }
	};

	FILE *f_text = NULL;
	char *p_text = NULL;
	FILE *f_phonemes_out = stdout;
	char *data_path = NULL; // use default path for espeak-ng-data

	int option_index = 0;
	int c;
	int ix;
	char *optarg2;
	int value;
	int flag_stdin = 0;
	int flag_compile = 0;
	int flag_load = 0;
	int filesize = 0;
	int synth_flags = espeakCHARS_AUTO | espeakPHONEMES | espeakENDPAUSE;

	int volume = -1;
	int speed = -1;
	int pitch = -1;
	int pitch_range = -1;
	int wordgap = -1;
	int option_capitals = -1;
	int option_punctuation = -1;
	int phonemes_separator = 0;
	int phoneme_options = 0;
	int option_linelength = 0;
	int option_waveout = 0;
	int ssml_break = -1;
	bool deterministic = 0;
	
	espeak_VOICE voice_select;
	char filename[200];
	char voicename[40];
	char devicename[200];
	#define N_PUNCTLIST 100
	wchar_t option_punctlist[N_PUNCTLIST];

	voicename[0] = 0;
	wavefile[0] = 0;
	filename[0] = 0;
	devicename[0] = 0;
	option_punctlist[0] = 0;

	while (true) {
		c = getopt_long(argc, argv, "a:b:Dd:f:g:hk:l:mp:P:qs:v:w:xXz",
		                long_options, &option_index);

		// Detect the end of the options.
		if (c == -1)
			break;
		optarg2 = optarg;

		switch (c)
		{
		case 'b':
			// input character encoding, 8bit, 16bit, UTF8
			if ((sscanf(optarg2, "%d", &value) == 1) && (value <= 4))
				synth_flags |= value;
			else
				synth_flags |= espeakCHARS_8BIT;
			break;
		case 'd':
			strncpy0(devicename, optarg2, sizeof(devicename));
			break;
		case 'D':
			deterministic = 1;
			break;
		case 'h':
			printf("\n");
			PrintVersion();
			printf("%s", help_text);
			return 0;
		case 'k':
			option_capitals = atoi(optarg2);
			break;
		case 'x':
			phoneme_options |= espeakPHONEMES_SHOW;
			break;
		case 'X':
			phoneme_options |= espeakPHONEMES_TRACE;
			break;
		case 'm':
			synth_flags |= espeakSSML;
			break;
		case 'p':
			pitch = atoi(optarg2);
			break;
		case 'P':
			pitch_range = atoi(optarg2);
			break;
		case 'q':
			quiet = true;
			break;
		case 'f':
			strncpy0(filename, optarg2, sizeof(filename));
			break;
		case 'l':
			option_linelength = atoi(optarg2);
			break;
		case 'a':
			volume = atoi(optarg2);
			break;
		case 's':
			speed = atoi(optarg2);
			break;
		case 'g':
			wordgap = atoi(optarg2);
			break;
		case 'v':
			strncpy0(voicename, optarg2, sizeof(voicename));
			break;
		case 'w':
			option_waveout = 1;
			strncpy0(wavefile, optarg2, sizeof(filename));
			break;
		case 'z': // remove pause from the end of a sentence
			synth_flags &= ~espeakENDPAUSE;
			break;
		case 0x100: // --stdin
			flag_stdin = 1;
			break;
		case 0x105: // --stdout
			option_waveout = 1;
			strcpy(wavefile, "stdout");
			break;
		case 0x101: // --compile-debug
		case 0x102: // --compile
			if (optarg2 != NULL && *optarg2) {
				strncpy0(voicename, optarg2, sizeof(voicename));
				flag_compile = c;
				quiet = true;
				break;
			} else {
				fprintf(stderr, "Voice name to '%s' not specified.\n", c == 0x101 ? "--compile-debug" : "--compile");
				exit(EXIT_FAILURE);
			}
		case 0x103: // --punct
			option_punctuation = 1;
			if (optarg2 != NULL) {
				ix = 0;
				while ((ix < N_PUNCTLIST) && ((option_punctlist[ix] = optarg2[ix]) != 0)) ix++;
				option_punctlist[N_PUNCTLIST-1] = 0;
				option_punctuation = 2;
			}
			break;
		case 0x104: // --voices
			espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path, 0);
			DisplayVoices(stdout, optarg2);
			exit(0);
		case 0x106: // -- split
			if (optarg2 == NULL)
				samples_split_seconds = 30 * 60; // default 30 minutes
			else
				samples_split_seconds = atoi(optarg2) * 60;
			break;
		case 0x107: // --path
			data_path = optarg2;
			break;
		case 0x108: // --phonout
			if ((f_phonemes_out = fopen(optarg2, "w")) == NULL)
				fprintf(stderr, "Can't write to: %s\n", optarg2);
			break;
		case 0x109: // --pho
			phoneme_options |= espeakPHONEMES_MBROLA;
			break;
		case 0x10a: // --ipa
			phoneme_options |= espeakPHONEMES_IPA;
			if (optarg2 != NULL) {
				// deprecated and obsolete
				switch (atoi(optarg2))
				{
				case 1:
					phonemes_separator = '_';
					break;
				case 2:
					phonemes_separator = 0x0361;
					phoneme_options |= espeakPHONEMES_TIE;
					break;
				case 3:
					phonemes_separator = 0x200d; // ZWJ
					phoneme_options |= espeakPHONEMES_TIE;
					break;
				}

			}
			break;
		case 0x10b: // --version
			PrintVersion();
			exit(0);
		case 0x10c: // --sep
			phoneme_options |= espeakPHONEMES_SHOW;
			if (optarg2 == 0)
				phonemes_separator = ' ';
			else
				utf8_in(&phonemes_separator, optarg2);
			if (phonemes_separator == 'z')
				phonemes_separator = 0x200c; // ZWNJ
			break;
		case 0x10d: // --tie
			phoneme_options |= (espeakPHONEMES_SHOW | espeakPHONEMES_TIE);
			if (optarg2 == 0)
				phonemes_separator = 0x0361; // default: combining-double-inverted-breve
			else
				utf8_in(&phonemes_separator, optarg2);
			if (phonemes_separator == 'z')
				phonemes_separator = 0x200d; // ZWJ
			break;
#if USE_MBROLA
		case 0x10e: // --compile-mbrola
		{
			espeak_ng_InitializePath(data_path);
			espeak_ng_ERROR_CONTEXT context = NULL;
			espeak_ng_STATUS result = espeak_ng_CompileMbrolaVoice(optarg2, stdout, &context);
			if (result != ENS_OK) {
				espeak_ng_PrintStatusCodeMessage(result, stderr, context);
				espeak_ng_ClearErrorContext(&context);
				return EXIT_FAILURE;
			}
			return EXIT_SUCCESS;
		}
#endif
		case 0x10f: // --compile-intonations
		{
			espeak_ng_InitializePath(data_path);
			espeak_ng_ERROR_CONTEXT context = NULL;
			espeak_ng_STATUS result = espeak_ng_CompileIntonation(stdout, &context);
			if (result != ENS_OK) {
				espeak_ng_PrintStatusCodeMessage(result, stderr, context);
				espeak_ng_ClearErrorContext(&context);
				return EXIT_FAILURE;
			}
			return EXIT_SUCCESS;
		}
		case 0x110: // --compile-phonemes
		{
			espeak_ng_InitializePath(data_path);
			espeak_ng_ERROR_CONTEXT context = NULL;
			espeak_ng_STATUS result;
			if (optarg2) {
				result = espeak_ng_CompilePhonemeDataPath(22050, optarg2, NULL, stdout, &context);
			} else {
				result = espeak_ng_CompilePhonemeData(22050, stdout, &context);
			}
			if (result != ENS_OK) {
				espeak_ng_PrintStatusCodeMessage(result, stderr, context);
				espeak_ng_ClearErrorContext(&context);
				return EXIT_FAILURE;
			}
			return EXIT_SUCCESS;
		}
		case 0x111: // --load
			flag_load = 1;
			break;
		case 0x112: // --ssml-break
			ssml_break = atoi(optarg2);
			break;
		default:
			exit(0);
		}
	}

	espeak_ng_InitializePath(data_path);
	espeak_ng_ERROR_CONTEXT context = NULL;
	espeak_ng_STATUS result = espeak_ng_Initialize(&context);
	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, context);
		espeak_ng_ClearErrorContext(&context);
		exit(1);
	}

	if (deterministic) {
		// Set random generator state to well-known
		espeak_ng_SetRandSeed(1);
	}

	if (option_waveout || quiet) {
		// writing to a file (or no output), we can use synchronous mode
		result = espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, devicename[0] ? devicename : NULL);
		samplerate = espeak_ng_GetSampleRate();
		samples_split = samplerate * samples_split_seconds;

		espeak_SetSynthCallback(SynthCallback);
		if (samples_split) {
			char *extn;
			extn = strrchr(wavefile, '.');
			if ((extn != NULL) && ((wavefile + strlen(wavefile) - extn) <= 4)) {
				strcpy(filetype, extn);
				*extn = 0;
			}
		}
	} else {
		// play the sound output
//		result = espeak_ng_InitializeOutput(PLAYBACK_MODE, 0, devicename[0] ? devicename : NULL);
		samplerate = espeak_ng_GetSampleRate();
	}

	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, NULL);
		exit(EXIT_FAILURE);
	}

	if (voicename[0] == 0)
		strcpy(voicename, ESPEAKNG_DEFAULT_VOICE);

	if(flag_load)
		result = espeak_ng_SetVoiceByFile(voicename);
	else
		result = espeak_ng_SetVoiceByName(voicename);
	if (result != ENS_OK) {
		memset(&voice_select, 0, sizeof(voice_select));
		voice_select.languages = voicename;
		result = espeak_ng_SetVoiceByProperties(&voice_select);
		if (result != ENS_OK) {
			espeak_ng_PrintStatusCodeMessage(result, stderr, NULL);
			exit(EXIT_FAILURE);
		}
	}

	if (flag_compile) {
		// This must be done after the voice is set
		espeak_ng_ERROR_CONTEXT context = NULL;
		espeak_ng_STATUS result = espeak_ng_CompileDictionary("", NULL, stderr, flag_compile & 0x1, &context);
		if (result != ENS_OK) {
			espeak_ng_PrintStatusCodeMessage(result, stderr, context);
			espeak_ng_ClearErrorContext(&context);
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	// set any non-default values of parameters. This must be done after espeak_Initialize()
	if (speed > 0)
		espeak_SetParameter(espeakRATE, speed, 0);
	if (volume >= 0)
		espeak_SetParameter(espeakVOLUME, volume, 0);
	if (pitch >= 0)
		espeak_SetParameter(espeakPITCH, pitch, 0);
	if (pitch_range >= 0)
		espeak_SetParameter(espeakRANGE, pitch_range, 0);
	if (option_capitals >= 0)
		espeak_SetParameter(espeakCAPITALS, option_capitals, 0);
	if (option_punctuation >= 0)
		espeak_SetParameter(espeakPUNCTUATION, option_punctuation, 0);
	if (wordgap >= 0)
		espeak_SetParameter(espeakWORDGAP, wordgap, 0);
	if (option_linelength > 0)
		espeak_SetParameter(espeakLINELENGTH, option_linelength, 0);
	if (ssml_break > 0)
		espeak_SetParameter(espeakSSML_BREAK_MUL, ssml_break, 0);
	if (option_punctuation == 2)
		espeak_SetPunctuationList(option_punctlist);

	espeak_SetPhonemeTrace(phoneme_options | (phonemes_separator << 8), f_phonemes_out);

	if (filename[0] == 0) {
		if ((optind < argc) && (flag_stdin == 0)) {
			// there's a non-option parameter, and no -f or --stdin
			// use it as text
			p_text = argv[optind];
		} else {
			f_text = stdin;
			if (flag_stdin == 0)
				flag_stdin = 2;
		}
	} else {
		struct stat st;
		if (stat(filename, &st) != 0) {
			fprintf(stderr, "Failed to stat() file '%s'\n", filename);
			exit(EXIT_FAILURE);
		}
		filesize = GetFileLength(filename);
		f_text = fopen(filename, "r");
		if (f_text == NULL) {
			fprintf(stderr, "Failed to read file '%s'\n", filename);
			exit(EXIT_FAILURE);
		}
		if (S_ISFIFO(st.st_mode)) {
			flag_stdin = 2;
		}
	}

	if (p_text != NULL) {
		int size;
		size = strlen(p_text);
		espeak_Synth(p_text, size+1, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
	} else if (flag_stdin) {
		size_t max = 1000;
		if ((p_text = (char *)malloc(max)) == NULL) {
//			espeak_ng_PrintStatusCodeMessage(ENOMEM, stderr, NULL);
			exit(EXIT_FAILURE);
		}

		if (flag_stdin == 2) {
			// line by line input on stdin or from FIFO
			while (fgets(p_text, max, f_text) != NULL) {
				p_text[max-1] = 0;
				espeak_Synth(p_text, max, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
				// Allow subprocesses to use the audio data through pipes.
				fflush(stdout);
			}
			if (f_text != stdin) {
				fclose(f_text);
			}
		} else {
			// bulk input on stdin
			ix = 0;
			while (true) {
				if ((c = fgetc(stdin)) == EOF)
					break;
				p_text[ix++] = (char)c;
				if (ix >= (max-1)) {
					char *new_text = NULL;
					if (max <= SIZE_MAX - 1000) {
						max += 1000;
						new_text = (char *)realloc(p_text, max);
					}
					if (new_text == NULL) {
						free(p_text);
//						espeak_ng_PrintStatusCodeMessage(ENOMEM, stderr, NULL);
						exit(EXIT_FAILURE);
					}
					p_text = new_text;
				}
			}
			if (ix > 0) {
				p_text[ix] = 0;
				espeak_Synth(p_text, ix, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
			}
		}

		free(p_text);
	} else if (f_text != NULL) {
		if ((p_text = (char *)malloc(filesize+1)) == NULL) {
//			espeak_ng_PrintStatusCodeMessage(ENOMEM, stderr, NULL);
			exit(EXIT_FAILURE);
		}

		fread(p_text, 1, filesize, f_text);
		p_text[filesize] = 0;
		espeak_Synth(p_text, filesize+1, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
		fclose(f_text);

		free(p_text);
	}

	result = espeak_ng_Synchronize();
	if (result != ENS_OK) {
		espeak_ng_PrintStatusCodeMessage(result, stderr, NULL);
		exit(EXIT_FAILURE);
	}

	if (f_phonemes_out != stdout)
		fclose(f_phonemes_out);

	CloseWavFile();
	espeak_ng_Terminate();
	return 0;
}
