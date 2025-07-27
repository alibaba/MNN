/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/encoding.h>

#include "phonemelist.h"
#include "phoneme.h"     // for PHONEME_TAB, phVOWEL, REPLACE_PHONEMES, phPAUSE
#include "synthdata.h"   // for InterpretPhoneme, SelectPhonemeTable
#include "synthesize.h"  // for PHONEME_LIST, PHONEME_LIST2, phoneme_tab
#include "translate.h"   // for Translator, LANGUAGE_OPTIONS, option_wordgap

#include "phoneme.h"
#include "synthesize.h"
#include "translate.h"
#include "speech.h"

static void SetRegressiveVoicing(int regression, PHONEME_LIST2 *plist2, PHONEME_TAB *ph, Translator *tr);
static void ReInterpretPhoneme(PHONEME_TAB *ph, PHONEME_TAB *ph2, PHONEME_LIST *plist3, PHONEME_LIST *plist3_start, Translator *tr, PHONEME_DATA *phdata, WORD_PH_DATA *worddata);

static const unsigned char pause_phonemes[8] = {
	0, phonPAUSE_VSHORT, phonPAUSE_SHORT, phonPAUSE, phonPAUSE_LONG, phonGLOTTALSTOP, phonPAUSE_LONG, phonPAUSE_LONG
};

extern int n_ph_list2;
extern PHONEME_LIST2 ph_list2[N_PHONEME_LIST]; // first stage of text->phonemes

static int SubstitutePhonemes(PHONEME_LIST *plist_out)
{
	// Copy the phonemes list and perform any substitutions that are required for the
	// current voice
	int ix;
	int k;
	int replace_flags;
	int n_plist_out = 0;
	bool word_end;
	PHONEME_LIST2 *plist2;
	PHONEME_TAB *next = NULL;
	int deleted_sourceix = -1;

	for (ix = 0; (ix < n_ph_list2) && (n_plist_out < N_PHONEME_LIST); ix++) {
		plist2 = &ph_list2[ix];
		if (deleted_sourceix != -1) {
			plist2->sourceix = deleted_sourceix;
			deleted_sourceix = -1;
		}

		if (plist2->phcode == phonSWITCH)
			SelectPhonemeTable(plist2->tone_ph);

		// don't do any substitution if the language has been temporarily changed
		if (!(plist2->synthflags & SFLAG_SWITCHED_LANG)) {
			if (ix < (n_ph_list2 -1))
				next = phoneme_tab[ph_list2[ix+1].phcode];

			word_end = false;
			if (ix == n_ph_list2 -1 || (plist2+1)->sourceix || ((next != 0) && (next->type == phPAUSE)))
				word_end = true; // this phoneme is the end of a word

			// check whether a Voice has specified that we should replace this phoneme
			for (k = 0; k < n_replace_phonemes; k++) {
				if (plist2->phcode == replace_phonemes[k].old_ph) {
					replace_flags = replace_phonemes[k].type;

					if ((replace_flags & 1) && (word_end == false))
						continue; // this replacement only occurs at the end of a word

					if ((replace_flags & 2) && ((plist2->stresslevel & 0x7) > 3))
						continue; // this replacement doesn't occur in stressed syllables

					if ((replace_flags & 4) && (plist2->sourceix == 0))
						continue; // this replacement only occurs at the start of a word

					// substitute the replacement phoneme
					plist2->phcode = replace_phonemes[k].new_ph;
					if ((plist2->stresslevel > 1) && (phoneme_tab[plist2->phcode]->phflags & phUNSTRESSED))
						plist2->stresslevel = 0; // the replacement must be unstressed
					break;
				}
			}

			if (plist2->phcode == 0) {
				deleted_sourceix = plist2->sourceix;
				continue; // phoneme has been replaced by NULL, so don't copy it
			}
		}

		// copy phoneme into the output list
		memcpy(&plist_out[n_plist_out], plist2, sizeof(PHONEME_LIST2));
		plist_out[n_plist_out].ph = phoneme_tab[plist2->phcode];
		plist_out[n_plist_out].type = plist_out[n_plist_out].ph->type;
		n_plist_out++;
	}
	return n_plist_out;
}

void MakePhonemeList(Translator *tr, int post_pause, bool start_sentence)
{
	int ix = 0;
	int j;
	int insert_ph = 0;
	PHONEME_LIST *phlist;
	PHONEME_TAB *ph = NULL;
	PHONEME_TAB *next;
	int unstress_count = 0;
	int word_stress = 0;
	int current_phoneme_tab;
	int max_stress;
	int end_sourceix;
	int alternative;
	int delete_count;
	int word_start;
	PHONEME_DATA phdata;
	bool start_of_clause = true;

	int n_ph_list3;
	PHONEME_LIST *plist3;
	PHONEME_LIST *plist3_inserted = NULL;
	PHONEME_LIST ph_list3[N_PHONEME_LIST];

	PHONEME_LIST2 *plist2;
	WORD_PH_DATA worddata;

	memset(&worddata, 0, sizeof(worddata));
	plist2 = ph_list2;
	phlist = phoneme_list;
	end_sourceix = plist2[n_ph_list2-1].sourceix;
	MAKE_MEM_UNDEFINED(&phoneme_list, sizeof(phoneme_list));

	// is the last word of the clause unstressed ?
	max_stress = 0;
	for (j = n_ph_list2-3; j >= 0; j--) {
		// start with the last phoneme (before the terminating pauses) and move backwards
		if ((plist2[j].stresslevel & 0x7f) > max_stress)
			max_stress = plist2[j].stresslevel & 0x7f;
		if (plist2[j].sourceix != 0)
			break;
	}
	if (max_stress < 4) {
		// the last word is unstressed, look for a previous word that can be stressed
		while (--j >= 0) {
			if (plist2[j].synthflags & SFLAG_PROMOTE_STRESS) { // dictionary flags indicated that this stress can be promoted
				plist2[j].stresslevel = 4; // promote to stressed
				break;
			}
			if (plist2[j].stresslevel >= 4) {
				// found a stressed syllable, so stop looking
				break;
			}
		}
	}

	// look for switch of phoneme tables
	delete_count = 0;
	current_phoneme_tab = tr->phoneme_tab_ix;
	int deleted_sourceix = -1;
	for (j = 0; j < n_ph_list2; j++) {
		if (current_phoneme_tab != tr->phoneme_tab_ix)
			plist2[j].synthflags |= SFLAG_SWITCHED_LANG;

		if (delete_count > 0) {
			memcpy(&plist2[j-delete_count], &plist2[j], sizeof(plist2[0]));
			if (deleted_sourceix != -1) {
				plist2[j-delete_count].sourceix = deleted_sourceix;
				deleted_sourceix = -1;
			}
		}

		if (plist2[j].phcode == phonSWITCH) {
			if ((!(plist2[j].synthflags & SFLAG_EMBEDDED)) && (
			        (plist2[j].tone_ph == current_phoneme_tab) ||
			        (plist2[j+1].phcode == phonSWITCH) ||
			        ((plist2[j+1].phcode == phonPAUSE) && (plist2[j+2].phcode == phonSWITCH))
			        )) {
				// delete this phonSWITCH if it's switching to the current phoneme table, or
				// delete this phonSWITCH if its followed by another phonSWITCH
				if (deleted_sourceix == -1 && plist2[j].sourceix != 0)
					deleted_sourceix = plist2[j].sourceix;
				delete_count++;
			} else
				current_phoneme_tab = plist2[j].tone_ph;
		}

	}
	n_ph_list2 -= delete_count;

	SelectPhonemeTable(current_phoneme_tab);

	int regression;
	if ((regression = tr->langopts.param[LOPT_REGRESSIVE_VOICING]) != 0) {
		SetRegressiveVoicing(regression, plist2, ph, tr);
	}

	SelectPhonemeTable(tr->phoneme_tab_ix);
	n_ph_list3 = SubstitutePhonemes(ph_list3) - 2;

	for (j = 0; (j < n_ph_list3) && (ix < N_PHONEME_LIST-3);) {
		if (ph_list3[j].sourceix) {
			// start of a word
			int k;
			int nextw;
			word_stress = 0;

			// find the highest stress level in this word
			for (nextw = j; nextw < n_ph_list3;) {
				if (ph_list3[nextw].stresslevel > word_stress)
					word_stress = ph_list3[nextw].stresslevel;

				nextw++;
				if (ph_list3[nextw].sourceix)
					break; // start of the next word
			}
			for (k = j; k < nextw; k++)
				ph_list3[k].wordstress = word_stress;
			j = nextw;
		} else
			j++;
	}

	// transfer all the phonemes of the clause into phoneme_list
	ph = phoneme_tab[phonPAUSE];
	ph_list3[0].ph = ph;
	word_start = 1;

	SelectPhonemeTable(tr->phoneme_tab_ix);
	for (j = 0; insert_ph || ((j < n_ph_list3) && (ix < N_PHONEME_LIST-3)); j++) {
		plist3 = &ph_list3[j];

		bool inserted = false;
		bool deleted = false;
		if (insert_ph != 0) {
			// we have a (linking) phoneme which we need to insert here
			next = phoneme_tab[plist3->phcode];      // this phoneme, i.e. after the insert

			// re-use the previous entry for the inserted phoneme.
			// That's OK because we don't look backwards from plist3   *** but CountVowelPosition() and isAfterStress does !!!
			j--;
			plist3 = plist3_inserted = &ph_list3[j];
			if (j > 0) {
				// move all previous phonemes in the word back one place
				int k;
				if (word_start > 0) {
					k = word_start;
					word_start--;
				} else
					k = 2;   // No more space, don't loose the start of word mark at ph_list2[word_start]
				for (; k <= j; k++)
					memcpy(&ph_list3[k-1], &ph_list3[k], sizeof(*plist3));
			}
			memset(&plist3[0], 0, sizeof(*plist3));
			plist3->phcode = insert_ph;
			ph = phoneme_tab[insert_ph];
			plist3->ph = ph;
			insert_ph = 0;
			inserted = true; // don't insert the same phoneme repeatedly
		} else {
			// otherwise get the next phoneme from the list
			if (plist3->sourceix != 0)
				word_start = j;

			ph = phoneme_tab[plist3->phcode];
			plist3[0].ph = ph;

			if (plist3->phcode == phonSWITCH) {
				// change phoneme table
				SelectPhonemeTable(plist3->tone_ph);
			}
			next = phoneme_tab[plist3[1].phcode]; // the phoneme after this one
			plist3[1].ph = next;
		}

		if (ph == NULL) continue;

		InterpretPhoneme(tr, 0x100, plist3, ph_list3, &phdata, &worddata);

		if ((alternative = phdata.pd_param[pd_CHANGE_NEXTPHONEME]) > 0) {
			ph_list3[j+1].ph = phoneme_tab[alternative];
			ph_list3[j+1].phcode = alternative;
			ph_list3[j+1].type = phoneme_tab[alternative]->type;
			next = phoneme_tab[alternative];
		}

		if (((alternative = phdata.pd_param[pd_INSERTPHONEME]) > 0) && (inserted == false)) {
			// PROBLEM: if we insert a phoneme before a vowel then we loose the stress.
			PHONEME_TAB *ph2;
			ph2 = ph;

			insert_ph = plist3->phcode;
			ph = phoneme_tab[alternative];
			plist3->ph = ph;
			plist3->phcode = alternative;

			ReInterpretPhoneme(ph, ph2, plist3, ph_list3, tr, &phdata, &worddata);
		}

		if ((alternative = phdata.pd_param[pd_CHANGEPHONEME]) > 0) {
			PHONEME_TAB *ph2;
			ph2 = ph;
			ph = phoneme_tab[alternative];
			plist3->ph = ph;
			plist3->phcode = alternative;

			if (alternative == 1)
				deleted = true; // NULL phoneme, discard
			else {
				ReInterpretPhoneme(ph, ph2, plist3, ph_list3, tr, &phdata, &worddata);
			}
		}

		if ((ph->type == phVOWEL) && (deleted == false)) {
			// Check for consecutive unstressed syllables, even across word boundaries.
			// Do this after changing phonemes according to stress level.
			if (plist3->stresslevel <= 1) {
				// an unstressed vowel
				unstress_count++;

				if (tr->langopts.stress_flags & 0x08) {
					// change sequences of consecutive unstressed vowels in unstressed words to diminished stress (TEST)
					PHONEME_LIST *p;
					for (p = plist3+1; p->type != phPAUSE; p++) {
						if (p->type == phVOWEL) {
							if (p->stresslevel <= 1) {
								if (plist3->wordstress < 4)
									plist3->stresslevel = 0;
								if (p->wordstress < 4)
									p->stresslevel = 0;
							}
							break;
						}
					}
				} else {
					if ((unstress_count > 1) && ((unstress_count & 1) == 0)) {
						// in a sequence of unstressed syllables, reduce alternate syllables to 'diminished'
						// stress.  But not for the last phoneme of a stressed word
						if ((tr->langopts.stress_flags & S_NO_DIM) || ((word_stress > 3) && ((plist3+1)->sourceix != 0))) {
							// An unstressed final vowel of a stressed word
							unstress_count = 1; // try again for next syllable
						} else
							plist3->stresslevel = 0; // change stress to 'diminished'
					}
				}
			} else
				unstress_count = 0;
		}

		if ((plist3+1)->synthflags & SFLAG_LENGTHEN) {
			static const char types_double[] = { phFRICATIVE, phVFRICATIVE, phNASAL, phLIQUID, 0 };
			if ((j > 0) && (strchr(types_double, next->type))) {
				// lengthen this consonant by doubling it
				// BUT, can't insert a phoneme at position plist3[0] because it crashes PrevPh()
				insert_ph = next->code;
				(plist3+1)->synthflags ^= SFLAG_LENGTHEN;
			}
		}

		if ((plist3+1)->sourceix != 0) {
			int x;

			if (tr->langopts.vowel_pause && (ph->type != phPAUSE)) {

				if ((ph->type != phVOWEL) && (tr->langopts.vowel_pause & 0x200)) {
					// add a pause after a word which ends in a consonant
					insert_ph = phonPAUSE_NOLINK;
				}

				if (next->type == phVOWEL) {
					if ((x = tr->langopts.vowel_pause & 0x0c) != 0) {
						// break before a word which starts with a vowel
						if (x == 0xc)
							insert_ph = phonPAUSE_NOLINK;
						else
							insert_ph = phonPAUSE_VSHORT;
					}

					if ((ph->type == phVOWEL) && ((x = tr->langopts.vowel_pause & 0x03) != 0)) {
						// adjacent vowels over a word boundary
						if (x == 2)
							insert_ph = phonPAUSE_SHORT;
						else
							insert_ph = phonPAUSE_VSHORT;
					}

					if (((plist3+1)->stresslevel >= 4) && (tr->langopts.vowel_pause & 0x100)) {
						// pause before a words which starts with a stressed vowel
						insert_ph = phonPAUSE_SHORT;
					}
				}
			}

			if ((plist3 != plist3_inserted) && (ix > 0)) {
				if ((x = (tr->langopts.word_gap & 0x7)) != 0) {
					if ((x > 1) || ((insert_ph != phonPAUSE_SHORT) && (insert_ph != phonPAUSE_NOLINK))) {
						// don't reduce the pause
						insert_ph = pause_phonemes[x];
					}
				}
				if (option_wordgap > 0)
					insert_ph = phonPAUSE_LONG;
			}
		}

		plist3[2].ph = phoneme_tab[plist3[2].phcode];

		if ((insert_ph == 0) && (phdata.pd_param[pd_APPENDPHONEME] != 0))
			insert_ph = phdata.pd_param[pd_APPENDPHONEME];

		if (deleted == false) {
			phlist[ix].ph = ph;
			phlist[ix].type = ph->type;
			phlist[ix].env = PITCHfall; // default, can be changed in the "intonation" module
			phlist[ix].synthflags = plist3->synthflags;
			phlist[ix].stresslevel = plist3->stresslevel & 0xf;
			phlist[ix].wordstress = plist3->wordstress;
			phlist[ix].tone_ph = plist3->tone_ph;
			phlist[ix].sourceix = 0;
			phlist[ix].phcode = ph->code;

			if (plist3->sourceix != 0) {
				phlist[ix].sourceix = plist3->sourceix;
				phlist[ix].newword = PHLIST_START_OF_WORD;

				if (start_sentence) {
					phlist[ix].newword |= PHLIST_START_OF_SENTENCE;
					start_sentence = false;
				}

				if (start_of_clause) {
					phlist[ix].newword |= PHLIST_START_OF_CLAUSE;
					start_of_clause = false;
				}
			} else
				phlist[ix].newword = 0;

			phlist[ix].length = phdata.pd_param[i_SET_LENGTH]*2;
			if ((ph->code == phonPAUSE_LONG) && (option_wordgap > 0) && (plist3[1].sourceix != 0)) {
				phlist[ix].ph = phoneme_tab[phonPAUSE_SHORT];
				phlist[ix].length = option_wordgap*14; // 10mS per unit at the default speed
			}

			if (ph->type == phVOWEL || ph->type == phLIQUID || ph->type == phNASAL || ph->type == phVSTOP || ph->type == phVFRICATIVE || (ph->phflags & phPREVOICE)) {
				phlist[ix].length = 128; // length_mod
				phlist[ix].env = PITCHfall;
			}

			phlist[ix].prepause = 0;
			phlist[ix].amp = 20; // default, will be changed later
			phlist[ix].pitch1 = 255;
			phlist[ix].pitch2 = 255;
			ix++;
		}
	}

	phlist[ix].newword = PHLIST_END_OF_CLAUSE;

	phlist[ix].phcode = phonPAUSE;
	phlist[ix].type = phPAUSE; // terminate with 2 Pause phonemes
	phlist[ix].length = post_pause; // length of the pause, depends on the punctuation
	phlist[ix].sourceix = end_sourceix;
	phlist[ix].synthflags = 0;
	phlist[ix].prepause = 0;
	phlist[ix++].ph = phoneme_tab[phonPAUSE];

	phlist[ix].newword = 0;

	phlist[ix].phcode = phonPAUSE;
	phlist[ix].type = phPAUSE;
	phlist[ix].length = 0;
	phlist[ix].sourceix = 0;
	phlist[ix].synthflags = 0;
	phlist[ix].prepause = 0;
	phlist[ix++].ph = phoneme_tab[phonPAUSE_SHORT];

	n_phoneme_list = ix;

	SelectPhonemeTable(tr->phoneme_tab_ix);
}

static void SetRegressiveVoicing(int regression, PHONEME_LIST2 *plist2, PHONEME_TAB *ph, Translator *tr) {
		// set consonant clusters to all voiced or all unvoiced
		// Regressive
		int type;
		bool stop_propagation = false;
		int voicing = 0;

		for (int j = n_ph_list2-1; j >= 0; j--) {
			if (plist2[j].phcode == phonSWITCH) {
				/* Find previous phonSWITCH to determine language we're switching back to */
				int k;
				for (k = j-1; k >= 0; k--)
					if (plist2[k].phcode == phonSWITCH)
						break;
				if (k >= 0)
					SelectPhonemeTable(plist2[k].tone_ph);
				else
					SelectPhonemeTable(tr->phoneme_tab_ix);
			}
			ph = phoneme_tab[plist2[j].phcode];
			if (ph == NULL)
				continue;

			if (plist2[j].synthflags & SFLAG_SWITCHED_LANG) {
				stop_propagation = false;
				voicing = 0;
				if (regression & 0x100)
					voicing = 1; // word-end devoicing
				continue;
			}

			type = ph->type;

			if (regression & 0x2) {
				// [v] amd [v;] don't cause regression, or [R^]
				if (((ph->mnemonic & 0xff) == 'v') || ((ph->mnemonic & 0xff) == 'R')) {
					stop_propagation = true;
					if (regression & 0x10)
						voicing = 0;
				}
			}

			if ((type == phSTOP) || type == (phFRICATIVE)) {
				if ((voicing == 0) && (regression & 0xf))
					voicing = 1;
				else if ((voicing == 2) && (ph->end_type != 0)) // use end_type field for voicing_switch for consonants
					plist2[j].phcode = ph->end_type; // change to voiced equivalent
			} else if ((type == phVSTOP) || type == (phVFRICATIVE)) {
				if ((voicing == 0) && (regression & 0xf))
					voicing = 2;
				else if ((voicing == 1) && (ph->end_type != 0))
					plist2[j].phcode = ph->end_type; // change to unvoiced equivalent
			} else {
				if (regression & 0x8) {
					// LANG=Polish, propagate through liquids and nasals
					if ((type == phPAUSE) || (type == phVOWEL))
						voicing = 0;
				} else
					voicing = 0;
			}
			if (stop_propagation) {
				voicing = 0;
				stop_propagation = false;
			}

			if (plist2[j].sourceix) {
				if (regression & 0x04) {
					// stop propagation at a word boundary
					voicing = 0;
				}
				if (regression & 0x100) {
					// devoice word-final consonants, unless propagating voiced
					if (voicing == 0)
						voicing = 1;
				}
			}
		}
	}

static void ReInterpretPhoneme(PHONEME_TAB *ph, PHONEME_TAB *ph2, PHONEME_LIST *plist3, PHONEME_LIST *plist3_start, Translator *tr, PHONEME_DATA *phdata, WORD_PH_DATA *worddata) {
if (ph->type == phVOWEL) {
				plist3->synthflags |= SFLAG_SYLLABLE;
				if (ph2->type != phVOWEL)
					plist3->stresslevel = 0; // change from non-vowel to vowel, make sure it's unstressed
			} else
				plist3->synthflags &= ~SFLAG_SYLLABLE;

			// re-interpret the changed phoneme
			// But it doesn't obey a second ChangePhoneme()
			InterpretPhoneme(tr, 0x100, plist3, plist3_start, phdata, worddata);
}
