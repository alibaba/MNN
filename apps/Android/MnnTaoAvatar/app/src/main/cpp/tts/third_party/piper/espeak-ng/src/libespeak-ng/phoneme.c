/*
 * Copyright (C) 2017 Reece H. Dunn
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
#include <string.h>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

#include "phoneme.h"

phoneme_feature_t phoneme_feature_from_string(const char *feature)
{
	if (!feature || strlen(feature) != 3)
		return inv;
	return (feature[0] << 16) | (feature[1] << 8) | feature[2];
}

espeak_ng_STATUS
phoneme_add_feature(PHONEME_TAB *phoneme,
                    phoneme_feature_t feature)
{
	if (!phoneme) return EINVAL;
	switch (feature)
	{
	// manner of articulation
	case nas:
		phoneme->type = phNASAL;
		break;
	case stp:
	case afr: // FIXME: eSpeak treats 'afr' as 'stp'.
		phoneme->type = phSTOP;
		break;
	case frc:
	case apr: // FIXME: eSpeak is using this for [h], with 'liquid' used for [l] and [r].
		phoneme->type = phFRICATIVE;
		break;
	case flp: // FIXME: Why is eSpeak using a vstop (vcd + stp) for this?
		phoneme->type = phVSTOP;
		break;
	case trl: // FIXME: 'trill' should be the type; 'liquid' should be a flag (phoneme files specify both).
		phoneme->phflags |= phTRILL;
		break;
	case clk:
	case ejc:
	case imp:
	case lat:
		// Not supported by eSpeak.
		break;
	case vwl:
		phoneme->type = phVOWEL;
		break;
	case sib:
		phoneme->phflags |= phSIBILANT;
		break;
	// place of articulation
	case blb:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_BILABIAL << 16;
		break;
	case lbd:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_LABIODENTAL << 16;
		break;
	case dnt:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_DENTAL << 16;
		break;
	case alv:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_ALVEOLAR << 16;
		break;
	case rfx:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_RETROFLEX << 16;
		break;
	case pla:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_PALATO_ALVEOLAR << 16;
		break;
	case pal:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_PALATAL << 16;
		phoneme->phflags |= phPALATAL;
		break;
	case vel:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_VELAR << 16;
		break;
	case lbv:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_LABIO_VELAR << 16;
		break;
	case uvl:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_UVULAR << 16;
		break;
	case phr:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_PHARYNGEAL << 16;
		break;
	case glt:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_GLOTTAL << 16;
		break;
	case bld:
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_BILABIAL << 16;
		break;
	case alp: // pla pzd
		phoneme->phflags &= ~phARTICULATION;
		phoneme->phflags |= phPLACE_PALATO_ALVEOLAR << 16;
		phoneme->phflags |= phPALATAL;
		break;
	// voice
	case vcd:
		phoneme->phflags |= phVOICED;
		break;
	case vls:
		phoneme->phflags |= phVOICELESS;
		break;
	// vowel height
	case hgh:
	case smh:
	case umd:
	case mid:
	case lmd:
	case sml:
	case low:
		// Not supported by eSpeak.
		break;
	// vowel backness
	case fnt:
	case cnt:
	case bck:
		// Not supported by eSpeak.
		break;
	// rounding
	case unr:
	case rnd:
		// Not supported by eSpeak.
		break;
	// articulation
	case lgl:
	case idt:
	case apc:
	case lmn:
		// Not supported by eSpeak.
		break;
	// air flow
	case egs:
	case igs:
		// Not supported by eSpeak.
		break;
	// phonation
	case brv:
	case slv:
	case stv:
	case crv:
	case glc:
		// Not supported by eSpeak.
		break;
	// rounding and labialization
	case ptr:
	case cmp:
	case mrd:
	case lrd:
		// Not supported by eSpeak.
		break;
	// syllabicity
	case syl:
		// Not supported by eSpeak.
		break;
	case nsy:
		phoneme->phflags |= phNONSYLLABIC;
		break;
	// consonant release
	case asp:
	case nrs:
	case lrs:
	case unx:
		// Not supported by eSpeak.
		break;
	// coarticulation
	case pzd:
		phoneme->phflags |= phPALATAL;
		break;
	case vzd:
	case fzd:
	case nzd:
	case rzd:
		// Not supported by eSpeak.
		break;
	// tongue root
	case atr:
	case rtr:
		// Not supported by eSpeak.
		break;
	// fortis and lenis
	case fts:
	case lns:
		// Not supported by eSpeak.
		break;
	// length
	case est:
	case hlg:
		// Not supported by eSpeak.
		break;
	case elg: // FIXME: Should be longer than 'lng'.
	case lng:
		phoneme->phflags |= phLONG;
		break;
	// invalid phoneme feature
	default:
		return ENS_UNKNOWN_PHONEME_FEATURE;
	}
	return ENS_OK;
}
