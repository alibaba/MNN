/* PropList APIs.
 *
 * Copyright (C) 2017-2018 Reece H. Dunn
 *
 * This file is part of ucd-tools.
 *
 * ucd-tools is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ucd-tools is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ucd-tools.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ucd/ucd.h"

static ucd_property properties_Cc(codepoint_t c)
{
	if (c >= 0x0009 && c <= 0x000D) return UCD_PROPERTY_WHITE_SPACE | UCD_PROPERTY_PATTERN_WHITE_SPACE;
	if (c == 0x0085)                return UCD_PROPERTY_WHITE_SPACE | UCD_PROPERTY_PATTERN_WHITE_SPACE;
	return 0;
}

static ucd_property properties_Cf(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c == 0x00AD)                return UCD_PROPERTY_HYPHEN;
		break;
	case 0x0600:
		if (c >= 0x0600 && c <= 0x0605) return UCD_PROPERTY_PREPENDED_CONCATENATION_MARK;
		if (c == 0x061C)                return UCD_PROPERTY_BIDI_CONTROL;
		if (c == 0x06DD)                return UCD_PROPERTY_PREPENDED_CONCATENATION_MARK;
		break;
	case 0x0700:
		if (c == 0x070F)                return UCD_PROPERTY_PREPENDED_CONCATENATION_MARK;
		break;
	case 0x0800:
		if (c == 0x08E2)                return UCD_PROPERTY_PREPENDED_CONCATENATION_MARK;
		break;
	case 0x2000:
		if (c == 0x200C)                return UCD_PROPERTY_JOIN_CONTROL | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x200D)                return UCD_PROPERTY_JOIN_CONTROL | UCD_PROPERTY_EMOJI_COMPONENT;
		if (c >= 0x200E && c <= 0x200F) return UCD_PROPERTY_BIDI_CONTROL | UCD_PROPERTY_PATTERN_WHITE_SPACE;
		if (c >= 0x202A && c <= 0x202E) return UCD_PROPERTY_BIDI_CONTROL;
		if (c >= 0x2061 && c <= 0x2064) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x2066 && c <= 0x2069) return UCD_PROPERTY_BIDI_CONTROL;
		if (c >= 0x206A && c <= 0x206F) return UCD_PROPERTY_DEPRECATED;
		break;
	case 0x011000:
		if (c == 0x0110BD)                  return UCD_PROPERTY_PREPENDED_CONCATENATION_MARK;
		if (c == 0x0110CD)                  return UCD_PROPERTY_PREPENDED_CONCATENATION_MARK;
		break;
	case 0x0E0000:
		if (c == 0x0E0001)                  return UCD_PROPERTY_DEPRECATED;
		if (c == 0x0E0021)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x0E002C)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x0E002E)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x0E003A)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT | ESPEAKNG_PROPERTY_COLON;
		if (c == 0x0E003B)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0x0E003F)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c >= 0x0E0020 && c <= 0x0E007F) return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND | UCD_PROPERTY_EMOJI_COMPONENT;
		break;
	}
	return 0;
}

static ucd_property properties_Cn(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x2000:
		if (c == 0x2065)                return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0x2400:
		if (c >= 0x2427 && c <= 0x243F) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x244B && c <= 0x245F) return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2B00:
		if (c >= 0x2B74 && c <= 0x2B75) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2B96 && c <= 0x2B97) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2BBA && c <= 0x2BBC) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2BC9)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2BD3 && c <= 0x2BEB) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2BF0 && c <= 0x2BFF) return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2E00:
		if (c >= 0x2E45 && c <= 0x2E7F) return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0xFD00:
		if (c >= 0xFDD0 && c <= 0xFDEF) return UCD_PROPERTY_NONCHARACTER_CODE_POINT;
		break;
	case 0xFF00:
		if (c >= 0xFFF0 && c <= 0xFFF8) return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0x1F000:
	case 0x1F100:
	case 0x1F200:
	case 0x1F300:
	case 0x1F400:
	case 0x1F500:
	case 0x1F600:
	case 0x1F700:
	case 0x1F800:
	case 0x1F900:
	case 0x1FA00:
	case 0x1FB00:
	case 0x1FC00:
	case 0x1FD00:
	case 0x1FE00:
	case 0x1FF00:
		if (c >= 0x1FFFE)               return UCD_PROPERTY_NONCHARACTER_CODE_POINT;
		return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	case 0x0E0000:
		if (c == 0xE0000)                 return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		if (c >= 0xE0002 && c <= 0xE001F) return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		if (c >= 0xE0080 && c <= 0xE00FF) return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0x0E0100:
		if (c >= 0xE01F0)                 return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0x0E0200:
	case 0x0E0300:
	case 0x0E0400:
	case 0x0E0500:
	case 0x0E0600:
	case 0x0E0700:
	case 0x0E0800:
	case 0x0E0900:
	case 0x0E0A00:
	case 0x0E0B00:
	case 0x0E0C00:
	case 0x0E0D00:
	case 0x0E0E00:
	case 0x0E0F00:
		return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
	}
	if ((c & 0x0000FFFF) >= 0xFFFE) return UCD_PROPERTY_NONCHARACTER_CODE_POINT;
	return 0;
}

static ucd_property properties_Ll(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c >= 0x0061 && c <= 0x0066) return UCD_PROPERTY_HEX_DIGIT | UCD_PROPERTY_ASCII_HEX_DIGIT;
		if (c >= 0x0069 && c <= 0x006A) return UCD_PROPERTY_SOFT_DOTTED;
		break;
	case 0x0100:
		if (c == 0x012F)                return UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x0149)                return UCD_PROPERTY_DEPRECATED;
		break;
	case 0x0200:
		if (c == 0x0249)                return UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x0268)                return UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x029D)                return UCD_PROPERTY_SOFT_DOTTED;
		break;
	case 0x0300:
		if (c >= 0x03D0 && c <= 0x03D2) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c == 0x03D5)                return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x03F0 && c <= 0x03F1) return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x03F3)                return UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x03F4 && c <= 0x03F5) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x0400:
		if (c == 0x0456)                return UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x0458)                return UCD_PROPERTY_SOFT_DOTTED;
		break;
	case 0x1D00:
		if (c == 0x1D96)                return UCD_PROPERTY_SOFT_DOTTED;
		break;
	case 0x1E00:
		if (c == 0x1E2D)                return UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x1ECB)                return UCD_PROPERTY_SOFT_DOTTED;
		break;
	case 0x2100:
		if (c >= 0x210A && c <= 0x2113) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x212F && c <= 0x2131) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x2133 && c <= 0x2134) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c == 0x2139)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x213C && c <= 0x213F) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x2145 && c <= 0x2147) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x2148 && c <= 0x2149) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		break;
	case 0xFF00:
		if (c >= 0xFF41 && c <= 0xFF46) return UCD_PROPERTY_HEX_DIGIT;
		break;
	case 0x01D400:
		if (c >= 0x01D422 && c <= 0x01D423) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D400 && c <= 0x01D454) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D456 && c <= 0x01D457) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D48A && c <= 0x01D48B) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D458 && c <= 0x01D49C) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D4AE && c <= 0x01D4B9) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c == 0x01D4BB)                  return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D4BE && c <= 0x01D4BF) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D4BD && c <= 0x01D4C3) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D4F2 && c <= 0x01D4F3) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D4C5)                  return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x01D500:
		if                  (c <= 0x01D505) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D526 && c <= 0x01D527) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D51E && c <= 0x01D539) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D55A && c <= 0x01D55B) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D58E && c <= 0x01D58F) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D5C2 && c <= 0x01D5C3) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D5F6 && c <= 0x01D5F7) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D552)                  return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x01D600:
		if (c >= 0x01D62A && c <= 0x01D62B) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D65E && c <= 0x01D65F) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x01D692 && c <= 0x01D693) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_SOFT_DOTTED;
		if                  (c <= 0x01D6A5) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D6C2 && c <= 0x01D6DA) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D6DC && c <= 0x01D6FA) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D6FC)                  return UCD_PROPERTY_OTHER_MATH;
		break;
	case 0x01D700:
		if                  (c <= 0x01D714) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D716 && c <= 0x01D734) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D736 && c <= 0x01D74E) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D750 && c <= 0x01D76E) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D770 && c <= 0x01D788) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D78A && c <= 0x01D7A8) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D7AA && c <= 0x01D7C2) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D7C4 && c <= 0x01D7CB) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	}
	return 0;
}

static ucd_property properties_Lm(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0200:
		if (c == 0x02B2)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x02B0 && c <= 0x02B8) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		if (c >= 0x02B9 && c <= 0x02BF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x02C0 && c <= 0x02C1) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		if (c >= 0x02C6 && c <= 0x02CF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x02D0 && c <= 0x02D1) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_EXTENDER;
		if (c >= 0x02E0 && c <= 0x02E4) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		if (c == 0x02EC)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x02EE)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0300:
		if (c == 0x0374)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x037A)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x0500:
		if (c == 0x0559)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0600:
		if (c == 0x0640)                return UCD_PROPERTY_EXTENDER;
		if (c >= 0x06E5 && c <= 0x06E6) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0700:
		if (c >= 0x07F4 && c <= 0x07F5) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x07FA)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0x0900:
		if (c == 0x0971)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0E00:
		if (c == 0x0E46)                return UCD_PROPERTY_EXTENDER;
		if (c == 0x0EC6)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0x1800:
		if (c == 0x1843)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0x1A00:
		if (c == 0x1AA7)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0x1C00:
		if (c == 0x1C7B)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_EXTENDER;
		if (c >= 0x1C78 && c <= 0x1C7D) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1D00:
		if (c == 0x1D62)                return UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x1D2C && c <= 0x1D6A) return UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_DIACRITIC;
		if (c == 0x1D78)                return UCD_PROPERTY_OTHER_LOWERCASE;
		if (c == 0x1DA4)                return UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x1DA8)                return UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_SOFT_DOTTED;
		if (c >= 0x1D9B && c <= 0x1DBF) return UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x2000:
		if (c == 0x2071)                return UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x207F)                return UCD_PROPERTY_OTHER_LOWERCASE;
		if (c >= 0x2090 && c <= 0x209C) return UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x2C00:
		if (c == 0x2C7C)                return UCD_PROPERTY_OTHER_LOWERCASE | UCD_PROPERTY_SOFT_DOTTED;
		if (c == 0x2C7D)                return UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x2E00:
		if (c == 0x2E2F)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x3000:
		if (c == 0x3005)                return UCD_PROPERTY_EXTENDER;
		if (c >= 0x3031 && c <= 0x3035) return UCD_PROPERTY_EXTENDER;
		if (c >= 0x309D && c <= 0x309E) return UCD_PROPERTY_EXTENDER;
		if (c == 0x30FC)                return UCD_PROPERTY_EXTENDER | UCD_PROPERTY_DIACRITIC;
		if (c >= 0x30FD && c <= 0x30FE) return UCD_PROPERTY_EXTENDER;
		break;
	case 0xA000:
		if (c == 0xA015)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0xA600:
		if (c == 0xA60C)                return UCD_PROPERTY_EXTENDER;
		if (c == 0xA67F)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xA69C && c <= 0xA69D) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0xA700:
		if (c >= 0xA717 && c <= 0xA71F) return UCD_PROPERTY_DIACRITIC;
		if (c == 0xA770)                return UCD_PROPERTY_OTHER_LOWERCASE;
		if (c == 0xA788)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xA7F8 && c <= 0xA7F9) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0xA900:
		if (c == 0xA9CF)                return UCD_PROPERTY_EXTENDER;
		if (c == 0xA9E6)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0xAA00:
		if (c == 0xAA70)                return UCD_PROPERTY_EXTENDER;
		if (c == 0xAADD)                return UCD_PROPERTY_EXTENDER;
		if (c >= 0xAAF3 && c <= 0xAAF4) return UCD_PROPERTY_EXTENDER;
		break;
	case 0xAB00:
		if (c >= 0xAB5C && c <= 0xAB5F) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0xFF00:
		if (c == 0xFF70)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_EXTENDER;
		if (c >= 0xFF9E && c <= 0xFF9F) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x016B00:
		if (c >= 0x016B42 && c <= 0x016B43) return UCD_PROPERTY_EXTENDER;
		break;
	case 0x016F00:
		if (c >= 0x016F93 && c <= 0x016F9F) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x016FE0 && c <= 0x016FE1) return UCD_PROPERTY_EXTENDER;
		break;
	}
	return 0;
}

static ucd_property properties_Lo(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c == 0x00AA)                return UCD_PROPERTY_OTHER_LOWERCASE;
		if (c == 0x00BA)                return UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x0600:
		if (c == 0x0673)                return UCD_PROPERTY_DEPRECATED;
		break;
	case 0x0E00:
		if (c >= 0x0E40 && c <= 0x0E44) return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		if (c == 0x0EAF)                return ESPEAKNG_PROPERTY_ELLIPSIS;
		if (c >= 0x0EC0 && c <= 0x0EC4) return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		break;
	case 0x1100:
		if (c >= 0x115F && c <= 0x1160) return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0x1700:
		if (c >= 0x17A3 && c <= 0x17A4) return UCD_PROPERTY_DEPRECATED;
		break;
	case 0x1900:
		if (c >= 0x19B5 && c <= 0x19B7) return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		if (c == 0x19BA)                return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		break;
	case 0x2100:
		if (c >= 0x2135 && c <= 0x2138) return UCD_PROPERTY_OTHER_MATH;
		break;
	case 0x3000:
		if (c == 0x3006)                return UCD_PROPERTY_IDEOGRAPHIC;
		break;
	case 0x3100:
		if (c == 0x3164)                return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0xAA00:
		if (c >= 0xAAB5 && c <= 0xAAB6) return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		if (c == 0xAAB9)                return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		if (c >= 0xAABB && c <= 0xAABC) return UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION;
		if (c == 0xAAC0)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xAAC2)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xFA00:
		if (c >= 0xFA0E && c <= 0xFA0F) return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c == 0xFA11)                return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0xFA13 && c <= 0xFA14) return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c == 0xFA1F)                return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c == 0xFA21)                return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0xFA23 && c <= 0xFA24) return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0xFA27 && c <= 0xFA29) return UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		break;
	case 0xFF00:
		if (c == 0xFFA0)                return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		break;
	case 0x10D00:
		if (c == 0x10D22)               return UCD_PROPERTY_DIACRITIC;
		if (c == 0x10D23)               return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x18700:
		if (c >= 0x187ED && c <= 0x187F1) return UCD_PROPERTY_IDEOGRAPHIC;
		break;
	case 0x11300:
		if (c == 0x1135D)               return UCD_PROPERTY_EXTENDER;
		break;
	case 0x1EE00:
		return UCD_PROPERTY_OTHER_MATH;
	}
	return 0;
}

static ucd_property properties_Lo_ideographic(codepoint_t c)
{
	switch (c & 0xFFFF0000)
	{
	case 0x000000:
		if (c >= 0x3400 && c <= 0x4DB5) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0x4E00 && c <= 0x9FEF) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0xF900 && c <= 0xFA6D) return UCD_PROPERTY_IDEOGRAPHIC;
		if (c >= 0xFA70 && c <= 0xFAD9) return UCD_PROPERTY_IDEOGRAPHIC;
		break;
	case 0x010000:
		if (c >= 0x017000 && c <= 0x0187EC) return UCD_PROPERTY_IDEOGRAPHIC;
		if (c >= 0x018800 && c <= 0x018AF2) return UCD_PROPERTY_IDEOGRAPHIC;
		if (c >= 0x01B170 && c <= 0x01B2FB) return UCD_PROPERTY_IDEOGRAPHIC;
		break;
	case 0x020000:
		if (c >= 0x020000 && c <= 0x02A6D6) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0x02A700 && c <= 0x02B734) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0x02B740 && c <= 0x02B81D) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0x02B820 && c <= 0x02CEA1) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0x02CEB0 && c <= 0x02EBE0) return UCD_PROPERTY_IDEOGRAPHIC | UCD_PROPERTY_UNIFIED_IDEOGRAPH;
		if (c >= 0x02F800 && c <= 0x02FA1D) return UCD_PROPERTY_IDEOGRAPHIC;
		break;
	}
	return 0;
}

static ucd_property properties_Lu(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c >= 0x0041 && c <= 0x0046) return UCD_PROPERTY_HEX_DIGIT | UCD_PROPERTY_ASCII_HEX_DIGIT;
		break;
	case 0x0300:
		if (c >= 0x03D0 && c <= 0x03D2) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x03F4 && c <= 0x03F5) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0xFF00:
		if (c >= 0xFF21 && c <= 0xFF26) return UCD_PROPERTY_HEX_DIGIT;
		break;
	case 0x2100:
		if (c == 0x2102)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x2107)                return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x210A && c <= 0x2113) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c == 0x2115)                return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x2119 && c <= 0x211D) return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x2124)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x2128)                return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x212C && c <= 0x212D) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x212F && c <= 0x2131) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x2133 && c <= 0x2134) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x213C && c <= 0x213F) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x2145 && c <= 0x2149) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x01D400:
		if (c >= 0x01D400 && c <= 0x01D454) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D456 && c <= 0x01D49C) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D49E && c <= 0x01D49F) return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x01D4A2)                  return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D4A5 && c <= 0x01D4A6) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D4A9 && c <= 0x01D4AC) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D4AE && c <= 0x01D4B9) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D4C5)                  return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x01D500:
		if                  (c <= 0x01D505) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D507 && c <= 0x01D50A) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D50D && c <= 0x01D514) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D516 && c <= 0x01D51C) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D51E && c <= 0x01D539) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D53B && c <= 0x01D53E) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D540 && c <= 0x01D544) return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x01D546)                  return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D54A && c <= 0x01D550) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D552)                  return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x01D600:
		if                  (c <= 0x01D6A5) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D6A8 && c <= 0x01D6C0) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x01D6DC && c <= 0x01D6FA) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	case 0x01D700:
		if (c >= 0x01D716 && c <= 0x01D734) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D750 && c <= 0x01D76E) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D78A && c <= 0x01D7A8) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		if (c >= 0x01D7C4 && c <= 0x01D7CB) return UCD_PROPERTY_OTHER_MATH; /* Ll|Lu */
		break;
	}
	return 0;
}

static ucd_property properties_Mc(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0900:
		if (c == 0x09BE)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x09D7)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x0B00:
		if (c == 0x0B3E)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0B57)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0BBE)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0BD7)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x0C00:
		if (c == 0x0CC2)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c >= 0x0CD5 && c <= 0x0CD6) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x0D00:
		if (c == 0x0D3E)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0D57)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0DCF)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0DDF)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x0F00:
		if (c >= 0x0F3E && c <= 0x0F3F) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1000:
		if (c >= 0x102B && c <= 0x102C) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1031)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1038)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x103B && c <= 0x103C) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1056 && c <= 0x1057) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1062)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1067 && c <= 0x1068) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1083 && c <= 0x1084) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1087 && c <= 0x108C) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x108F)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x109A && c <= 0x109B) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x109C)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		return 0;
	case 0x1B00:
		if (c == 0x1B04)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B35)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B3B)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1B3D && c <= 0x1B41) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B43)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B44)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x1B82)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1BA1)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1BA6 && c <= 0x1BA7) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1BAA)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x1BE7)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1BEA && c <= 0x1BEC) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1BEE)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		return 0;
	case 0x1C00:
		if (c == 0x1CE1)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x1CF7)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x3000:
		if (c >= 0x302E && c <= 0x302F) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0xA900:
		if (c == 0xA953)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xA9C0)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xAA00:
		if (c == 0xAA7B)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xAA7D)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xAB00:
		if (c == 0xABEC)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011100:
		if (c == 0x0111C0)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011200:
		if (c == 0x011235)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011300:
		if (c == 0x01133E)                  return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x01134D)                  return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011357)                  return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x011400:
		if (c == 0x0114B0)                  return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x0114BD)                  return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x011500:
		if (c == 0x0115AF)                  return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	case 0x011600:
		if (c == 0x0116B6)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x01D100:
		if (c == 0x01D165)                  return UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		if (c == 0x01D166)                  return 0;
		if (c == 0x01D16D)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x01D16E && c <= 0x01D172) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_GRAPHEME_EXTEND;
		break;
	}
	return UCD_PROPERTY_OTHER_ALPHABETIC;
}

static ucd_property properties_Me(codepoint_t c)
{
	if (c == 0x20E3) return UCD_PROPERTY_EMOJI_COMPONENT;
	return 0;
}

static ucd_property properties_Mn(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0300:
		if (c >= 0x0300 && c <= 0x0344) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0345)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_LOWERCASE;
		if (c >= 0x0346 && c <= 0x034E) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x034F)                return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		if (c >= 0x0350 && c <= 0x0357) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x035D && c <= 0x0362) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0400:
		if (c >= 0x0483 && c <= 0x0487) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0500:
		if (c >= 0x0591 && c <= 0x05A1) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x05A3 && c <= 0x05AF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x05B0 && c <= 0x05BD) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x05BF)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x05C1 && c <= 0x05C2) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x05C4)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x05C5)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x05C7)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0600:
		if (c >= 0x0610 && c <= 0x061A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x064B && c <= 0x0652) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0653 && c <= 0x0656) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0657)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0658)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0659 && c <= 0x065F) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0670)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x06D6 && c <= 0x06DC) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x06DF && c <= 0x06E0) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x06E1 && c <= 0x06E4) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x06E7 && c <= 0x06E8) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x06EA && c <= 0x06EC) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x06ED)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0700:
		if (c == 0x0711)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0730 && c <= 0x073F) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0740 && c <= 0x074A) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x07A6 && c <= 0x07B0) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x07EB && c <= 0x07F3) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0800:
		if (c >= 0x0816 && c <= 0x0817) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0818 && c <= 0x0819) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x081B && c <= 0x0823) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0825 && c <= 0x0827) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0829 && c <= 0x082C) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x08D4 && c <= 0x08DF) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x08E3 && c <= 0x08E9) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x08EA && c <= 0x08EF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x08F0 && c <= 0x08FE) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x08FF)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0900:
		if                (c <= 0x0902) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x093A)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x093C)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0941 && c <= 0x0948) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x094D)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0951 && c <= 0x0954) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0955 && c <= 0x0957) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0962 && c <= 0x0963) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0981)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x09BC)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x09C1 && c <= 0x09C4) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x09CD)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x09E2 && c <= 0x09E3) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0A00:
		if (c >= 0x0A01 && c <= 0x0A02) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0A3C)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0A41 && c <= 0x0A42) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0A47 && c <= 0x0A48) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0A4B && c <= 0x0A4C) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0A4D)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0A51)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0A70 && c <= 0x0A71) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0A75)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0A81 && c <= 0x0A82) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0ABC)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0AC1 && c <= 0x0AC5) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0AC7 && c <= 0x0AC8) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0ACD)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0AE2 && c <= 0x0AE3) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0AFA && c <= 0x0AFC) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0AFD && c <= 0x0AFF) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0B00:
		if (c == 0x0B01)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0B3C)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0B3F)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0B41 && c <= 0x0B44) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0B4D)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0B56)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0B62 && c <= 0x0B63) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0B82)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0BC0)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0BCD)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0C00:
		if (c == 0x0C00)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0C3E && c <= 0x0C40) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0C46 && c <= 0x0C48) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0C4A && c <= 0x0C4C) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0C4D)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0C55 && c <= 0x0C56) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0C62 && c <= 0x0C63) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0C81)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0CBC)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0CBF)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0CC6)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0CCC)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0CCD)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0CE2 && c <= 0x0CE3) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0D00:
		if (c >= 0x0D00 && c <= 0x0D01) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0D3B && c <= 0x0D3C) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0D41 && c <= 0x0D44) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0D4D)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0D62 && c <= 0x0D63) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0DCA)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0DD2 && c <= 0x0DD4) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0DD6)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0E00:
		if (c == 0x0E31)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0E34 && c <= 0x0E3A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0E47 && c <= 0x0E4C) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0E4D)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0E4E)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0EB1)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0EB4 && c <= 0x0EB9) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0EBB && c <= 0x0EBC) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0EC8 && c <= 0x0ECC) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0ECD)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x0F00:
		if (c >= 0x0F18 && c <= 0x0F19) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0F35)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0F37)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0F39)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x0F77)                return UCD_PROPERTY_DEPRECATED | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0F79)                return UCD_PROPERTY_DEPRECATED | UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0F71 && c <= 0x0F7E) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0F80 && c <= 0x0F81) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0F82 && c <= 0x0F84) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0F86 && c <= 0x0F87) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0F8D && c <= 0x0F97) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0F99 && c <= 0x0FBC) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0FC6)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1000:
		if (c >= 0x102D && c <= 0x1030) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1032 && c <= 0x1036) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1037)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1039 && c <= 0x103A) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x103D && c <= 0x103E) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1058 && c <= 0x1059) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x105E && c <= 0x1060) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1071 && c <= 0x1074) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1082)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1085 && c <= 0x1086) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x108D)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x109D)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x1300:
		if (c == 0x135F)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x1700:
		if (c >= 0x1712 && c <= 0x1713) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1732 && c <= 0x1733) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1752 && c <= 0x1753) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1772 && c <= 0x1773) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x17B4 && c <= 0x17B5) return UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT;
		if (c >= 0x17B7 && c <= 0x17BD) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x17C6)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x17C9 && c <= 0x17D3) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x17DD)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1800:
		if (c >= 0x180B && c <= 0x180D) return UCD_PROPERTY_VARIATION_SELECTOR;
		if (c >= 0x1885 && c <= 0x1886) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_ID_START;
		if (c == 0x18A9)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x1900:
		if (c >= 0x1920 && c <= 0x1922) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1927 && c <= 0x1928) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1932)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1939 && c <= 0x193B) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1A00:
		if (c >= 0x1A17 && c <= 0x1A18) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1A1B)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1A56)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1A58 && c <= 0x1A5E) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1A62)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1A65 && c <= 0x1A6C) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1A73 && c <= 0x1A74) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1A75 && c <= 0x1A7C) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x1A7F)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1AB0 && c <= 0x1ABD) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1B00:
		if (c >= 0x1B00 && c <= 0x1B03) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B34)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1B36 && c <= 0x1B3A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B3C)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1B42)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1B6B && c <= 0x1B73) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1B80 && c <= 0x1B81) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1BA2 && c <= 0x1BA5) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1BA8 && c <= 0x1BA9) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1BAB)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1BAC && c <= 0x1BAD) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1BE8 && c <= 0x1BE9) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1BED)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1BEF && c <= 0x1BF1) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x1C00:
		if (c >= 0x1C2C && c <= 0x1C33) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x1C36)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_EXTENDER;
		if (c == 0x1C37)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1CD0 && c <= 0x1CD2) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1CD4 && c <= 0x1CE0) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1CE2 && c <= 0x1CE8) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x1CED)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x1CF4)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1CF8 && c <= 0x1CF9) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1D00:
		if (c >= 0x1DC4 && c <= 0x1DCF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1DE7 && c <= 0x1DF4) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x1DF5 && c <= 0x1DF9) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1DFD && c <= 0x1DFF) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x2000:
		if (c >= 0x20D0 && c <= 0x20DC) return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x20E1)                return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x20E5 && c <= 0x20E6) return UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x20EB && c <= 0x20EF) return UCD_PROPERTY_OTHER_MATH;
		break;
	case 0x2C00:
		if (c >= 0x2CEF && c <= 0x2CF1) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x2D00:
		if (c >= 0x2DE0 && c <= 0x2DFF) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x3000:
		if (c >= 0x302A && c <= 0x302D) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x3099 && c <= 0x309A) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xA600:
		if (c == 0xA66F)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xA674 && c <= 0xA67B) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xA67C && c <= 0xA67D) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xA69E && c <= 0xA69F) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xA6F0 && c <= 0xA6F1) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xA800:
		if (c >= 0xA825 && c <= 0xA826) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xA8C4)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xA8C5)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xA8E0 && c <= 0xA8F1) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xA900:
		if (c >= 0xA926 && c <= 0xA92A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xA92B && c <= 0xA92D) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xA947 && c <= 0xA951) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xA980 && c <= 0xA982) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xA9B3)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xA9B6 && c <= 0xA9B9) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xA9BC)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xA9E5)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xAA00:
		if (c >= 0xAA29 && c <= 0xAA2E) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xAA31 && c <= 0xAA32) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xAA35 && c <= 0xAA36) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xAA43)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xAA4C)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xAA7C)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xAAB0)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xAAB2 && c <= 0xAAB4) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0xAAB7 && c <= 0xAAB8) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xAABE)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xAABF)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xAAC1)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0xAAEC && c <= 0xAAED) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xAAF6)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xAB00:
		if (c == 0xABE5)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xABE8)                return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0xABED)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xFB00:
		if (c == 0xFB1E)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0xFE00:
		if (c >= 0xFE00 && c <= 0xFE0E) return UCD_PROPERTY_VARIATION_SELECTOR;
		if (c == 0xFE0F)                return UCD_PROPERTY_VARIATION_SELECTOR | UCD_PROPERTY_EMOJI_COMPONENT;
		if (c >= 0xFE20 && c <= 0xFE2F) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x010200:
		if (c == 0x0102E0)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x010300:
		if (c >= 0x010376 && c <= 0x01037A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x010A00:
		if (c >= 0x010A01 && c <= 0x010A03) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x010A05 && c <= 0x010A06) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x010A0C && c <= 0x010A0F) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x010AE5 && c <= 0x010AE6) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x010D00:
		if (c >= 0x010D24 && c <= 0x010D27) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x010F00:
		if (c >= 0x010F46 && c <= 0x010F50) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011000:
		if (c == 0x011001)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011038 && c <= 0x011045) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0110B3 && c <= 0x0110B6) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0110B9 && c <= 0x0110BA) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011100:
		if (c >= 0x011100 && c <= 0x011102) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011127 && c <= 0x01112B) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x01112D && c <= 0x011132) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011133 && c <= 0x011134) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011173)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x011180 && c <= 0x011181) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0111B6 && c <= 0x0111BE) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0111CA && c <= 0x0111CC) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011200:
		if (c >= 0x01122F && c <= 0x011231) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011234)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011236)                  return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011237)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x01123E)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0112DF)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0112E3 && c <= 0x0112E8) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0112E9 && c <= 0x0112EA) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011300:
		if (c >= 0x011300 && c <= 0x011301) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x01133C)                  return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011340)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011366 && c <= 0x01136C) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x011370 && c <= 0x011374) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011400:
		if (c >= 0x011438 && c <= 0x01143F) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011442)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x011443 && c <= 0x011444) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011446)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0114B3 && c <= 0x0114B8) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0114BA)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0114BF && c <= 0x0114C0) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0114C2 && c <= 0x0114C3) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011500:
		if (c >= 0x0115B2 && c <= 0x0115B5) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0115BC && c <= 0x0115BD) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0115BF && c <= 0x0115C0) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0115DC && c <= 0x0115DD) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x011600:
		if (c >= 0x011633 && c <= 0x01163A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x01163D)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x01163F)                  return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011640)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0116AB)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0116AD)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x0116B0 && c <= 0x0116B5) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x0116B7)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011700:
		if (c >= 0x01171D && c <= 0x01171F) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011722 && c <= 0x011725) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011727 && c <= 0x01172A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x01172B)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011800:
		if (c >= 0x01182F && c <= 0x011838) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011839 && c <= 0x01183A) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011A00:
		if (c >= 0x011A01 && c <= 0x011A0A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011A34)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x011A35 && c <= 0x011A3E) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011A47)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x011A51 && c <= 0x011A5B) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011A8A && c <= 0x011A96) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011A98)                  return UCD_PROPERTY_EXTENDER;
		if (c == 0x011A99)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011C00:
		if (c >= 0x011C30 && c <= 0x011C36) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011C38 && c <= 0x011C3D) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011C3F)                  return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x011C92 && c <= 0x011CA7) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011CAA && c <= 0x011CB0) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011CB2 && c <= 0x011CB3) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011CB5 && c <= 0x011CB6) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x011D00:
		if (c >= 0x011D31 && c <= 0x011D36) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011D3A)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011D3C && c <= 0x011D3D) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011D3F && c <= 0x011D41) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011D42)                  return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011D43)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x011D44 && c <= 0x011D45) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x011D47)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011D90)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011D91)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011D95)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011D97)                  return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x011E00:
		if (c == 0x011EF3)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c == 0x011EF4)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x016A00:
		if (c >= 0x016AF0 && c <= 0x016AF4) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x016B00:
		if (c >= 0x016B30 && c <= 0x016B36) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x016F00:
		if (c >= 0x016F8F && c <= 0x016F92) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x01BC00:
		if (c == 0x01BC9E)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x01D100:
		if (c >= 0x01D167 && c <= 0x01D169) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x01D17B && c <= 0x01D182) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x01D185 && c <= 0x01D18B) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x01D1AA && c <= 0x01D1AD) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x01E000:
		if (c >= 0x01E000 && c <= 0x01E006) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x01E008 && c <= 0x01E018) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x01E01B && c <= 0x01E021) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x01E023 && c <= 0x01E024) return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x01E026 && c <= 0x01E02A) return UCD_PROPERTY_OTHER_ALPHABETIC;
		break;
	case 0x01E800:
		if (c >= 0x01E8D0 && c <= 0x01E8D6) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x01E900:
		if (c >= 0x01E944 && c <= 0x01E946) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_EXTENDER;
		if (c == 0x01E947)                  return UCD_PROPERTY_OTHER_ALPHABETIC;
		if (c >= 0x01E948 && c <= 0x01E94A) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0E0100:
		if (c >= 0x0E0100 && c <= 0x0E01EF) return UCD_PROPERTY_VARIATION_SELECTOR;
		break;
	}
	return 0;
}

static ucd_property properties_Nd(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c >= 0x0030 && c <= 0x0039) return UCD_PROPERTY_HEX_DIGIT | UCD_PROPERTY_ASCII_HEX_DIGIT | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EMOJI_COMPONENT;
		break;
	case 0xFF00:
		if (c >= 0xFF10 && c <= 0xFF19) return UCD_PROPERTY_HEX_DIGIT;
		break;
	case 0x01D700:
		if (c >= 0x01D7CE && c <= 0x01D7FF) return UCD_PROPERTY_OTHER_MATH;
		break;
	}
	return 0;
}

static ucd_property properties_Nl(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x2100:
		if (c >= 0x2160 && c <= 0x216F) return UCD_PROPERTY_OTHER_UPPERCASE;
		if (c >= 0x2170 && c <= 0x217F) return UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x3000:
		if (c == 0x3007)                return UCD_PROPERTY_IDEOGRAPHIC;
		if (c >= 0x3021 && c <= 0x3029) return UCD_PROPERTY_IDEOGRAPHIC;
		if (c >= 0x3038 && c <= 0x303A) return UCD_PROPERTY_IDEOGRAPHIC;
		break;
	}
	return 0;
}

static ucd_property properties_No(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x1300:
		if (c >= 0x1369 && c <= 0x1371) return UCD_PROPERTY_OTHER_ID_CONTINUE;
		break;
	case 0x1900:
		if (c == 0x19DA)                return UCD_PROPERTY_OTHER_ID_CONTINUE;
		break;
	case 0x2400:
		if (c >= 0x2488 && c <= 0x249B) return ESPEAKNG_PROPERTY_FULL_STOP;
		break;
	case 0x01F100:
		if (c == 0x01F100)                  return ESPEAKNG_PROPERTY_FULL_STOP;
		if (c >= 0x01F101 && c <= 0x01F10A) return ESPEAKNG_PROPERTY_COMMA;
		break;
	}
	return 0;
}

static ucd_property properties_Pc(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x2000:
		if (c == 0x2040)                return UCD_PROPERTY_OTHER_MATH;
		break;
	}
	return 0;
}

static ucd_property properties_Pd(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x0500:
		if (c == 0x058A)                return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN;
		break;
	case 0x1800:
		if (c == 0x1806)                return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN;
		break;
	case 0x2000:
		if (c >= 0x2010 && c <= 0x2011) return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2013 && c <= 0x2014) return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_EXTENDED_DASH;
		return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2E00:
		if (c == 0x2E17)                return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2E3A && c <= 0x2E3B) return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_EXTENDED_DASH;
		return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x3000:
		if (c == 0x301C)                return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x3030)                return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0xFE00:
		if (c >= 0xFE31 && c <= 0xFE32) return UCD_PROPERTY_DASH | ESPEAKNG_PROPERTY_EXTENDED_DASH;
		if (c == 0xFE63)                return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN | UCD_PROPERTY_OTHER_MATH;
		break;
	case 0xFF00:
		if (c == 0xFF0D)                return UCD_PROPERTY_DASH | UCD_PROPERTY_HYPHEN;
		break;
	}
	return UCD_PROPERTY_DASH;
}

static ucd_property properties_Pe(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2000:
		if (c == 0x2046)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x207E)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x208E)                return UCD_PROPERTY_OTHER_MATH;
		break;
	case 0x2300:
		if (c == 0x2309)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x230B)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x232A)                return UCD_PROPERTY_DEPRECATED | UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2700:
		if (c == 0x27C6)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x27E6 && c <= 0x27EF) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX; /* Pe|Ps */
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2900:
		return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX; /* Pe|Ps */
	case 0x2E00:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x3000:
		if (c == 0x300D)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_QUOTATION_MARK;
		if (c == 0x300F)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_QUOTATION_MARK;
		if (c >= 0x301E && c <= 0x301F) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_QUOTATION_MARK;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0xFD00:
		if (c == 0xFD3E)                return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0xFE00:
		if (c == 0xFE42)                return UCD_PROPERTY_QUOTATION_MARK;
		if (c == 0xFE44)                return UCD_PROPERTY_QUOTATION_MARK;
		break;
	case 0xFF00:
		if (c == 0xFF63)                return UCD_PROPERTY_QUOTATION_MARK;
		break;
	}
	return 0;
}

static ucd_property properties_Pf(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
	case 0x2000:
		return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2E00:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	}
	return 0;
}

static ucd_property properties_Pi(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
	case 0x2000:
		return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2E00:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	}
	return 0;
}

static ucd_property properties_Po(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c == 0x0021)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x0022)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x0023)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI_COMPONENT;
		if (c == 0x0027)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x002A)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI_COMPONENT;
		if (c == 0x002C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x002E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x003A)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COLON;
		if (c == 0x003B)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0x003F)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x00A1)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER | ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION;
		if (c == 0x00B7)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_EXTENDER | UCD_PROPERTY_OTHER_ID_CONTINUE;
		if (c == 0x00BF)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER | ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x0300:
		if (c == 0x037E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x0387)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_OTHER_ID_CONTINUE | ESPEAKNG_PROPERTY_SEMI_COLON;
		break;
	case 0x0500:
		if (c >= 0x055B && c <= 0x055C) return ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD;
		if (c == 0x055D)                return ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x055E)                return ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD;
		if (c == 0x0589)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0x05C3)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x0600:
		if (c == 0x060C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x061B)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0x061E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x061F)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x06D4)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		break;
	case 0x0700:
		if (c == 0x0700)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR;
		if (c == 0x0701)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x0702)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x0703)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x0704)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x0705)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x0706 && c <= 0x0707) return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		if (c == 0x0708)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0x0709)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x070A)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x070C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x07F8)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x07F9)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		break;
	case 0x0800:
		if (c == 0x0837)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x0839)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x083D)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x083E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x0830 && c <= 0x083E) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x085E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x0900:
		if (c == 0x0964)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0x0965)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR;
		break;
	case 0x0D00:
		if (c == 0x0DF4)                return ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		break;
	case 0x0E00:
		if (c >= 0x0E5A && c <= 0x0E5B) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x0F00:
		if (c == 0x0F08)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x0F0D)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0x0F0E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR;
		if (c >= 0x0F0E && c <= 0x0F12) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x0F14)                return ESPEAKNG_PROPERTY_COMMA;
		break;
	case 0x1000:
		if (c >= 0x104A && c <= 0x104B) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x10FB)                return ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR;
		break;
	case 0x1300:
		if (c == 0x1361)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x1362)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x1363)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x1364)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c >= 0x1365 && c <= 0x1366) return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		if (c == 0x1367)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x1368)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR;
		break;
	case 0x1600:
		if (c == 0x166D)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x166E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c >= 0x16EB && c <= 0x16ED) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x1700:
		if (c >= 0x1735 && c <= 0x1736) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x17D4 && c <= 0x17D6) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x17DA)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x1800:
		if (c == 0x1801)                return ESPEAKNG_PROPERTY_ELLIPSIS;
		if (c == 0x1802)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x1803)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x1804)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		if (c == 0x1805)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x1808)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x1809)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x180A)                return UCD_PROPERTY_EXTENDER;
		break;
	case 0x1900:
		if (c == 0x1944)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x1945)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		break;
	case 0x1A00:
		if (c >= 0x1AA8 && c <= 0x1AAB) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x1B00:
		if (c >= 0x1B5A && c <= 0x1B5B) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x1B5D)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x1B5E && c <= 0x1B5F) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x1C00:
		if (c >= 0x1C3B && c <= 0x1C3C) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x1C3D && c <= 0x1C3F) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x1C7E && c <= 0x1C7F) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x1CD3)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x2000:
		if (c == 0x2016)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2017)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2026)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_ELLIPSIS;
		if (c >= 0x2020 && c <= 0x2027) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2032 && c <= 0x2034) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_OTHER_MATH;
		if (c >= 0x2030 && c <= 0x2038) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x203C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_EXCLAMATION_MARK | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x203D)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x203B && c <= 0x203E) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2041 && c <= 0x2043) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2047)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x2048)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x2049)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_EXCLAMATION_MARK | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x204F)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c >= 0x204A && c <= 0x2051) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2053)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_DASH;
		if (c >= 0x2055 && c <= 0x205E) return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2C00:
		if (c == 0x2CF9)                return ESPEAKNG_PROPERTY_FULL_STOP;
		if (c >= 0x2CFA && c <= 0x2CFB) return ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0x2CFE)                return ESPEAKNG_PROPERTY_FULL_STOP;
		break;
	case 0x2E00:
		if (c == 0x2E2E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2E32)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x2E33)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x2E34)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x2E35)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0x2E3C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x2E41)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x2E4C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2E4E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x3000:
		if (c == 0x3001)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COMMA | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0x3002)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0x3003)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x303D)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x30FB)                return UCD_PROPERTY_HYPHEN;
		break;
	case 0xA400:
		if (c == 0xA4FE)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0xA4FF)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		break;
	case 0xA600:
		if (c == 0xA60D)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0xA60E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0xA60F)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0xA6F3)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0xA6F4)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		if (c == 0xA6F5)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0xA6F6)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0xA6F7)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		break;
	case 0xA800:
		if (c >= 0xA876 && c <= 0xA877) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0xA8CE && c <= 0xA8CF) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0xA900:
		if (c == 0xA92E)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xA92F)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0xA9C7)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0xA9C8 && c <= 0xA9C9) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0xAA00:
		if (c >= 0xAA5D && c <= 0xAA5F) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0xAADF)                return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0xAAF0 && c <= 0xAAF1) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0xAB00:
		if (c == 0xABEB)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0xFE00:
		if (c >= 0xFE10 && c <= 0xFE11) return ESPEAKNG_PROPERTY_COMMA;
		if (c == 0xFE12)                return ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0xFE13)                return ESPEAKNG_PROPERTY_COLON;
		if (c == 0xFE14)                return ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0xFE15)                return ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0xFE16)                return ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0xFE19)                return ESPEAKNG_PROPERTY_ELLIPSIS;
		if (c >= 0xFE45 && c <= 0xFE46) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0xFE50 && c <= 0xFE51) return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0xFE52)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0xFE54)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0xFE55)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		if (c == 0xFE56)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c == 0xFE57)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0xFE61)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0xFE68)                return UCD_PROPERTY_OTHER_MATH;
		break;
	case 0xFF00:
		if (c == 0xFF01)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0xFF02)                return UCD_PROPERTY_QUOTATION_MARK;
		if (c == 0xFF07)                return UCD_PROPERTY_QUOTATION_MARK;
		if (c == 0xFF0C)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0xFF0E)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0xFF3C)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0xFF65)                return UCD_PROPERTY_HYPHEN;
		if (c == 0xFF1A)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0xFF1B)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0xFF1F)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER;
		if (c == 0xFF61)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0xFF64)                return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		break;
	case 0x10300:
		if (c == 0x01039F)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x0103D0)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x10800:
		if (c == 0x010857)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x10900:
		if (c == 0x01091F)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x10A00:
		if (c >= 0x010A56 && c <= 0x010A57) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x010AF0 && c <= 0x010AF5) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x10B00:
		if (c >= 0x010B3A && c <= 0x010B3F) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x010B99 && c <= 0x010B9C) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x10F00:
		if (c >= 0x10F55 && c <= 0x10F59)   return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11000:
		if (c >= 0x011047 && c <= 0x011048) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x011049 && c <= 0x01104D) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x0110BE && c <= 0x0110C1) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11100:
		if (c >= 0x011141 && c <= 0x011142) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x011143)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_QUESTION_MARK;
		if (c >= 0x0111C5 && c <= 0x0111C6) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x0111CD)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x0111DE && c <= 0x0111DF) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11200:
		if (c == 0x01123A)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x011238 && c <= 0x01123C) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x0112A9)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11400:
		if (c >= 0x01144B && c <= 0x01144C) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x01144D)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x01145B)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x11500:
		if (c >= 0x0115C2 && c <= 0x0115C3) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x0115C4 && c <= 0x0115C5) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c >= 0x0115C6 && c <= 0x0115C8) return UCD_PROPERTY_EXTENDER;
		if (c >= 0x0115C9 && c <= 0x0115D7) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11600:
		if (c >= 0x011641 && c <= 0x011642) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11700:
		if (c >= 0x01173C && c <= 0x01173E) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x11A00:
		if (c >= 0x011A42 && c <= 0x011A43) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x011A9B && c <= 0x011A9C) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c >= 0x011AA1 && c <= 0x011AA2) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x11C00:
		if (c >= 0x011C41 && c <= 0x011C42) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x011C43)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x011C71)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x11E00:
		if (c >= 0x11EF7 && c <= 0x11EF8)   return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x12400:
		if (c >= 0x012471 && c <= 0x012472) return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		if (c >= 0x012470 && c <= 0x012474) return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		break;
	case 0x16E00:
		if (c == 0x016E97)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x016E98)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x16A00:
		if (c >= 0x016A6E && c <= 0x016A6F) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x016AF5)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		break;
	case 0x16B00:
		if (c >= 0x016B37 && c <= 0x016B38) return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		if (c == 0x016B39)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION;
		if (c == 0x016B44)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL;
		break;
	case 0x1BC00:
		if (c == 0x01BC9F)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		break;
	case 0x1DA00:
		if (c == 0x01DA87)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COMMA;
		if (c == 0x01DA88)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | UCD_PROPERTY_SENTENCE_TERMINAL | ESPEAKNG_PROPERTY_FULL_STOP;
		if (c == 0x01DA89)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_SEMI_COLON;
		if (c == 0x01DA8A)                  return UCD_PROPERTY_TERMINAL_PUNCTUATION | ESPEAKNG_PROPERTY_COLON;
		break;
	case 0x1E900:
		if (c == 0x01E95E)              return ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x01E95F)              return ESPEAKNG_PROPERTY_QUESTION_MARK;
		break;
	}
	return 0;
}

static ucd_property properties_Ps(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2000:
		if (c == 0x201A)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x201E)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2045)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x207D)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x208D)                return UCD_PROPERTY_OTHER_MATH;
		break;
	case 0x2300:
		if (c == 0x2308)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x230A)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2329)                return UCD_PROPERTY_DEPRECATED | UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2700:
		if (c == 0x27C5)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x27E6 && c <= 0x27EF) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX; /* Pe|Ps */
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2900:
		return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2E00:
		if (c == 0x2E42)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x3000:
		if (c == 0x300C)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x300E)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x301D)                return UCD_PROPERTY_QUOTATION_MARK | UCD_PROPERTY_PATTERN_SYNTAX;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0xFD00:
		if (c == 0xFD3F)                return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0xFE00:
		if (c == 0xFE41)                return UCD_PROPERTY_QUOTATION_MARK;
		if (c == 0xFE43)                return UCD_PROPERTY_QUOTATION_MARK;
		break;
	case 0xFF00:
		if (c == 0xFF62)                return UCD_PROPERTY_QUOTATION_MARK;
		break;
	}
	return 0;
}

static ucd_property properties_Sc(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	}
	return 0;
}

static ucd_property properties_Sk(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c == 0x005E)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x0060)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x00A8)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x00AF)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x00B4)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0x00B8)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0200:
		if (c >= 0x02C2 && c <= 0x02C5) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x02D2 && c <= 0x02DF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x02E5 && c <= 0x02EB) return UCD_PROPERTY_DIACRITIC;
		if (c == 0x02ED)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x02EF && c <= 0x02FF) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x0300:
		if (c == 0x0375)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x0384 && c <= 0x0385) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x1F00:
		if (c == 0x1FBD)                return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1FBF && c <= 0x1FC1) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1FCD && c <= 0x1FCF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1FDD && c <= 0x1FDF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1FED && c <= 0x1FEF) return UCD_PROPERTY_DIACRITIC;
		if (c >= 0x1FFD && c <= 0x1FFE) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x3000:
		if (c >= 0x309B && c <= 0x309C) return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_ID_START;
		break;
	case 0xA700:
		if (c >= 0xA720 && c <= 0xA721) return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xAB00:
		if (c == 0xAB5B)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0xFF00:
		if (c == 0xFF3E)                return UCD_PROPERTY_DIACRITIC | UCD_PROPERTY_OTHER_MATH;
		if (c == 0xFF40)                return UCD_PROPERTY_DIACRITIC;
		if (c == 0xFFE3)                return UCD_PROPERTY_DIACRITIC;
		break;
	case 0x01F300:
		return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER | UCD_PROPERTY_EMOJI_COMPONENT;
	}
	return 0;
}

static ucd_property properties_Sm(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2000:
		if (c == 0x2044)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2052)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x207B)                return UCD_PROPERTY_DASH;
		if (c == 0x208B)                return UCD_PROPERTY_DASH;
		break;
	case 0x2100:
		if (c == 0x2118)                return UCD_PROPERTY_OTHER_ID_START;
		if (c == 0x2194)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2190)                return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2200:
		if (c == 0x2212)                return UCD_PROPERTY_DASH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x22EE && c <= 0x22F1) return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_ELLIPSIS;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2300:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2500:
		if (c >= 0x25FB && c <= 0x25FC) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x25FD && c <= 0x25FE) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2600:
		if (c == 0x266F)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	case 0x2700:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2900:
		if (c >= 0x2934 && c <= 0x2935) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2982)                return UCD_PROPERTY_PATTERN_SYNTAX | ESPEAKNG_PROPERTY_COLON;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2A00:
	case 0x2B00:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	}
	return 0;
}

static ucd_property properties_So_002600(codepoint_t c)
{
	switch (c & 0xFFFFFFF0)
	{
	case 0x2600:
		if (c <= 0x2604)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2605)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2606)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x260E)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2610:
		if (c == 0x2611)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2613)                return 0;
		if (c == 0x2614)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x2615)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x2618)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x261D)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		break;
	case 0x2620:
		if (c == 0x2620)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2622 && c <= 0x2623) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2626)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x262A)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x262E)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2630:
		if (c >= 0x2638 && c <= 0x263A) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2640:
		if (c == 0x2640)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2642)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2648)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x2650:
		if (c <= 0x2653)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x265F)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2660:
		if (c == 0x2606)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x2660)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2661 && c <= 0x2662) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2663)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2665 && c <= 0x2666) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2668)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x266D && c <= 0x266E) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2670:
		if (c == 0x267B)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x267E)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x267F)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x2680:
		if (c >= 0x2686)                return 0;
		break;
	case 0x2690:
		if (c == 0x2693)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x2692 && c <= 0x2697) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2699)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x269B && c <= 0x269C) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x26A0:
		if (c == 0x26A0)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26A1)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x26AA && c <= 0x26AB) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x26B0:
		if (c >= 0x26B0 && c <= 0x26B1) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x26BD && c <= 0x26BE) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x26C0:
		if (c >= 0x26C4 && c <= 0x26C5) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x26C8)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26CE)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x26CF)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x26D0:
		if (c == 0x26D1)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26D3)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26D4)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x26E0:
		if (c == 0x26E9)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26EA)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x26F0:
		if (c <= 0x26F1)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26F4)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c <= 0x26F5)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x26F7 && c <= 0x26F8) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x26F9)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x26FA)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x26FD)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	}
	return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
}

static ucd_property properties_So_002700(codepoint_t c)
{
	switch (c & 0xFFFFFFF0)
	{
	case 0x2700:
		if (c == 0x2702)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c <= 0x2704)                return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2705)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x270A && c <= 0x270B) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x270C && c <= 0x270D) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x2708 && c <= 0x270D) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x270E)                return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x270F)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2710:
		if (c <= 0x2711)                return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2712)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2714)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2716)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x271D)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2720:
		if (c == 0x2721)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2728)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x2730:
		if (c == 0x2733)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2734)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2740:
		if (c == 0x2744)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2747)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x274C)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x274E)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x2750:
		if (c >= 0x2753 && c <= 0x2754) return ESPEAKNG_PROPERTY_QUESTION_MARK | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x2755)                return ESPEAKNG_PROPERTY_EXCLAMATION_MARK | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x2757)                return ESPEAKNG_PROPERTY_EXCLAMATION_MARK | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x2760:
		if (c == 0x2762)                return ESPEAKNG_PROPERTY_EXCLAMATION_MARK;
		if (c == 0x2763)                return ESPEAKNG_PROPERTY_EXCLAMATION_MARK | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2764)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2765)                return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x2790:
		if (c >= 0x2795 && c <= 0x2797) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	case 0x27A0:
		if (c == 0x27A1)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x27B0:
		if (c == 0x27B0)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x27BF)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		break;
	}
	return 0;
}

static ucd_property properties_So(codepoint_t c)
{
	switch (c & 0xFFFFFF00)
	{
	case 0x0000:
		if (c == 0x00A9)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x00AE)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2100:
		if (c == 0x2122)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x2129)                return UCD_PROPERTY_OTHER_MATH;
		if (c == 0x212E)                return UCD_PROPERTY_OTHER_ID_START;
		if (c == 0x21A8)                return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x21A9)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x21AA)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2195 && c <= 0x2199) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x219C && c <= 0x21AD) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21B0 && c <= 0x21B1) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21B6 && c <= 0x21B7) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21AF && c <= 0x21BB) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21BC && c <= 0x21CD) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21D0 && c <= 0x21D1) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x21D3)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21D5 && c <= 0x21DB) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x21DD)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21E4 && c <= 0x21E5) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x21D5 && c <= 0x21F3) return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2300:
		if                (c <= 0x2307) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x231A && c <= 0x231B) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x230C && c <= 0x231F) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x2322 && c <= 0x2327) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2328)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x232B && c <= 0x237B) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x2388)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x237D && c <= 0x239A) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x23B4 && c <= 0x23B5) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x23B7)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x23CF)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x23D0)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x23B4 && c <= 0x23DB) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x23E2)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x23E9 && c <= 0x23EC) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x23F0)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x23F3)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x23E9 && c <= 0x23F3) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x23F8 && c <= 0x23FA) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x23E3)                return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x2400:
		if (c >= 0x2400 && c <= 0x244A) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x24C2)                return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x24B6 && c <= 0x24CF) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE;
		if (c >= 0x24D0 && c <= 0x24E9) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_LOWERCASE;
		break;
	case 0x2500:
		if (c >= 0x25A0 && c <= 0x25A1) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x25AA && c <= 0x25AB) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x25AE && c <= 0x25B5) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x25B6)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x25BC && c <= 0x25BF) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x25C0)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x25C6 && c <= 0x25C7) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x25CA && c <= 0x25CB) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x25CF && c <= 0x25D3) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x25E2)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x25E4)                return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		if (c >= 0x25E7 && c <= 0x25EC) return UCD_PROPERTY_OTHER_MATH | UCD_PROPERTY_PATTERN_SYNTAX;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2600:
		return properties_So_002600(c) | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2700:
		return properties_So_002700(c) | UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2800:
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2B00:
		if (c >= 0x2B05 && c <= 0x2B07) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x2B1B && c <= 0x2B1C) return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x2B50)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x2B55)                return UCD_PROPERTY_PATTERN_SYNTAX | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		return UCD_PROPERTY_PATTERN_SYNTAX;
	case 0x2E00:
		if (c >= 0x2E80 && c <= 0x2E99) return UCD_PROPERTY_RADICAL;
		if (c >= 0x2E9B && c <= 0x2EF3) return UCD_PROPERTY_RADICAL;
		break;
	case 0x2F00:
		if                (c <= 0x2FD5) return UCD_PROPERTY_RADICAL;
		if (c >= 0x2FF0 && c <= 0x2FF1) return UCD_PROPERTY_IDS_BINARY_OPERATOR;
		if (c >= 0x2FF2 && c <= 0x2FF3) return UCD_PROPERTY_IDS_TRINARY_OPERATOR;
		if (c >= 0x2FF4 && c <= 0x2FFB) return UCD_PROPERTY_IDS_BINARY_OPERATOR;
		break;
	case 0x3000:
		if (c >= 0x3012 && c <= 0x3013) return UCD_PROPERTY_PATTERN_SYNTAX;
		if (c == 0x3020)                return UCD_PROPERTY_PATTERN_SYNTAX;
		break;
	case 0x3200:
		if (c == 0x3297)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x3299)                return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x01F000:
		if (c == 0x01F004)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F0CF)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	case 0x01F100:
		if (c == 0x01F12F)                  return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F130 && c <= 0x01F149) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE;
		if (c >= 0x01F150 && c <= 0x01F169) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE;
		if (c >= 0x01F170 && c <= 0x01F171) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F17E && c <= 0x01F17F) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE | UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F170 && c <= 0x01F189) return UCD_PROPERTY_OTHER_ALPHABETIC | UCD_PROPERTY_OTHER_UPPERCASE;
		if (c == 0x01F18E)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F191 && c <= 0x01F19A) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F1E6)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_REGIONAL_INDICATOR | UCD_PROPERTY_EMOJI_COMPONENT;
		break;
	case 0x01F200:
		if (c == 0x01F201)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F202)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F21A)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F22F)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F237)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F232 && c <= 0x01F23A) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F250 && c <= 0x01F251) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F260)                  return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x01F300:
		if                  (c <= 0x01F320) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F321)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F32D && c <= 0x01F335) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F337 && c <= 0x01F37C) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F385)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F37E && c <= 0x01F393) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F324 && c <= 0x01F393) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F396 && c <= 0x01F397) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F399 && c <= 0x01F39B) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F3C2 && c <= 0x01F3C4) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F3C7)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F3A0 && c <= 0x01F3C9) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F3CA)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F3CB && c <= 0x01F3CC) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F3CF && c <= 0x01F3D3) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F3E0 && c <= 0x01F3F0) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F39E && c <= 0x01F3F0) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F3F3)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F3F4)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F3F5)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F3F7)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F3F8 && c <= 0x01F3FA) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	case 0x01F400:
		if (c == 0x01F43F)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F441)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F442 && c <= 0x01F443) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F446 && c <= 0x01F450) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F466 && c <= 0x01F469) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F46E)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F470 && c <= 0x01F478) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F47C)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F481 && c <= 0x01F483) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F485 && c <= 0x01F487) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F4AA)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F4FE)                  return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F4FD)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
	case 0x01F500:
		if                  (c <= 0x01F53D) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if                  (c <= 0x01F545) return 0;
		if (c >= 0x01F549 && c <= 0x01F54A) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F54B && c <= 0x01F54E) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F550 && c <= 0x01F567) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F56F && c <= 0x01F570) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F574 && c <= 0x01F575) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F573 && c <= 0x01F579) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F57A)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F587)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F58A && c <= 0x01F58D) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F590)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F595 && c <= 0x01F596) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F5A4)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F5A5)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5A8)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F5B1 && c <= 0x01F5B2) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5BC)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F5C2 && c <= 0x01F5C4) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F5D1 && c <= 0x01F5D3) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F5DC && c <= 0x01F5DE) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5E1)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5E3)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5E8)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5EF)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5F3)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F5FA)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F5FB)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	case 0x01F600:
		if (c >= 0x01F645 && c <= 0x01F647) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F64B && c <= 0x01F64F) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if                  (c <= 0x01F64F) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F650 && c <= 0x01F67F) return 0;
		if (c == 0x01F6A3)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F6B4 && c <= 0x01F6B6) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F6C0)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F680 && c <= 0x01F6C5) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F6CC)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F6CB && c <= 0x01F6CF) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F6D0 && c <= 0x01F6D2) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c >= 0x01F6E0 && c <= 0x01F6E5) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F6E9)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F6EB && c <= 0x01F6EC) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		if (c == 0x01F6F0)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c == 0x01F6F3)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		if (c >= 0x01F6F4 && c <= 0x01F6F9) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
		return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	case 0x01F700:
		if (c >= 0x01F7D5 && c <= 0x01F7D8) return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
		break;
	case 0x01F900:
		if (c <= 0x01F90B)                  return 0;
		if (c >= 0x01F918 && c <= 0x01F91C) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F91E && c <= 0x01F91F) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F926)                  return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F930 && c <= 0x01F939) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F93B)                  return 0;
		if (c >= 0x01F93D && c <= 0x01F93E) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c == 0x01F946)                  return 0;
		if (c >= 0x01F9B0 && c <= 0x01F9B3) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_COMPONENT;
		if (c >= 0x01F9B5 && c <= 0x01F9B6) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F9B8 && c <= 0x01F9B9) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		if (c >= 0x01F9D1 && c <= 0x01F9DD) return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION | UCD_PROPERTY_EMOJI_MODIFIER_BASE;
		return UCD_PROPERTY_EMOJI | UCD_PROPERTY_EXTENDED_PICTOGRAPHIC | UCD_PROPERTY_EMOJI_PRESENTATION;
	case 0x01FA00:
		return UCD_PROPERTY_EXTENDED_PICTOGRAPHIC;
	}
	return 0;
}

static ucd_property properties_Zs(codepoint_t c)
{
	if (c == 0x0020) return UCD_PROPERTY_WHITE_SPACE | UCD_PROPERTY_PATTERN_WHITE_SPACE;
	return UCD_PROPERTY_WHITE_SPACE;
}

ucd_property ucd_properties(codepoint_t c, ucd_category category)
{
	switch (category)
	{
	case UCD_CATEGORY_Cc: return properties_Cc(c);
	case UCD_CATEGORY_Cf: return properties_Cf(c);
	case UCD_CATEGORY_Cn: return properties_Cn(c);
	case UCD_CATEGORY_Ll: return properties_Ll(c);
	case UCD_CATEGORY_Lm: return properties_Lm(c);
	case UCD_CATEGORY_Lo: return properties_Lo(c) | properties_Lo_ideographic(c);
	case UCD_CATEGORY_Lu: return properties_Lu(c);
	case UCD_CATEGORY_Mc: return properties_Mc(c);
	case UCD_CATEGORY_Me: return properties_Me(c);
	case UCD_CATEGORY_Mn: return properties_Mn(c);
	case UCD_CATEGORY_Nd: return properties_Nd(c);
	case UCD_CATEGORY_Nl: return properties_Nl(c);
	case UCD_CATEGORY_No: return properties_No(c);
	case UCD_CATEGORY_Pc: return properties_Pc(c);
	case UCD_CATEGORY_Pd: return properties_Pd(c);
	case UCD_CATEGORY_Pe: return properties_Pe(c);
	case UCD_CATEGORY_Pf: return properties_Pf(c);
	case UCD_CATEGORY_Pi: return properties_Pi(c);
	case UCD_CATEGORY_Po: return properties_Po(c);
	case UCD_CATEGORY_Ps: return properties_Ps(c);
	case UCD_CATEGORY_Sc: return properties_Sc(c);
	case UCD_CATEGORY_Sk: return properties_Sk(c);
	case UCD_CATEGORY_Sm: return properties_Sm(c);
	case UCD_CATEGORY_So: return properties_So(c);
	case UCD_CATEGORY_Zl: return UCD_PROPERTY_WHITE_SPACE | UCD_PROPERTY_PATTERN_WHITE_SPACE;
	case UCD_CATEGORY_Zp: return UCD_PROPERTY_WHITE_SPACE | UCD_PROPERTY_PATTERN_WHITE_SPACE | ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR;
	case UCD_CATEGORY_Zs: return properties_Zs(c);
	default:              return 0; /* Co Cs Ii Lt */
	};
}
