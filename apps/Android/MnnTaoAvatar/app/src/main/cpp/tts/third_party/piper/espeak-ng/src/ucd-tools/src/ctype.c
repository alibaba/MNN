/* ctype-style APIs.
 *
 * Copyright (C) 2012-2018 Reece H. Dunn
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

int ucd_isalnum(codepoint_t c)
{
	ucd_category cat = ucd_lookup_category(c);
	switch (cat)
	{
	case UCD_CATEGORY_Lu:
	case UCD_CATEGORY_Ll:
	case UCD_CATEGORY_Lt:
	case UCD_CATEGORY_Lm:
	case UCD_CATEGORY_Lo:
	case UCD_CATEGORY_Nl:
	case UCD_CATEGORY_Nd:
	case UCD_CATEGORY_No:
		return 1;
	case UCD_CATEGORY_Mn:
	case UCD_CATEGORY_Mc:
	case UCD_CATEGORY_So:
		return (ucd_properties(c, cat) & UCD_PROPERTY_OTHER_ALPHABETIC) == UCD_PROPERTY_OTHER_ALPHABETIC;
	default:
		return 0;
	}
}

int ucd_isalpha(codepoint_t c)
{
	ucd_category cat = ucd_lookup_category(c);
	switch (cat)
	{
	case UCD_CATEGORY_Lu:
	case UCD_CATEGORY_Ll:
	case UCD_CATEGORY_Lt:
	case UCD_CATEGORY_Lm:
	case UCD_CATEGORY_Lo:
	case UCD_CATEGORY_Nl:
		return 1;
	case UCD_CATEGORY_Mn:
	case UCD_CATEGORY_Mc:
	case UCD_CATEGORY_So:
		return (ucd_properties(c, cat) & UCD_PROPERTY_OTHER_ALPHABETIC) == UCD_PROPERTY_OTHER_ALPHABETIC;
	default:
		return 0;
	}
}

int ucd_isblank(codepoint_t c)
{
	switch (ucd_lookup_category(c))
	{
	case UCD_CATEGORY_Zs:
		switch (c) /* Exclude characters with the <noBreak> DispositionType */
		{
		case 0x00A0: /* U+00A0 : NO-BREAK SPACE */
		case 0x2007: /* U+2007 : FIGURE SPACE */
		case 0x202F: /* U+202F : NARROW NO-BREAK SPACE */
			return 0;
		}
		return 1;
	case UCD_CATEGORY_Cc:
		return c == 0x09; /* U+0009 : CHARACTER TABULATION */
	default:
		return 0;
	}
}

int ucd_iscntrl(codepoint_t c)
{
	return ucd_lookup_category(c) == UCD_CATEGORY_Cc;
}

int ucd_isdigit(codepoint_t c)
{
	return (c >= 0x30 && c <= 0x39); /* [0-9] */
}

int ucd_isgraph(codepoint_t c)
{
	switch (ucd_lookup_category(c))
	{
	case UCD_CATEGORY_Cc:
	case UCD_CATEGORY_Cf:
	case UCD_CATEGORY_Cn:
	case UCD_CATEGORY_Co:
	case UCD_CATEGORY_Cs:
	case UCD_CATEGORY_Zl:
	case UCD_CATEGORY_Zp:
	case UCD_CATEGORY_Zs:
	case UCD_CATEGORY_Ii:
		return 0;
	default:
		return 1;
	}
}

int ucd_islower(codepoint_t c)
{
	ucd_category cat = ucd_lookup_category(c);
	switch (cat)
	{
	case UCD_CATEGORY_Ll:
		return 1;
	case UCD_CATEGORY_Lt:
		return ucd_toupper(c) != c;
	case UCD_CATEGORY_Lm:
	case UCD_CATEGORY_Lo:
	case UCD_CATEGORY_Mn:
	case UCD_CATEGORY_Nl:
	case UCD_CATEGORY_So:
		return (ucd_properties(c, cat) & UCD_PROPERTY_OTHER_LOWERCASE) == UCD_PROPERTY_OTHER_LOWERCASE;
	default:
		return 0;
	}
}

int ucd_isprint(codepoint_t c)
{
	switch (ucd_lookup_category(c))
	{
	case UCD_CATEGORY_Cc:
	case UCD_CATEGORY_Cf:
	case UCD_CATEGORY_Cn:
	case UCD_CATEGORY_Co:
	case UCD_CATEGORY_Cs:
	case UCD_CATEGORY_Ii:
		return 0;
	default:
		return 1;
	}
}

int ucd_ispunct(codepoint_t c)
{
	return ucd_isgraph(c) && !ucd_isalnum(c);
}

int ucd_isspace(codepoint_t c)
{
	switch (ucd_lookup_category(c))
	{
	case UCD_CATEGORY_Zl:
	case UCD_CATEGORY_Zp:
		return 1;
	case UCD_CATEGORY_Zs:
		switch (c) /* Exclude characters with the <noBreak> DispositionType */
		{
		case 0x00A0: /* U+00A0 : NO-BREAK SPACE */
		case 0x2007: /* U+2007 : FIGURE SPACE */
		case 0x202F: /* U+202F : NARROW NO-BREAK SPACE */
			return 0;
		}
		return 1;
	case UCD_CATEGORY_Cc:
		switch (c) /* Include control characters marked as White_Space */
		{
		case 0x09: /* U+0009 : CHARACTER TABULATION */
		case 0x0A: /* U+000A : LINE FEED */
		case 0x0B: /* U+000B : LINE TABULATION */
		case 0x0C: /* U+000C : FORM FEED */
		case 0x0D: /* U+000D : CARRIAGE RETURN */
		case 0x85: /* U+0085 : NEXT LINE */
			return 1;
		}
	default:
		return 0;
	}
}

int ucd_isupper(codepoint_t c)
{
	ucd_category cat = ucd_lookup_category(c);
	switch (cat)
	{
	case UCD_CATEGORY_Lu:
		return 1;
	case UCD_CATEGORY_Lt:
		return ucd_tolower(c) != c;
	case UCD_CATEGORY_Nl:
	case UCD_CATEGORY_So:
		return (ucd_properties(c, cat) & UCD_PROPERTY_OTHER_UPPERCASE) == UCD_PROPERTY_OTHER_UPPERCASE;
	default:
		return 0;
	}
}

int ucd_isxdigit(codepoint_t c)
{
	return (c >= 0x30 && c <= 0x39)  /* [0-9] */
	    || (c >= 0x41 && c <= 0x46)  /* [A-Z] */
	    || (c >= 0x61 && c <= 0x66); /* [a-z] */
}
