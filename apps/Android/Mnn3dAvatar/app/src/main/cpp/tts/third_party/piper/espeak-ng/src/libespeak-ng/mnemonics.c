/*
 * Copyright (C) 2005 to 2014 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2017 Reece H. Dunn
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

#include <string.h>

#include <espeak-ng/espeak_ng.h>

#include "mnemonics.h"  // for MNEM_TAB, LookupMnem, LookupMnemName

int LookupMnem(const MNEM_TAB *table, const char *string)
{
	while (table->mnem != NULL) {
		if (string && strcmp(string, table->mnem) == 0)
			return table->value;
		table++;
	}
	return table->value;
}

const char *LookupMnemName(const MNEM_TAB *table, const int value)
{
	while (table->mnem != NULL) {
		if (table->value == value)
			return table->mnem;
		table++;
	}
	return ""; // not found
}
