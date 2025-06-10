/*
 * Copyright (C) 2022 Ulrich Müller <ulm@gentoo.org>
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
 *
 *
 * Alternatively, at your option, you can distribute this file under
 * the terms of the 2-clause BSD license.
 */

#include <math.h>
#include <stdint.h>
#include "ieee80.h"

#ifndef INFINITY
# define INFINITY 0.
#endif

#ifndef NAN
# define NAN 0.
#endif

/*
 * Convert an IEEE 754 80-bit extended precision floating-point number
 * to a double. Input is expected as 10 bytes in big-endian order.
 *
 * Implemented according to the format described in:
 * https://en.wikipedia.org/wiki/Extended_precision
 * https://en.wikipedia.org/wiki/IEEE_754
 */
double
ieee_extended_to_double(const unsigned char *bytes)
{
	int sign, exp, i;
	uint64_t mant;
	double ret;

	sign = (bytes[0] & 0x80) != 0;
	exp = (bytes[0] & 0x7f) << 8 | bytes[1];

	/* Unfortunately, there is no 64-bit variant of ntohl(), and we
	   cannot use be64toh() either, because it is nonstandard */
	mant = 0;
	for (i = 2; i < 10; i++)
		mant = (mant << 8) | bytes[i];

	switch (exp) {
	case 0:			/* zero or denormalized number */
		ret = (mant == 0) ? 0. : ldexp(mant, - (16382 + 63));
		break;
	case 0x7fff:		/* infinity or not a number */
		/* Convert infinity to INFINITY, and anything else
		   (signalling NaN, quiet NaN, indefinite) to NAN */
		ret = ((mant & 0x7fffffffffffffff) == 0) ? INFINITY : NAN;
		break;
	default:
		ret = ldexp(mant, exp - (16383 + 63));
	}

	if (sign) ret = -ret;

	return ret;
}
