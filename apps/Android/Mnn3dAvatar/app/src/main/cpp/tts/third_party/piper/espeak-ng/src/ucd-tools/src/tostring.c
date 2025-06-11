/* Enumeration types to string.
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

const char *ucd_get_category_group_string(ucd_category_group c)
{
	switch (c)
	{
	case UCD_CATEGORY_GROUP_C: return "C";
	case UCD_CATEGORY_GROUP_I: return "I";
	case UCD_CATEGORY_GROUP_L: return "L";
	case UCD_CATEGORY_GROUP_M: return "M";
	case UCD_CATEGORY_GROUP_N: return "N";
	case UCD_CATEGORY_GROUP_P: return "P";
	case UCD_CATEGORY_GROUP_S: return "S";
	case UCD_CATEGORY_GROUP_Z: return "Z";
	default: return "-";
	}
}

const char *ucd_get_category_string(ucd_category c)
{
	switch (c)
	{
	case UCD_CATEGORY_Cc: return "Cc";
	case UCD_CATEGORY_Cf: return "Cf";
	case UCD_CATEGORY_Cn: return "Cn";
	case UCD_CATEGORY_Co: return "Co";
	case UCD_CATEGORY_Cs: return "Cs";
	case UCD_CATEGORY_Ii: return "Ii";
	case UCD_CATEGORY_Ll: return "Ll";
	case UCD_CATEGORY_Lm: return "Lm";
	case UCD_CATEGORY_Lo: return "Lo";
	case UCD_CATEGORY_Lt: return "Lt";
	case UCD_CATEGORY_Lu: return "Lu";
	case UCD_CATEGORY_Mc: return "Mc";
	case UCD_CATEGORY_Me: return "Me";
	case UCD_CATEGORY_Mn: return "Mn";
	case UCD_CATEGORY_Nd: return "Nd";
	case UCD_CATEGORY_Nl: return "Nl";
	case UCD_CATEGORY_No: return "No";
	case UCD_CATEGORY_Pc: return "Pc";
	case UCD_CATEGORY_Pd: return "Pd";
	case UCD_CATEGORY_Pe: return "Pe";
	case UCD_CATEGORY_Pf: return "Pf";
	case UCD_CATEGORY_Pi: return "Pi";
	case UCD_CATEGORY_Po: return "Po";
	case UCD_CATEGORY_Ps: return "Ps";
	case UCD_CATEGORY_Sc: return "Sc";
	case UCD_CATEGORY_Sk: return "Sk";
	case UCD_CATEGORY_Sm: return "Sm";
	case UCD_CATEGORY_So: return "So";
	case UCD_CATEGORY_Zl: return "Zl";
	case UCD_CATEGORY_Zp: return "Zp";
	case UCD_CATEGORY_Zs: return "Zs";
	default: return "--";
	}
}

const char *ucd_get_script_string(ucd_script s)
{
	static const char *scripts[] =
	{
		"Adlm",
		"Afak",
		"Aghb",
		"Ahom",
		"Arab",
		"Armi",
		"Armn",
		"Avst",
		"Bali",
		"Bamu",
		"Bass",
		"Batk",
		"Beng",
		"Bhks",
		"Blis",
		"Bopo",
		"Brah",
		"Brai",
		"Bugi",
		"Buhd",
		"Cakm",
		"Cans",
		"Cari",
		"Cham",
		"Cher",
		"Cirt",
		"Copt",
		"Cprt",
		"Cyrl",
		"Cyrs",
		"Deva",
		"Dogr",
		"Dsrt",
		"Dupl",
		"Egyd",
		"Egyh",
		"Egyp",
		"Elba",
		"Ethi",
		"Geok",
		"Geor",
		"Glag",
		"Gong",
		"Gonm",
		"Goth",
		"Gran",
		"Grek",
		"Gujr",
		"Guru",
		"Hang",
		"Hani",
		"Hano",
		"Hans",
		"Hant",
		"Hatr",
		"Hebr",
		"Hira",
		"Hluw",
		"Hmng",
		"Hrkt",
		"Hung",
		"Inds",
		"Ital",
		"Java",
		"Jpan",
		"Jurc",
		"Kali",
		"Kana",
		"Khar",
		"Khmr",
		"Khoj",
		"Knda",
		"Kore",
		"Kpel",
		"Kthi",
		"Lana",
		"Laoo",
		"Latf",
		"Latg",
		"Latn",
		"Lepc",
		"Limb",
		"Lina",
		"Linb",
		"Lisu",
		"Loma",
		"Lyci",
		"Lydi",
		"Mahj",
		"Maka",
		"Mand",
		"Mani",
		"Marc",
		"Maya",
		"Medf",
		"Mend",
		"Merc",
		"Mero",
		"Mlym",
		"Modi",
		"Mong",
		"Moon",
		"Mroo",
		"Mtei",
		"Mult",
		"Mymr",
		"Narb",
		"Nbat",
		"Newa",
		"Nkgb",
		"Nkoo",
		"Nshu",
		"Ogam",
		"Olck",
		"Orkh",
		"Orya",
		"Osge",
		"Osma",
		"Palm",
		"Pauc",
		"Perm",
		"Phag",
		"Phli",
		"Phlp",
		"Phlv",
		"Phnx",
		"Plrd",
		"Prti",
		"Qaak",
		"Rjng",
		"Rohg",
		"Roro",
		"Runr",
		"Samr",
		"Sara",
		"Sarb",
		"Saur",
		"Sgnw",
		"Shaw",
		"Shrd",
		"Sidd",
		"Sind",
		"Sinh",
		"Sogd",
		"Sogo",
		"Sora",
		"Soyo",
		"Sund",
		"Sylo",
		"Syrc",
		"Syre",
		"Syrj",
		"Syrn",
		"Tagb",
		"Takr",
		"Tale",
		"Talu",
		"Taml",
		"Tang",
		"Tavt",
		"Telu",
		"Teng",
		"Tfng",
		"Tglg",
		"Thaa",
		"Thai",
		"Tibt",
		"Tirh",
		"Ugar",
		"Vaii",
		"Visp",
		"Wara",
		"Wole",
		"Xpeo",
		"Xsux",
		"Yiii",
		"Zanb",
		"Zinh",
		"Zmth",
		"Zsym",
		"Zxxx",
		"Zyyy",
		"Zzzz",
	};

	if ((unsigned int)s >= (sizeof(scripts)/sizeof(scripts[0])))
		return "----";
	return scripts[(unsigned int)s];
}
