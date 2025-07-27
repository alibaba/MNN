/* Unicode Character Database API
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

#ifndef UNICODE_CHARACTER_DATA_H
#define UNICODE_CHARACTER_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Represents a Unicode codepoint.
  */
typedef uint32_t codepoint_t;

/** @brief Unicode General Category Groups
  * @see   http://www.unicode.org/reports/tr44/
  */
typedef enum ucd_category_group_
{
	UCD_CATEGORY_GROUP_C, /**< @brief Other */
	UCD_CATEGORY_GROUP_I, /**< @brief Invalid */
	UCD_CATEGORY_GROUP_L, /**< @brief Letter */
	UCD_CATEGORY_GROUP_M, /**< @brief Mark */
	UCD_CATEGORY_GROUP_N, /**< @brief Number */
	UCD_CATEGORY_GROUP_P, /**< @brief Punctuation */
	UCD_CATEGORY_GROUP_S, /**< @brief Symbol */
	UCD_CATEGORY_GROUP_Z, /**< @brief Separator */
} ucd_category_group;

/** @brief Get a string representation of the category_group enumeration value.
  *
  * @param c The value to get the string representation for.
  *
  * @return The string representation, or "-" if the value is not recognized.
  */
const char *ucd_get_category_group_string(ucd_category_group c);

/** @brief Unicode General Category Values
  * @see   http://www.unicode.org/reports/tr44/
  */
typedef enum ucd_category_
{
	UCD_CATEGORY_Cc, /**< @brief Control Character */
	UCD_CATEGORY_Cf, /**< @brief Format Control Character */
	UCD_CATEGORY_Cn, /**< @brief Unassigned */
	UCD_CATEGORY_Co, /**< @brief Private Use */
	UCD_CATEGORY_Cs, /**< @brief Surrogate Code Point */

	UCD_CATEGORY_Ii, /**< @brief Invalid Unicode Codepoint */

	UCD_CATEGORY_Ll, /**< @brief Lower Case Letter */
	UCD_CATEGORY_Lm, /**< @brief Letter Modifier */
	UCD_CATEGORY_Lo, /**< @brief Other Letter */
	UCD_CATEGORY_Lt, /**< @brief Title Case Letter */
	UCD_CATEGORY_Lu, /**< @brief Upper Case Letter */

	UCD_CATEGORY_Mc, /**< @brief Spacing Mark */
	UCD_CATEGORY_Me, /**< @brief Enclosing Mark */
	UCD_CATEGORY_Mn, /**< @brief Non-Spacing Mark */

	UCD_CATEGORY_Nd, /**< @brief Decimal Digit */
	UCD_CATEGORY_Nl, /**< @brief Letter-Like Number */
	UCD_CATEGORY_No, /**< @brief Other Number */

	UCD_CATEGORY_Pc, /**< @brief Connector */
	UCD_CATEGORY_Pd, /**< @brief Dash/Hyphen */
	UCD_CATEGORY_Pe, /**< @brief Close Punctuation Mark */
	UCD_CATEGORY_Pf, /**< @brief Final Quotation Mark */
	UCD_CATEGORY_Pi, /**< @brief Initial Quotation Mark */
	UCD_CATEGORY_Po, /**< @brief Other */
	UCD_CATEGORY_Ps, /**< @brief Open Punctuation Mark */

	UCD_CATEGORY_Sc, /**< @brief Currency Symbol */
	UCD_CATEGORY_Sk, /**< @brief Modifier Symbol */
	UCD_CATEGORY_Sm, /**< @brief Math Symbol */
	UCD_CATEGORY_So, /**< @brief Other Symbol */

	UCD_CATEGORY_Zl, /**< @brief Line Separator */
	UCD_CATEGORY_Zp, /**< @brief Paragraph Separator */
	UCD_CATEGORY_Zs, /**< @brief Space Separator */
} ucd_category;

/** @brief Get a string representation of the category enumeration value.
  *
  * @param c The value to get the string representation for.
  *
  * @return The string representation, or "--" if the value is not recognized.
  */
const char *ucd_get_category_string(ucd_category c);

/** @brief Lookup the General Category Group for a General Category.
  *
  * @param c The General Category to lookup.
  * @return  The General Category Group of the General Category.
  */
ucd_category_group ucd_get_category_group_for_category(ucd_category c);

/** @brief Lookup the General Category Group for a Unicode codepoint.
  *
  * @param c The Unicode codepoint to lookup.
  * @return  The General Category Group of the Unicode codepoint.
  */
ucd_category_group ucd_lookup_category_group(codepoint_t c);

/** @brief Lookup the General Category for a Unicode codepoint.
  *
  * @param c The Unicode codepoint to lookup.
  * @return  The General Category of the Unicode codepoint.
  */
ucd_category ucd_lookup_category(codepoint_t c);

/** @brief Unicode Script
  * @see   http://www.iana.org/assignments/language-subtag-registry
  * @see   http://www.unicode.org/iso15924/iso15924-codes.html
  */
typedef enum ucd_script_
{
	UCD_SCRIPT_Adlm, /**< @brief Adlam Script */
	UCD_SCRIPT_Afak, /**< @brief Afaka Script */
	UCD_SCRIPT_Aghb, /**< @brief Caucasian Albanian Script */
	UCD_SCRIPT_Ahom, /**< @brief Tai Ahom Script */
	UCD_SCRIPT_Arab, /**< @brief Arabic Script */
	UCD_SCRIPT_Armi, /**< @brief Imperial Aramaic Script */
	UCD_SCRIPT_Armn, /**< @brief Armenian Script */
	UCD_SCRIPT_Avst, /**< @brief Avestan Script */
	UCD_SCRIPT_Bali, /**< @brief Balinese Script */
	UCD_SCRIPT_Bamu, /**< @brief Bamum Script */
	UCD_SCRIPT_Bass, /**< @brief Bassa Vah Script */
	UCD_SCRIPT_Batk, /**< @brief Batak Script */
	UCD_SCRIPT_Beng, /**< @brief Bengali Script */
	UCD_SCRIPT_Bhks, /**< @brief Bhaiksuki Script */
	UCD_SCRIPT_Blis, /**< @brief Blissymbols Script */
	UCD_SCRIPT_Bopo, /**< @brief Bopomofo Script */
	UCD_SCRIPT_Brah, /**< @brief Brahmi Script */
	UCD_SCRIPT_Brai, /**< @brief Braille Script */
	UCD_SCRIPT_Bugi, /**< @brief Buginese Script */
	UCD_SCRIPT_Buhd, /**< @brief Buhid Script */
	UCD_SCRIPT_Cakm, /**< @brief Chakma Script */
	UCD_SCRIPT_Cans, /**< @brief Unified Canadian Aboriginal Syllabics */
	UCD_SCRIPT_Cari, /**< @brief Carian Script */
	UCD_SCRIPT_Cham, /**< @brief Cham Script */
	UCD_SCRIPT_Cher, /**< @brief Cherokee Script */
	UCD_SCRIPT_Cirt, /**< @brief Cirth Script */
	UCD_SCRIPT_Copt, /**< @brief Coptic Script */
	UCD_SCRIPT_Cprt, /**< @brief Cypriot Script */
	UCD_SCRIPT_Cyrl, /**< @brief Cyrillic Script */
	UCD_SCRIPT_Cyrs, /**< @brief Cyrillic (Old Church Slavonic variant) Script */
	UCD_SCRIPT_Deva, /**< @brief Devanagari Script */
	UCD_SCRIPT_Dogr, /**< @brief Dogra Script */
	UCD_SCRIPT_Dsrt, /**< @brief Deseret Script */
	UCD_SCRIPT_Dupl, /**< @brief Duployan Shorthand Script */
	UCD_SCRIPT_Egyd, /**< @brief Egyptian Demotic Script */
	UCD_SCRIPT_Egyh, /**< @brief Egyptian Hieratic Script */
	UCD_SCRIPT_Egyp, /**< @brief Egyptian Hiegoglyphs */
	UCD_SCRIPT_Elba, /**< @brief Elbasan Script */
	UCD_SCRIPT_Ethi, /**< @brief Ethiopic Script */
	UCD_SCRIPT_Geok, /**< @brief Khutsuri Script */
	UCD_SCRIPT_Geor, /**< @brief Geirgian Script */
	UCD_SCRIPT_Glag, /**< @brief Glagolitic Script */
	UCD_SCRIPT_Gong, /**< @brief Gunjala Gondi */
	UCD_SCRIPT_Gonm, /**< @brief Masaram Gondi */
	UCD_SCRIPT_Goth, /**< @brief Gothic Script */
	UCD_SCRIPT_Gran, /**< @brief Grantha Script */
	UCD_SCRIPT_Grek, /**< @brief Greek Script */
	UCD_SCRIPT_Gujr, /**< @brief Gujarati Script */
	UCD_SCRIPT_Guru, /**< @brief Gurmukhi Script */
	UCD_SCRIPT_Hang, /**< @brief Hangul Script */
	UCD_SCRIPT_Hani, /**< @brief Han (Hanzi, Kanji, Hanja) Script */
	UCD_SCRIPT_Hano, /**< @brief Hanunoo Script */
	UCD_SCRIPT_Hans, /**< @brief Han (Simplified) Script */
	UCD_SCRIPT_Hant, /**< @brief Han (Traditional) Script */
	UCD_SCRIPT_Hatr, /**< @brief Hatran Script */
	UCD_SCRIPT_Hebr, /**< @brief Hebrew Script */
	UCD_SCRIPT_Hira, /**< @brief Hiragana Script */
	UCD_SCRIPT_Hluw, /**< @brief Anatolian Hieroglyphs */
	UCD_SCRIPT_Hmng, /**< @brief Pahawh Hmong Script */
	UCD_SCRIPT_Hrkt, /**< @brief Japanese Syllabaries */
	UCD_SCRIPT_Hung, /**< @brief Old Hungarian Script */
	UCD_SCRIPT_Inds, /**< @brief Indus Script */
	UCD_SCRIPT_Ital, /**< @brief Old Italic Script */
	UCD_SCRIPT_Java, /**< @brief Javanese Script */
	UCD_SCRIPT_Jpan, /**< @brief Japanese (Han + Hiragana + Katakana) Scripts */
	UCD_SCRIPT_Jurc, /**< @brief Jurchen Script */
	UCD_SCRIPT_Kali, /**< @brief Kayah Li Script */
	UCD_SCRIPT_Kana, /**< @brief Katakana Script */
	UCD_SCRIPT_Khar, /**< @brief Kharoshthi Script */
	UCD_SCRIPT_Khmr, /**< @brief Khmer Script */
	UCD_SCRIPT_Khoj, /**< @brief Khojki Script */
	UCD_SCRIPT_Knda, /**< @brief Kannada Script */
	UCD_SCRIPT_Kore, /**< @brief Korean (Hangul + Han) Scripts */
	UCD_SCRIPT_Kpel, /**< @brief Kpelle Script */
	UCD_SCRIPT_Kthi, /**< @brief Kaithi Script */
	UCD_SCRIPT_Lana, /**< @brief Tai Tham Script */
	UCD_SCRIPT_Laoo, /**< @brief Lao Script */
	UCD_SCRIPT_Latf, /**< @brief Latin Script (Fractur Variant) */
	UCD_SCRIPT_Latg, /**< @brief Latin Script (Gaelic Variant) */
	UCD_SCRIPT_Latn, /**< @brief Latin Script */
	UCD_SCRIPT_Lepc, /**< @brief Lepcha Script */
	UCD_SCRIPT_Limb, /**< @brief Limbu Script */
	UCD_SCRIPT_Lina, /**< @brief Linear A Script */
	UCD_SCRIPT_Linb, /**< @brief Linear B Script */
	UCD_SCRIPT_Lisu, /**< @brief Lisu Script */
	UCD_SCRIPT_Loma, /**< @brief Loma Script */
	UCD_SCRIPT_Lyci, /**< @brief Lycian Script */
	UCD_SCRIPT_Lydi, /**< @brief Lydian Script */
	UCD_SCRIPT_Mahj, /**< @brief Mahajani Script */
	UCD_SCRIPT_Maka, /**< @brief Makasar Script */
	UCD_SCRIPT_Mand, /**< @brief Mandaic Script */
	UCD_SCRIPT_Mani, /**< @brief Manichaean Script */
	UCD_SCRIPT_Marc, /**< @brief Marchen Script */
	UCD_SCRIPT_Maya, /**< @brief Mayan Hieroglyphs */
	UCD_SCRIPT_Medf, /**< @brief Medefaidrin (Oberi Okaime) Script */
	UCD_SCRIPT_Mend, /**< @brief Mende Kikakui Script */
	UCD_SCRIPT_Merc, /**< @brief Meroitic Cursive Script */
	UCD_SCRIPT_Mero, /**< @brief Meroitic Hieroglyphs */
	UCD_SCRIPT_Mlym, /**< @brief Malayalam Script */
	UCD_SCRIPT_Modi, /**< @brief Modi Script */
	UCD_SCRIPT_Mong, /**< @brief Mongolian Script */
	UCD_SCRIPT_Moon, /**< @brief Moon Script */
	UCD_SCRIPT_Mroo, /**< @brief Mro Script */
	UCD_SCRIPT_Mtei, /**< @brief Meitei Mayek Script */
	UCD_SCRIPT_Mult, /**< @brief Multani Script */
	UCD_SCRIPT_Mymr, /**< @brief Myanmar (Burmese) Script */
	UCD_SCRIPT_Narb, /**< @brief Old North Arabian Script */
	UCD_SCRIPT_Nbat, /**< @brief Nabataean Script */
	UCD_SCRIPT_Newa, /**< @brief Newa Script */
	UCD_SCRIPT_Nkgb, /**< @brief Nakhi Geba Script */
	UCD_SCRIPT_Nkoo, /**< @brief N'Ko Script */
	UCD_SCRIPT_Nshu, /**< @brief Nushu Script */
	UCD_SCRIPT_Ogam, /**< @brief Ogham Script */
	UCD_SCRIPT_Olck, /**< @brief Ol Chiki Script */
	UCD_SCRIPT_Orkh, /**< @brief Old Turkic Script */
	UCD_SCRIPT_Orya, /**< @brief Oriya Script */
	UCD_SCRIPT_Osge, /**< @brief Osage Script */
	UCD_SCRIPT_Osma, /**< @brief Osmanya Script */
	UCD_SCRIPT_Palm, /**< @brief Palmyrene Script */
	UCD_SCRIPT_Pauc, /**< @brief Pau Cin Hau Script */
	UCD_SCRIPT_Perm, /**< @brief Old Permic */
	UCD_SCRIPT_Phag, /**< @brief Phags-Pa Script */
	UCD_SCRIPT_Phli, /**< @brief Inscriptional Pahlavi Script */
	UCD_SCRIPT_Phlp, /**< @brief Psalter Pahlavi Script */
	UCD_SCRIPT_Phlv, /**< @brief Book Pahlavi Script */
	UCD_SCRIPT_Phnx, /**< @brief Phoenician Script */
	UCD_SCRIPT_Plrd, /**< @brief Miao Script */
	UCD_SCRIPT_Prti, /**< @brief Inscriptional Parthian Script */
	UCD_SCRIPT_Qaak, /**< @brief Klingon Script (Private Use) */
	UCD_SCRIPT_Rjng, /**< @brief Rejang Script */
	UCD_SCRIPT_Rohg, /**< @brief Hanifi Rohingya Script */
	UCD_SCRIPT_Roro, /**< @brief Rongorongo Script */
	UCD_SCRIPT_Runr, /**< @brief Runic Script */
	UCD_SCRIPT_Samr, /**< @brief Samaritan Script */
	UCD_SCRIPT_Sara, /**< @brief Sarati Script */
	UCD_SCRIPT_Sarb, /**< @brief Old South Arabian Script */
	UCD_SCRIPT_Saur, /**< @brief Saurashtra Script */
	UCD_SCRIPT_Sgnw, /**< @brief Sign Writing */
	UCD_SCRIPT_Shaw, /**< @brief Shavian Script */
	UCD_SCRIPT_Shrd, /**< @brief Sharada Script */
	UCD_SCRIPT_Sidd, /**< @brief Siddham Script */
	UCD_SCRIPT_Sind, /**< @brief Sindhi Script */
	UCD_SCRIPT_Sinh, /**< @brief Sinhala Script */
	UCD_SCRIPT_Sogd, /**< @brief Sogdian Script */
	UCD_SCRIPT_Sogo, /**< @brief Old Sogdian Script */
	UCD_SCRIPT_Sora, /**< @brief Sora Sompeng Script */
	UCD_SCRIPT_Soyo, /**< @brief Soyombo */
	UCD_SCRIPT_Sund, /**< @brief Sundanese Script */
	UCD_SCRIPT_Sylo, /**< @brief Syloti Nagri Script */
	UCD_SCRIPT_Syrc, /**< @brief Syriac Script */
	UCD_SCRIPT_Syre, /**< @brief Syriac Script (Estrangelo Variant) */
	UCD_SCRIPT_Syrj, /**< @brief Syriac Script (Western Variant) */
	UCD_SCRIPT_Syrn, /**< @brief Syriac Script (Eastern Variant) */
	UCD_SCRIPT_Tagb, /**< @brief Tagbanwa Script */
	UCD_SCRIPT_Takr, /**< @brief Takri Script */
	UCD_SCRIPT_Tale, /**< @brief Tai Le Script */
	UCD_SCRIPT_Talu, /**< @brief New Tai Lue Script */
	UCD_SCRIPT_Taml, /**< @brief Tamil Script */
	UCD_SCRIPT_Tang, /**< @brief Tangut Script */
	UCD_SCRIPT_Tavt, /**< @brief Tai Viet Script */
	UCD_SCRIPT_Telu, /**< @brief Telugu Script */
	UCD_SCRIPT_Teng, /**< @brief Tengwar Script */
	UCD_SCRIPT_Tfng, /**< @brief Tifinagh Script */
	UCD_SCRIPT_Tglg, /**< @brief Tagalog Script */
	UCD_SCRIPT_Thaa, /**< @brief Thaana Script */
	UCD_SCRIPT_Thai, /**< @brief Thai Script */
	UCD_SCRIPT_Tibt, /**< @brief Tibetan Script */
	UCD_SCRIPT_Tirh, /**< @brief Tirhuta Script */
	UCD_SCRIPT_Ugar, /**< @brief Ugaritic Script */
	UCD_SCRIPT_Vaii, /**< @brief Vai Script */
	UCD_SCRIPT_Visp, /**< @brief Visible Speech Script */
	UCD_SCRIPT_Wara, /**< @brief Warang Citi Script */
	UCD_SCRIPT_Wole, /**< @brief Woleai Script */
	UCD_SCRIPT_Xpeo, /**< @brief Old Persian Script */
	UCD_SCRIPT_Xsux, /**< @brief Cuneiform Script */
	UCD_SCRIPT_Yiii, /**< @brief Yi Script */
	UCD_SCRIPT_Zanb, /**< @brief Zanabazar Square */
	UCD_SCRIPT_Zinh, /**< @brief Inherited Script */
	UCD_SCRIPT_Zmth, /**< @brief Mathematical Notation */
	UCD_SCRIPT_Zsym, /**< @brief Symbols */
	UCD_SCRIPT_Zxxx, /**< @brief Unwritten Documents */
	UCD_SCRIPT_Zyyy, /**< @brief Undetermined Script */
	UCD_SCRIPT_Zzzz, /**< @brief Uncoded Script */
} ucd_script;

/** @brief Get a string representation of the script enumeration value.
  *
  * @param s The value to get the string representation for.
  *
  * @return The string representation, or "----" if the value is not recognized.
  */
const char *ucd_get_script_string(ucd_script s);

/** @brief Lookup the Script for a Unicode codepoint.
  *
  * @param c The Unicode codepoint to lookup.
  * @return  The Script of the Unicode codepoint.
  */
ucd_script ucd_lookup_script(codepoint_t c);

/** @brief Properties
 */
typedef uint64_t ucd_property;

#define UCD_PROPERTY_WHITE_SPACE                        0x0000000000000001ull /**< @brief White_Space */
#define UCD_PROPERTY_BIDI_CONTROL                       0x0000000000000002ull /**< @brief Bidi_Control */
#define UCD_PROPERTY_JOIN_CONTROL                       0x0000000000000004ull /**< @brief Join_Control */
#define UCD_PROPERTY_DASH                               0x0000000000000008ull /**< @brief Dash */
#define UCD_PROPERTY_HYPHEN                             0x0000000000000010ull /**< @brief Hyphen */
#define UCD_PROPERTY_QUOTATION_MARK                     0x0000000000000020ull /**< @brief Quotation_Mark */
#define UCD_PROPERTY_TERMINAL_PUNCTUATION               0x0000000000000040ull /**< @brief Terminal_Punctuation */
#define UCD_PROPERTY_OTHER_MATH                         0x0000000000000080ull /**< @brief Other_Math */
#define UCD_PROPERTY_HEX_DIGIT                          0x0000000000000100ull /**< @brief Hex_Digit */
#define UCD_PROPERTY_ASCII_HEX_DIGIT                    0x0000000000000200ull /**< @brief ASCII_Hex_Digit */
#define UCD_PROPERTY_OTHER_ALPHABETIC                   0x0000000000000400ull /**< @brief Other_Alphabetic */
#define UCD_PROPERTY_IDEOGRAPHIC                        0x0000000000000800ull /**< @brief Ideographic */
#define UCD_PROPERTY_DIACRITIC                          0x0000000000001000ull /**< @brief Diacritic */
#define UCD_PROPERTY_EXTENDER                           0x0000000000002000ull /**< @brief Extender */
#define UCD_PROPERTY_OTHER_LOWERCASE                    0x0000000000004000ull /**< @brief Other_Lowercase */
#define UCD_PROPERTY_OTHER_UPPERCASE                    0x0000000000008000ull /**< @brief Other_Uppercase */
#define UCD_PROPERTY_NONCHARACTER_CODE_POINT            0x0000000000010000ull /**< @brief Noncharacter_Code_Point */
#define UCD_PROPERTY_OTHER_GRAPHEME_EXTEND              0x0000000000020000ull /**< @brief Other_Grapheme_Extend */
#define UCD_PROPERTY_IDS_BINARY_OPERATOR                0x0000000000040000ull /**< @brief IDS_Binary_Operator */
#define UCD_PROPERTY_IDS_TRINARY_OPERATOR               0x0000000000080000ull /**< @brief IDS_Trinary_Operator */
#define UCD_PROPERTY_RADICAL                            0x0000000000100000ull /**< @brief Radical */
#define UCD_PROPERTY_UNIFIED_IDEOGRAPH                  0x0000000000200000ull /**< @brief Unified_Ideograph */
#define UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT 0x0000000000400000ull /**< @brief Other_Default_Ignorable_Code_Point */
#define UCD_PROPERTY_DEPRECATED                         0x0000000000800000ull /**< @brief Deprecated */
#define UCD_PROPERTY_SOFT_DOTTED                        0x0000000001000000ull /**< @brief Soft_Dotted */
#define UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION            0x0000000002000000ull /**< @brief Logical_Order_Exception */
#define UCD_PROPERTY_OTHER_ID_START                     0x0000000004000000ull /**< @brief Other_ID_Start */
#define UCD_PROPERTY_OTHER_ID_CONTINUE                  0x0000000008000000ull /**< @brief Other_ID_Continue */
#define UCD_PROPERTY_SENTENCE_TERMINAL                  0x0000000010000000ull /**< @brief Sentence_Terminal */
#define UCD_PROPERTY_VARIATION_SELECTOR                 0x0000000020000000ull /**< @brief Variation_Selector */
#define UCD_PROPERTY_PATTERN_WHITE_SPACE                0x0000000040000000ull /**< @brief Pattern_White_Space */
#define UCD_PROPERTY_PATTERN_SYNTAX                     0x0000000080000000ull /**< @brief Pattern_Syntax */
#define UCD_PROPERTY_PREPENDED_CONCATENATION_MARK       0x0000000100000000ull /**< @brief Prepended_Concatenation_Mark */
#define UCD_PROPERTY_EMOJI                              0x0000000200000000ull /**< @brief Emoji */
#define UCD_PROPERTY_EMOJI_PRESENTATION                 0x0000000400000000ull /**< @brief Emoji_Presentation */
#define UCD_PROPERTY_EMOJI_MODIFIER                     0x0000000800000000ull /**< @brief Emoji_Modifier */
#define UCD_PROPERTY_EMOJI_MODIFIER_BASE                0x0000001000000000ull /**< @brief Emoji_Modifier_Base */
#define UCD_PROPERTY_REGIONAL_INDICATOR                 0x0000002000000000ull /**< @brief Regional_Indicator */
#define UCD_PROPERTY_EMOJI_COMPONENT                    0x0000004000000000ull /**< @brief Emoji_Component */
#define UCD_PROPERTY_EXTENDED_PICTOGRAPHIC              0x0000008000000000ull /**< @brief Extended_Pictographic */

// eSpeak NG extended properties:
#define ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION 0x0010000000000000ull /**< @brief Inverted_Terminal_Punctuation */
#define ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD           0x0020000000000000ull /**< @brief Punctuation_In_Word */
#define ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER          0x0040000000000000ull /**< @brief Optional_Space_After */
#define ESPEAKNG_PROPERTY_EXTENDED_DASH                 0x0080000000000000ull /**< @brief Extended_Dash */
#define ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR           0x0100000000000000ull /**< @brief Paragraph_Separator */
#define ESPEAKNG_PROPERTY_ELLIPSIS                      0x0200000000000000ull /**< @brief Ellipsis */
#define ESPEAKNG_PROPERTY_SEMI_COLON                    0x0400000000000000ull /**< @brief Semi_Colon */
#define ESPEAKNG_PROPERTY_COLON                         0x0800000000000000ull /**< @brief Colon */
#define ESPEAKNG_PROPERTY_COMMA                         0x1000000000000000ull /**< @brief Comma */
#define ESPEAKNG_PROPERTY_EXCLAMATION_MARK              0x2000000000000000ull /**< @brief Exclamation_Mark */
#define ESPEAKNG_PROPERTY_QUESTION_MARK                 0x4000000000000000ull /**< @brief Question_Mark */
#define ESPEAKNG_PROPERTY_FULL_STOP                     0x8000000000000000ull /**< @brief Full_Stop */

/** @brief Return the properties of the specified codepoint.
 *
 * @param c        The Unicode codepoint to lookup.
 * @param category The General Category of the codepoint.
 * @return         The properties associated with the codepoint.
 */
ucd_property ucd_properties(codepoint_t c, ucd_category category);

/** @brief Is the codepoint in the 'alnum' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'alnum' class, zero otherwise.
  */
int ucd_isalnum(codepoint_t c);

/** @brief Is the codepoint in the 'alpha' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'alpha' class, zero otherwise.
  */
int ucd_isalpha(codepoint_t c);

/** @brief Is the codepoint in the 'blank' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'blank' class, zero otherwise.
  */
int ucd_isblank(codepoint_t c);

/** @brief Is the codepoint in the 'cntrl' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'cntrl' class, zero otherwise.
  */
int ucd_iscntrl(codepoint_t c);

/** @brief Is the codepoint in the 'digit' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'digit' class, zero otherwise.
  */
int ucd_isdigit(codepoint_t c);

/** @brief Is the codepoint in the 'graph' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'graph' class, zero otherwise.
  */
int ucd_isgraph(codepoint_t c);

/** @brief Is the codepoint in the 'lower' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'lower' class, zero otherwise.
  */
int ucd_islower(codepoint_t c);

/** @brief Is the codepoint in the 'print' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'print' class, zero otherwise.
  */
int ucd_isprint(codepoint_t c);

/** @brief Is the codepoint in the 'punct' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'punct' class, zero otherwise.
  */
int ucd_ispunct(codepoint_t c);

/** @brief Is the codepoint in the 'space' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'space' class, zero otherwise.
  */
int ucd_isspace(codepoint_t c);

/** @brief Is the codepoint in the 'upper' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'upper' class, zero otherwise.
  */
int ucd_isupper(codepoint_t c);

/** @brief Is the codepoint in the 'xdigit' class?
  *
  * @param c The Unicode codepoint to check.
  * @return  Non-zero if the codepoint is in the 'xdigit' class, zero otherwise.
  */
int ucd_isxdigit(codepoint_t c);

/** @brief Convert the Unicode codepoint to upper-case.
  *
  * This function only uses the simple case mapping present in the
  * UnicodeData file. The data in SpecialCasing requires Unicode
  * codepoints to be mapped to multiple codepoints.
  *
  * @param c The Unicode codepoint to convert.
  * @return  The upper-case Unicode codepoint for this codepoint, or
  *          this codepoint if there is no upper-case codepoint.
  */
codepoint_t ucd_toupper(codepoint_t c);

/** @brief Convert the Unicode codepoint to lower-case.
  *
  * This function only uses the simple case mapping present in the
  * UnicodeData file. The data in SpecialCasing requires Unicode
  * codepoints to be mapped to multiple codepoints.
  *
  * @param c The Unicode codepoint to convert.
  * @return  The lower-case Unicode codepoint for this codepoint, or
  *          this codepoint if there is no upper-case codepoint.
  */
codepoint_t ucd_tolower(codepoint_t c);

/** @brief Convert the Unicode codepoint to title-case.
  *
  * This function only uses the simple case mapping present in the
  * UnicodeData file. The data in SpecialCasing requires Unicode
  * codepoints to be mapped to multiple codepoints.
  *
  * @param c The Unicode codepoint to convert.
  * @return  The title-case Unicode codepoint for this codepoint, or
  *          this codepoint if there is no upper-case codepoint.
  */
codepoint_t ucd_totitle(codepoint_t c);

#ifdef __cplusplus
}

/** @brief Unicode Character Database
  */
namespace ucd
{
	/** @brief Represents a Unicode codepoint.
	  */
	using ::codepoint_t;

	/** @brief Unicode General Category Groups
	  * @see   http://www.unicode.org/reports/tr44/
	  */
	enum category_group
	{
		C = UCD_CATEGORY_GROUP_C, /**< @brief Other */
		I = UCD_CATEGORY_GROUP_I, /**< @brief Invalid */
		L = UCD_CATEGORY_GROUP_L, /**< @brief Letter */
		M = UCD_CATEGORY_GROUP_M, /**< @brief Mark */
		N = UCD_CATEGORY_GROUP_N, /**< @brief Number */
		P = UCD_CATEGORY_GROUP_P, /**< @brief Punctuation */
		S = UCD_CATEGORY_GROUP_S, /**< @brief Symbol */
		Z = UCD_CATEGORY_GROUP_Z, /**< @brief Separator */
	};

	/** @brief Get a string representation of the category_group enumeration value.
	  *
	  * @param c The value to get the string representation for.
	  *
	  * @return The string representation, or "-" if the value is not recognized.
	  */
	inline const char *get_category_group_string(category_group c)
	{
		return ucd_get_category_group_string((ucd_category_group)c);
	}

	/** @brief Unicode General Category Values
	  * @see   http://www.unicode.org/reports/tr44/
	  */
	enum category
	{
		Cc = UCD_CATEGORY_Cc, /**< @brief Control Character */
		Cf = UCD_CATEGORY_Cf, /**< @brief Format Control Character */
		Cn = UCD_CATEGORY_Cn, /**< @brief Unassigned */
		Co = UCD_CATEGORY_Co, /**< @brief Private Use */
		Cs = UCD_CATEGORY_Cs, /**< @brief Surrogate Code Point */

		Ii = UCD_CATEGORY_Ii, /**< @brief Invalid Unicode Codepoint */

		Ll = UCD_CATEGORY_Ll, /**< @brief Lower Case Letter */
		Lm = UCD_CATEGORY_Lm, /**< @brief Letter Modifier */
		Lo = UCD_CATEGORY_Lo, /**< @brief Other Letter */
		Lt = UCD_CATEGORY_Lt, /**< @brief Title Case Letter */
		Lu = UCD_CATEGORY_Lu, /**< @brief Upper Case Letter */

		Mc = UCD_CATEGORY_Mc, /**< @brief Spacing Mark */
		Me = UCD_CATEGORY_Me, /**< @brief Enclosing Mark */
		Mn = UCD_CATEGORY_Mn, /**< @brief Non-Spacing Mark */

		Nd = UCD_CATEGORY_Nd, /**< @brief Decimal Digit */
		Nl = UCD_CATEGORY_Nl, /**< @brief Letter-Like Number */
		No = UCD_CATEGORY_No, /**< @brief Other Number */

		Pc = UCD_CATEGORY_Pc, /**< @brief Connector */
		Pd = UCD_CATEGORY_Pd, /**< @brief Dash/Hyphen */
		Pe = UCD_CATEGORY_Pe, /**< @brief Close Punctuation Mark */
		Pf = UCD_CATEGORY_Pf, /**< @brief Final Quotation Mark */
		Pi = UCD_CATEGORY_Pi, /**< @brief Initial Quotation Mark */
		Po = UCD_CATEGORY_Po, /**< @brief Other */
		Ps = UCD_CATEGORY_Ps, /**< @brief Open Punctuation Mark */

		Sc = UCD_CATEGORY_Sc, /**< @brief Currency Symbol */
		Sk = UCD_CATEGORY_Sk, /**< @brief Modifier Symbol */
		Sm = UCD_CATEGORY_Sm, /**< @brief Math Symbol */
		So = UCD_CATEGORY_So, /**< @brief Other Symbol */

		Zl = UCD_CATEGORY_Zl, /**< @brief Line Separator */
		Zp = UCD_CATEGORY_Zp, /**< @brief Paragraph Separator */
		Zs = UCD_CATEGORY_Zs, /**< @brief Space Separator */
	};

	/** @brief Get a string representation of the category enumeration value.
	  *
	  * @param c The value to get the string representation for.
	  *
	  * @return The string representation, or "--" if the value is not recognized.
	  */
	inline const char *get_category_string(category c)
	{
		return ucd_get_category_string((ucd_category)c);
	}

	/** @brief Lookup the General Category Group for a General Category.
	  *
	  * @param c The General Category to lookup.
	  * @return  The General Category Group of the General Category.
	  */
	inline category_group lookup_category_group(category c)
	{
		return (category_group)ucd_get_category_group_for_category((ucd_category)c);
	}

	/** @brief Lookup the General Category Group for a Unicode codepoint.
	  *
	  * @param c The Unicode codepoint to lookup.
	  * @return  The General Category Group of the Unicode codepoint.
	  */
	inline category_group lookup_category_group(codepoint_t c)
	{
		return (category_group)ucd_lookup_category_group(c);
	}

	/** @brief Lookup the General Category for a Unicode codepoint.
	  *
	  * @param c The Unicode codepoint to lookup.
	  * @return  The General Category of the Unicode codepoint.
	  */
	inline category lookup_category(codepoint_t c)
	{
		return (category)ucd_lookup_category(c);
	}

	/** @brief Unicode Script
	  * @see   http://www.iana.org/assignments/language-subtag-registry
	  * @see   http://www.unicode.org/iso15924/iso15924-codes.html
	  */
	enum script
	{
		Adlm = UCD_SCRIPT_Adlm, /**< @brief Adlam Script */
		Afak = UCD_SCRIPT_Afak, /**< @brief Afaka Script */
		Aghb = UCD_SCRIPT_Aghb, /**< @brief Caucasian Albanian Script */
		Ahom = UCD_SCRIPT_Ahom, /**< @brief Tai Ahom Script */
		Arab = UCD_SCRIPT_Arab, /**< @brief Arabic Script */
		Armi = UCD_SCRIPT_Armi, /**< @brief Imperial Aramaic Script */
		Armn = UCD_SCRIPT_Armn, /**< @brief Armenian Script */
		Avst = UCD_SCRIPT_Avst, /**< @brief Avestan Script */
		Bali = UCD_SCRIPT_Bali, /**< @brief Balinese Script */
		Bamu = UCD_SCRIPT_Bamu, /**< @brief Bamum Script */
		Bass = UCD_SCRIPT_Bass, /**< @brief Bassa Vah Script */
		Batk = UCD_SCRIPT_Batk, /**< @brief Batak Script */
		Beng = UCD_SCRIPT_Beng, /**< @brief Bengali Script */
		Bhks = UCD_SCRIPT_Bhks, /**< @brief Bhaiksuki Script */
		Blis = UCD_SCRIPT_Blis, /**< @brief Blissymbols Script */
		Bopo = UCD_SCRIPT_Bopo, /**< @brief Bopomofo Script */
		Brah = UCD_SCRIPT_Brah, /**< @brief Brahmi Script */
		Brai = UCD_SCRIPT_Brai, /**< @brief Braille Script */
		Bugi = UCD_SCRIPT_Bugi, /**< @brief Buginese Script */
		Buhd = UCD_SCRIPT_Buhd, /**< @brief Buhid Script */
		Cakm = UCD_SCRIPT_Cakm, /**< @brief Chakma Script */
		Cans = UCD_SCRIPT_Cans, /**< @brief Unified Canadian Aboriginal Syllabics */
		Cari = UCD_SCRIPT_Cari, /**< @brief Carian Script */
		Cham = UCD_SCRIPT_Cham, /**< @brief Cham Script */
		Cher = UCD_SCRIPT_Cher, /**< @brief Cherokee Script */
		Cirt = UCD_SCRIPT_Cirt, /**< @brief Cirth Script */
		Copt = UCD_SCRIPT_Copt, /**< @brief Coptic Script */
		Cprt = UCD_SCRIPT_Cprt, /**< @brief Cypriot Script */
		Cyrl = UCD_SCRIPT_Cyrl, /**< @brief Cyrillic Script */
		Cyrs = UCD_SCRIPT_Cyrs, /**< @brief Cyrillic (Old Church Slavonic variant) Script */
		Deva = UCD_SCRIPT_Deva, /**< @brief Devanagari Script */
		Dogr = UCD_SCRIPT_Dogr, /**< @brief Dogra Script */
		Dsrt = UCD_SCRIPT_Dsrt, /**< @brief Deseret Script */
		Dupl = UCD_SCRIPT_Dupl, /**< @brief Duployan Shorthand Script */
		Egyd = UCD_SCRIPT_Egyd, /**< @brief Egyptian Demotic Script */
		Egyh = UCD_SCRIPT_Egyh, /**< @brief Egyptian Hieratic Script */
		Egyp = UCD_SCRIPT_Egyp, /**< @brief Egyptian Hiegoglyphs */
		Elba = UCD_SCRIPT_Elba, /**< @brief Elbasan Script */
		Ethi = UCD_SCRIPT_Ethi, /**< @brief Ethiopic Script */
		Geok = UCD_SCRIPT_Geok, /**< @brief Khutsuri Script */
		Geor = UCD_SCRIPT_Geor, /**< @brief Geirgian Script */
		Glag = UCD_SCRIPT_Glag, /**< @brief Glagolitic Script */
		Gong = UCD_SCRIPT_Gong, /**< @brief Gunjala Gondi */
		Gonm = UCD_SCRIPT_Gonm, /**< @brief Masaram Gondi */
		Goth = UCD_SCRIPT_Goth, /**< @brief Gothic Script */
		Gran = UCD_SCRIPT_Gran, /**< @brief Grantha Script */
		Grek = UCD_SCRIPT_Grek, /**< @brief Greek Script */
		Gujr = UCD_SCRIPT_Gujr, /**< @brief Gujarati Script */
		Guru = UCD_SCRIPT_Guru, /**< @brief Gurmukhi Script */
		Hang = UCD_SCRIPT_Hang, /**< @brief Hangul Script */
		Hani = UCD_SCRIPT_Hani, /**< @brief Han (Hanzi, Kanji, Hanja) Script */
		Hano = UCD_SCRIPT_Hano, /**< @brief Hanunoo Script */
		Hans = UCD_SCRIPT_Hans, /**< @brief Han (Simplified) Script */
		Hant = UCD_SCRIPT_Hant, /**< @brief Han (Traditional) Script */
		Hatr = UCD_SCRIPT_Hatr, /**< @brief Hatran Script */
		Hebr = UCD_SCRIPT_Hebr, /**< @brief Hebrew Script */
		Hira = UCD_SCRIPT_Hira, /**< @brief Hiragana Script */
		Hluw = UCD_SCRIPT_Hluw, /**< @brief Anatolian Hieroglyphs */
		Hmng = UCD_SCRIPT_Hmng, /**< @brief Pahawh Hmong Script */
		Hrkt = UCD_SCRIPT_Hrkt, /**< @brief Japanese Syllabaries */
		Hung = UCD_SCRIPT_Hung, /**< @brief Old Hungarian Script */
		Inds = UCD_SCRIPT_Inds, /**< @brief Indus Script */
		Ital = UCD_SCRIPT_Ital, /**< @brief Old Italic Script */
		Java = UCD_SCRIPT_Java, /**< @brief Javanese Script */
		Jpan = UCD_SCRIPT_Jpan, /**< @brief Japanese (Han + Hiragana + Katakana) Scripts */
		Jurc = UCD_SCRIPT_Jurc, /**< @brief Jurchen Script */
		Kali = UCD_SCRIPT_Kali, /**< @brief Kayah Li Script */
		Kana = UCD_SCRIPT_Kana, /**< @brief Katakana Script */
		Khar = UCD_SCRIPT_Khar, /**< @brief Kharoshthi Script */
		Khmr = UCD_SCRIPT_Khmr, /**< @brief Khmer Script */
		Khoj = UCD_SCRIPT_Khoj, /**< @brief Khojki Script */
		Knda = UCD_SCRIPT_Knda, /**< @brief Kannada Script */
		Kore = UCD_SCRIPT_Kore, /**< @brief Korean (Hangul + Han) Scripts */
		Kpel = UCD_SCRIPT_Kpel, /**< @brief Kpelle Script */
		Kthi = UCD_SCRIPT_Kthi, /**< @brief Kaithi Script */
		Lana = UCD_SCRIPT_Lana, /**< @brief Tai Tham Script */
		Laoo = UCD_SCRIPT_Laoo, /**< @brief Lao Script */
		Latf = UCD_SCRIPT_Latf, /**< @brief Latin Script (Fractur Variant) */
		Latg = UCD_SCRIPT_Latg, /**< @brief Latin Script (Gaelic Variant) */
		Latn = UCD_SCRIPT_Latn, /**< @brief Latin Script */
		Lepc = UCD_SCRIPT_Lepc, /**< @brief Lepcha Script */
		Limb = UCD_SCRIPT_Limb, /**< @brief Limbu Script */
		Lina = UCD_SCRIPT_Lina, /**< @brief Linear A Script */
		Linb = UCD_SCRIPT_Linb, /**< @brief Linear B Script */
		Lisu = UCD_SCRIPT_Lisu, /**< @brief Lisu Script */
		Loma = UCD_SCRIPT_Loma, /**< @brief Loma Script */
		Lyci = UCD_SCRIPT_Lyci, /**< @brief Lycian Script */
		Lydi = UCD_SCRIPT_Lydi, /**< @brief Lydian Script */
		Mahj = UCD_SCRIPT_Mahj, /**< @brief Mahajani Script */
		Maka = UCD_SCRIPT_Maka, /**< @brief Mahajani Script */
		Mand = UCD_SCRIPT_Mand, /**< @brief Mandaic Script */
		Mani = UCD_SCRIPT_Mani, /**< @brief Manichaean Script */
		Marc = UCD_SCRIPT_Marc, /**< @brief Marchen Script */
		Maya = UCD_SCRIPT_Maya, /**< @brief Mayan Hieroglyphs */
		Medf = UCD_SCRIPT_Medf, /**< @brief Medefaidrin (Oberi Okaime) Script */
		Mend = UCD_SCRIPT_Mend, /**< @brief Mende Kikakui Script */
		Merc = UCD_SCRIPT_Merc, /**< @brief Meroitic Cursive Script */
		Mero = UCD_SCRIPT_Mero, /**< @brief Meroitic Hieroglyphs */
		Mlym = UCD_SCRIPT_Mlym, /**< @brief Malayalam Script */
		Modi = UCD_SCRIPT_Modi, /**< @brief Modi Script */
		Mong = UCD_SCRIPT_Mong, /**< @brief Mongolian Script */
		Moon = UCD_SCRIPT_Moon, /**< @brief Moon Script */
		Mroo = UCD_SCRIPT_Mroo, /**< @brief Mro Script */
		Mtei = UCD_SCRIPT_Mtei, /**< @brief Meitei Mayek Script */
		Mult = UCD_SCRIPT_Mult, /**< @brief Multani Script */
		Mymr = UCD_SCRIPT_Mymr, /**< @brief Myanmar (Burmese) Script */
		Narb = UCD_SCRIPT_Narb, /**< @brief Old North Arabian Script */
		Nbat = UCD_SCRIPT_Nbat, /**< @brief Nabataean Script */
		Newa = UCD_SCRIPT_Newa, /**< @brief Newa Script */
		Nkgb = UCD_SCRIPT_Nkgb, /**< @brief Nakhi Geba Script */
		Nkoo = UCD_SCRIPT_Nkoo, /**< @brief N'Ko Script */
		Nshu = UCD_SCRIPT_Nshu, /**< @brief Nushu Script */
		Ogam = UCD_SCRIPT_Ogam, /**< @brief Ogham Script */
		Olck = UCD_SCRIPT_Olck, /**< @brief Ol Chiki Script */
		Orkh = UCD_SCRIPT_Orkh, /**< @brief Old Turkic Script */
		Orya = UCD_SCRIPT_Orya, /**< @brief Oriya Script */
		Osge = UCD_SCRIPT_Osge, /**< @brief Osage Script */
		Osma = UCD_SCRIPT_Osma, /**< @brief Osmanya Script */
		Palm = UCD_SCRIPT_Palm, /**< @brief Palmyrene Script */
		Pauc = UCD_SCRIPT_Pauc, /**< @brief Pau Cin Hau Script */
		Perm = UCD_SCRIPT_Perm, /**< @brief Old Permic */
		Phag = UCD_SCRIPT_Phag, /**< @brief Phags-Pa Script */
		Phli = UCD_SCRIPT_Phli, /**< @brief Inscriptional Pahlavi Script */
		Phlp = UCD_SCRIPT_Phlp, /**< @brief Psalter Pahlavi Script */
		Phlv = UCD_SCRIPT_Phlv, /**< @brief Book Pahlavi Script */
		Phnx = UCD_SCRIPT_Phnx, /**< @brief Phoenician Script */
		Plrd = UCD_SCRIPT_Plrd, /**< @brief Miao Script */
		Prti = UCD_SCRIPT_Prti, /**< @brief Inscriptional Parthian Script */
		Qaak = UCD_SCRIPT_Qaak, /**< @brief Klingon Script (Private Use) */
		Rjng = UCD_SCRIPT_Rjng, /**< @brief Rejang Script */
		Rohg = UCD_SCRIPT_Rohg, /**< @brief Hanifi Rohingya Script */
		Roro = UCD_SCRIPT_Roro, /**< @brief Rongorongo Script */
		Runr = UCD_SCRIPT_Runr, /**< @brief Runic Script */
		Samr = UCD_SCRIPT_Samr, /**< @brief Samaritan Script */
		Sara = UCD_SCRIPT_Sara, /**< @brief Sarati Script */
		Sarb = UCD_SCRIPT_Sarb, /**< @brief Old South Arabian Script */
		Saur = UCD_SCRIPT_Saur, /**< @brief Saurashtra Script */
		Sgnw = UCD_SCRIPT_Sgnw, /**< @brief Sign Writing */
		Shaw = UCD_SCRIPT_Shaw, /**< @brief Shavian Script */
		Shrd = UCD_SCRIPT_Shrd, /**< @brief Sharada Script */
		Sidd = UCD_SCRIPT_Sidd, /**< @brief Siddham Script */
		Sind = UCD_SCRIPT_Sind, /**< @brief Sindhi Script */
		Sinh = UCD_SCRIPT_Sinh, /**< @brief Sinhala Script */
		Sogd = UCD_SCRIPT_Sogd, /**< @brief Sogdian Script */
		Sogo = UCD_SCRIPT_Sogo, /**< @brief Old Sogdian Script */
		Sora = UCD_SCRIPT_Sora, /**< @brief Sora Sompeng Script */
		Soyo = UCD_SCRIPT_Soyo, /**< @brief Soyombo */
		Sund = UCD_SCRIPT_Sund, /**< @brief Sundanese Script */
		Sylo = UCD_SCRIPT_Sylo, /**< @brief Syloti Nagri Script */
		Syrc = UCD_SCRIPT_Syrc, /**< @brief Syriac Script */
		Syre = UCD_SCRIPT_Syre, /**< @brief Syriac Script (Estrangelo Variant) */
		Syrj = UCD_SCRIPT_Syrj, /**< @brief Syriac Script (Western Variant) */
		Syrn = UCD_SCRIPT_Syrn, /**< @brief Syriac Script (Eastern Variant) */
		Tagb = UCD_SCRIPT_Tagb, /**< @brief Tagbanwa Script */
		Takr = UCD_SCRIPT_Takr, /**< @brief Takri Script */
		Tale = UCD_SCRIPT_Tale, /**< @brief Tai Le Script */
		Talu = UCD_SCRIPT_Talu, /**< @brief New Tai Lue Script */
		Taml = UCD_SCRIPT_Taml, /**< @brief Tamil Script */
		Tang = UCD_SCRIPT_Tang, /**< @brief Tangut Script */
		Tavt = UCD_SCRIPT_Tavt, /**< @brief Tai Viet Script */
		Telu = UCD_SCRIPT_Telu, /**< @brief Telugu Script */
		Teng = UCD_SCRIPT_Teng, /**< @brief Tengwar Script */
		Tfng = UCD_SCRIPT_Tfng, /**< @brief Tifinagh Script */
		Tglg = UCD_SCRIPT_Tglg, /**< @brief Tagalog Script */
		Thaa = UCD_SCRIPT_Thaa, /**< @brief Thaana Script */
		Thai = UCD_SCRIPT_Thai, /**< @brief Thai Script */
		Tibt = UCD_SCRIPT_Tibt, /**< @brief Tibetan Script */
		Tirh = UCD_SCRIPT_Tirh, /**< @brief Tirhuta Script */
		Ugar = UCD_SCRIPT_Ugar, /**< @brief Ugaritic Script */
		Vaii = UCD_SCRIPT_Vaii, /**< @brief Vai Script */
		Visp = UCD_SCRIPT_Visp, /**< @brief Visible Speech Script */
		Wara = UCD_SCRIPT_Wara, /**< @brief Warang Citi Script */
		Wole = UCD_SCRIPT_Wole, /**< @brief Woleai Script */
		Xpeo = UCD_SCRIPT_Xpeo, /**< @brief Old Persian Script */
		Xsux = UCD_SCRIPT_Xsux, /**< @brief Cuneiform Script */
		Yiii = UCD_SCRIPT_Yiii, /**< @brief Yi Script */
		Zanb = UCD_SCRIPT_Zanb, /**< @brief Zanabazar Square */
		Zinh = UCD_SCRIPT_Zinh, /**< @brief Inherited Script */
		Zmth = UCD_SCRIPT_Zmth, /**< @brief Mathematical Notation */
		Zsym = UCD_SCRIPT_Zsym, /**< @brief Symbols */
		Zxxx = UCD_SCRIPT_Zxxx, /**< @brief Unwritten Documents */
		Zyyy = UCD_SCRIPT_Zyyy, /**< @brief Undetermined Script */
		Zzzz = UCD_SCRIPT_Zzzz, /**< @brief Uncoded Script */
	};

	/** @brief Get a string representation of the script enumeration value.
	  *
	  * @param s The value to get the string representation for.
	  *
	  * @return The string representation, or "----" if the value is not recognized.
	  */
	inline const char *get_script_string(script s)
	{
		return ucd_get_script_string((ucd_script)s);
	}

	/** @brief Lookup the Script for a Unicode codepoint.
	  *
	  * @param c The Unicode codepoint to lookup.
	  * @return  The Script of the Unicode codepoint.
	  */
	inline script lookup_script(codepoint_t c)
	{
		return (script)ucd_lookup_script(c);
	}

	/** @brief Properties
	 */
	typedef ucd_property property;
	enum
	{
		White_Space = UCD_PROPERTY_WHITE_SPACE, /**< @brief White_Space */
		Bidi_Control = UCD_PROPERTY_BIDI_CONTROL, /**< @brief Bidi_Control */
		Join_Control = UCD_PROPERTY_JOIN_CONTROL, /**< @brief Join_Control */
		Dash = UCD_PROPERTY_DASH, /**< @brief Dash */
		Hyphen = UCD_PROPERTY_HYPHEN, /**< @brief Hyphen */
		Quotation_Mark = UCD_PROPERTY_QUOTATION_MARK, /**< @brief Quotation_Mark */
		Terminal_Punctuation = UCD_PROPERTY_TERMINAL_PUNCTUATION, /**< @brief Terminal_Punctuation */
		Other_Math = UCD_PROPERTY_OTHER_MATH, /**< @brief Other_Math */
		Hex_Digit = UCD_PROPERTY_HEX_DIGIT, /**< @brief Hex_Digit */
		ASCII_Hex_Digit = UCD_PROPERTY_ASCII_HEX_DIGIT, /**< @brief ASCII_Hex_Digit */
		Other_Alphabetic = UCD_PROPERTY_OTHER_ALPHABETIC, /**< @brief Other_Alphabetic */
		Ideographic = UCD_PROPERTY_IDEOGRAPHIC, /**< @brief Ideographic */
		Diacritic = UCD_PROPERTY_DIACRITIC, /**< @brief Diacritic */
		Extender = UCD_PROPERTY_EXTENDER, /**< @brief Extender */
		Other_Lowercase = UCD_PROPERTY_OTHER_LOWERCASE, /**< @brief Other_Lowercase */
		Other_Uppercase = UCD_PROPERTY_OTHER_UPPERCASE, /**< @brief Other_Uppercase */
		Noncharacter_Code_Point = UCD_PROPERTY_NONCHARACTER_CODE_POINT, /**< @brief Noncharacter_Code_Point */
		Other_Grapheme_Extend = UCD_PROPERTY_OTHER_GRAPHEME_EXTEND, /**< @brief Other_Grapheme_Extend */
		IDS_Binary_Operator = UCD_PROPERTY_IDS_BINARY_OPERATOR, /**< @brief IDS_Binary_Operator */
		IDS_Trinary_Operator = UCD_PROPERTY_IDS_TRINARY_OPERATOR, /**< @brief IDS_Trinary_Operator */
		Radical = UCD_PROPERTY_RADICAL, /**< @brief Radical */
		Unified_Ideograph = UCD_PROPERTY_UNIFIED_IDEOGRAPH, /**< @brief Unified_Ideograph */
		Other_Default_Ignorable_Code_Point = UCD_PROPERTY_OTHER_DEFAULT_IGNORABLE_CODE_POINT, /**< @brief Other_Default_Ignorable_Code_Point */
		Deprecated = UCD_PROPERTY_DEPRECATED, /**< @brief Deprecated */
		Soft_Dotted = UCD_PROPERTY_SOFT_DOTTED, /**< @brief Soft_Dotted */
		Logical_Order_Exception = UCD_PROPERTY_LOGICAL_ORDER_EXCEPTION, /**< @brief Logical_Order_Exception */
		Other_ID_Start = UCD_PROPERTY_OTHER_ID_START, /**< @brief Other_ID_Start */
		Other_ID_Continue = UCD_PROPERTY_OTHER_ID_CONTINUE, /**< @brief Other_ID_Continue */
		Sentence_Terminal = UCD_PROPERTY_SENTENCE_TERMINAL, /**< @brief Sentence_Terminal */
		Variation_Selector = UCD_PROPERTY_VARIATION_SELECTOR, /**< @brief Variation_Selector */
		Pattern_White_Space = UCD_PROPERTY_PATTERN_WHITE_SPACE, /**< @brief Pattern_White_Space */
		Pattern_Syntax = UCD_PROPERTY_PATTERN_SYNTAX, /**< @brief Pattern_Syntax */
		Prepended_Concatenation_Mark = UCD_PROPERTY_PREPENDED_CONCATENATION_MARK, /**< @brief Prepended_Concatenation_Mark */
		Emoji = UCD_PROPERTY_EMOJI, /**< @brief Emoji */
		Emoji_Presentation = UCD_PROPERTY_EMOJI_PRESENTATION, /**< @brief Emoji_Presentation */
		Emoji_Modifier = UCD_PROPERTY_EMOJI_MODIFIER, /**< @brief Emoji_Modifier */
		Emoji_Modifier_Base = UCD_PROPERTY_EMOJI_MODIFIER_BASE, /**< @brief Emoji_Modifier_Base */
		Regional_Indicator = UCD_PROPERTY_REGIONAL_INDICATOR, /**< @brief Regional_Indicator */
		Emoji_Component = UCD_PROPERTY_EMOJI_COMPONENT, /**< @brief Emoji_Component */
		Extended_Pictographic = UCD_PROPERTY_EXTENDED_PICTOGRAPHIC, /**< @brief Extended_Pictographic */
	};

	/** @brief Return the properties of the specified codepoint.
	 *
	 * @param c   The Unicode codepoint to lookup.
	 * @param cat The General Category of the codepoint.
	 * @return    The properties associated with the codepoint.
	 */
	inline property properties(codepoint_t c, category cat)
	{
		return (property)ucd_properties(c, (ucd_category)cat);
	}

	/** @brief Is the codepoint in the 'alnum' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'alnum' class, zero otherwise.
	  */
	inline int isalnum(codepoint_t c)
	{
		return ucd_isalnum(c);
	}

	/** @brief Is the codepoint in the 'alpha' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'alpha' class, zero otherwise.
	  */
	inline int isalpha(codepoint_t c)
	{
		return ucd_isalpha(c);
	}

	/** @brief Is the codepoint in the 'blank' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'blank' class, zero otherwise.
	  */
	inline int isblank(codepoint_t c)
	{
		return ucd_isblank(c);
	}

	/** @brief Is the codepoint in the 'cntrl' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'cntrl' class, zero otherwise.
	  */
	inline int iscntrl(codepoint_t c)
	{
		return ucd_iscntrl(c);
	}

	/** @brief Is the codepoint in the 'digit' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'digit' class, zero otherwise.
	  */
	inline int isdigit(codepoint_t c)
	{
		return ucd_isdigit(c);
	}

	/** @brief Is the codepoint in the 'graph' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'graph' class, zero otherwise.
	  */
	inline int isgraph(codepoint_t c)
	{
		return ucd_isgraph(c);
	}

	/** @brief Is the codepoint in the 'lower' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'lower' class, zero otherwise.
	  */
	inline int islower(codepoint_t c)
	{
		return ucd_islower(c);
	}

	/** @brief Is the codepoint in the 'print' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'print' class, zero otherwise.
	  */
	inline int isprint(codepoint_t c)
	{
		return ucd_isprint(c);
	}

	/** @brief Is the codepoint in the 'punct' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'punct' class, zero otherwise.
	  */
	inline int ispunct(codepoint_t c)
	{
		return ucd_ispunct(c);
	}

	/** @brief Is the codepoint in the 'space' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'space' class, zero otherwise.
	  */
	inline int isspace(codepoint_t c)
	{
		return ucd_isspace(c);
	}

	/** @brief Is the codepoint in the 'upper' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'upper' class, zero otherwise.
	  */
	inline int isupper(codepoint_t c)
	{
		return ucd_isupper(c);
	}

	/** @brief Is the codepoint in the 'xdigit' class?
	  *
	  * @param c The Unicode codepoint to check.
	  * @return  Non-zero if the codepoint is in the 'xdigit' class, zero otherwise.
	  */
	inline int isxdigit(codepoint_t c)
	{
		return ucd_isxdigit(c);
	}

	/** @brief Convert the Unicode codepoint to upper-case.
	  *
	  * This function only uses the simple case mapping present in the
	  * UnicodeData file. The data in SpecialCasing requires Unicode
	  * codepoints to be mapped to multiple codepoints.
	  *
	  * @param c The Unicode codepoint to convert.
	  * @return  The upper-case Unicode codepoint for this codepoint, or
	  *          this codepoint if there is no upper-case codepoint.
	  */
	inline codepoint_t toupper(codepoint_t c)
	{
		return ucd_toupper(c);
	}

	/** @brief Convert the Unicode codepoint to lower-case.
	  *
	  * This function only uses the simple case mapping present in the
	  * UnicodeData file. The data in SpecialCasing requires Unicode
	  * codepoints to be mapped to multiple codepoints.
	  *
	  * @param c The Unicode codepoint to convert.
	  * @return  The lower-case Unicode codepoint for this codepoint, or
	  *          this codepoint if there is no upper-case codepoint.
	  */
	inline codepoint_t tolower(codepoint_t c)
	{
		return ucd_tolower(c);
	}

	/** @brief Convert the Unicode codepoint to title-case.
	  *
	  * This function only uses the simple case mapping present in the
	  * UnicodeData file. The data in SpecialCasing requires Unicode
	  * codepoints to be mapped to multiple codepoints.
	  *
	  * @param c The Unicode codepoint to convert.
	  * @return  The title-case Unicode codepoint for this codepoint, or
	  *          this codepoint if there is no upper-case codepoint.
	  */
	inline codepoint_t totitle(codepoint_t c)
	{
		return ucd_totitle(c);
	}
}
#endif

#endif
