#!/usr/bin/python

# Copyright (C) 2012-2017 Reece H. Dunn
#
# This file is part of ucd-tools.
#
# ucd-tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ucd-tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ucd-tools.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import ucd

ucd_rootdir = sys.argv[1]
emoji_rootdir = 'data/emoji'
csur_rootdir = 'data/csur'

null = ucd.CodePoint('0000')

properties = [
    (ucd_rootdir, 'PropList'),
    (ucd_rootdir, 'DerivedCoreProperties'),
    (emoji_rootdir, 'emoji-data'),
    ('data/espeak-ng', 'PropList')
]

unicode_chars = {}
for data in ucd.parse_ucd_data(ucd_rootdir, 'UnicodeData'):
	for codepoint in data['CodePoint']:
		unicode_chars[codepoint] = data
for propdir, propfile in properties:
	for data in ucd.parse_ucd_data(propdir, propfile):
		for codepoint in data['Range']:
			try:
				unicode_chars[codepoint][data['Property']] = 1
			except KeyError:
				unicode_chars[codepoint] = {'CodePoint': codepoint}
				unicode_chars[codepoint][data['Property']] = 1
for data in ucd.parse_ucd_data(ucd_rootdir, 'Scripts'):
	for codepoint in data['Range']:
		unicode_chars[codepoint]['Script'] = data['Script']
if '--with-csur' in sys.argv:
	for csur in ['Klingon']:
		for data in ucd.parse_ucd_data('data/csur', csur):
			for codepoint in data['CodePoint']:
				unicode_chars[codepoint] = data

def iscntrl(data):
	return 1 if data.get('Name', '') == '<control>' else 0

def isdigit(data):
	return 1 if data['CodePoint'].char() in '0123456789' else 0

def isxdigit(data):
	return 1 if data['CodePoint'].char() in '0123456789ABCDEFabcdef' else 0

def isspace(data):
	if data.get('White_Space', 0):
		dt = data.get('DecompositionType', '')
		return 1 if dt == None or not dt.startswith('<noBreak>') else 0
	else:
		return 0

def isblank(data): # word separator
	if data.get('GeneralCategory', 'Cn') == 'Zs' or data['CodePoint'].char() == '\t':
		dt = data.get('DecompositionType', '')
		return 1 if dt == None or not dt.startswith('<noBreak>') else 0
	else:
		return 0

def ispunct(data):
	return 1 if isgraph(data) and not isalnum(data) else 0

def isprint(data):
	if data.get('GeneralCategory', 'Cn')[0] in 'LMNPSZ': # not in 'CI'
		return 1
	else:
		return 0

def isgraph(data):
	if data.get('GeneralCategory', 'Cn')[0] in 'LMNPS': # not in 'CZI'
		return 1
	else:
		return 0

def isalnum(data):
	if data.get('GeneralCategory', 'Cn')[0] in 'N':
		return 1
	else:
		return data.get('Alphabetic', 0)

def isalpha(data):
	return data.get('Alphabetic', 0)

def isupper(data):
	if data.get('Uppercase', 0):
		return 1
	elif data.get('LowerCase', null) != null: # Some Lt characters have lowercase forms.
		return 1
	else:
		return 0

def islower(data):
	if data.get('Lowercase', 0):
		return 1
	elif data.get('UpperCase', null) != null:
		return 1
	else:
		return 0

def decomposition_type(data, dtype):
	value = data.get('DecompositionType', None)
	if value and value.startswith(dtype):
		return value
	return None

def properties(data):
	props  = 0
	props += (2 **  0) * data.get('White_Space', 0)
	props += (2 **  1) * data.get('Bidi_Control', 0)
	props += (2 **  2) * data.get('Join_Control', 0)
	props += (2 **  3) * data.get('Dash', 0)
	props += (2 **  4) * data.get('Hyphen', 0)
	props += (2 **  5) * data.get('Quotation_Mark', 0)
	props += (2 **  6) * data.get('Terminal_Punctuation', 0)
	props += (2 **  7) * data.get('Other_Math', 0)
	props += (2 **  8) * data.get('Hex_Digit', 0)
	props += (2 **  9) * data.get('ASCII_Hex_Digit', 0)
	props += (2 ** 10) * data.get('Other_Alphabetic', 0)
	props += (2 ** 11) * data.get('Ideographic', 0)
	props += (2 ** 12) * data.get('Diacritic', 0)
	props += (2 ** 13) * data.get('Extender', 0)
	props += (2 ** 14) * data.get('Other_Lowercase', 0)
	props += (2 ** 15) * data.get('Other_Uppercase', 0)
	props += (2 ** 16) * data.get('Noncharacter_Code_Point', 0)
	props += (2 ** 17) * data.get('Other_Grapheme_Extend', 0)
	props += (2 ** 18) * data.get('IDS_Binary_Operator', 0)
	props += (2 ** 19) * data.get('IDS_Trinary_Operator', 0)
	props += (2 ** 20) * data.get('Radical', 0)
	props += (2 ** 21) * data.get('Unified_Ideograph', 0)
	props += (2 ** 22) * data.get('Other_Default_Ignorable_Code_Point', 0)
	props += (2 ** 23) * data.get('Deprecated', 0)
	props += (2 ** 24) * data.get('Soft_Dotted', 0)
	props += (2 ** 25) * data.get('Logical_Order_Exception', 0)
	props += (2 ** 26) * data.get('Other_ID_Start', 0)
	props += (2 ** 27) * data.get('Other_ID_Continue', 0)
	props += (2 ** 28) * data.get('Sentence_Terminal', 0)
	props += (2 ** 29) * data.get('Variation_Selector', 0)
	props += (2 ** 30) * data.get('Pattern_White_Space', 0)
	props += (2 ** 31) * data.get('Pattern_Syntax', 0)
	props += (2 ** 32) * data.get('Prepended_Concatenation_Mark', 0)
	props += (2 ** 33) * data.get('Emoji', 0) # emoji-data
	props += (2 ** 34) * data.get('Emoji_Presentation', 0) # emoji-data
	props += (2 ** 35) * data.get('Emoji_Modifier', 0) # emoji-data
	props += (2 ** 36) * data.get('Emoji_Modifier_Base', 0) # emoji-data
	props += (2 ** 37) * data.get('Regional_Indicator', 0) # PropList 10.0.0
	props += (2 ** 38) * data.get('Emoji_Component', 0) # emoji-data 5.0
	props += (2 ** 39) * data.get('Extended_Pictographic', 0) # emoji-data 11.0
        # eSpeak NG extended properties:
	props += (2 ** 52) * data.get('Inverted_Terminal_Punctuation', 0)
	props += (2 ** 53) * data.get('Punctuation_In_Word', 0)
	props += (2 ** 54) * data.get('Optional_Space_After', 0)
	props += (2 ** 55) * data.get('Extended_Dash', 0)
	props += (2 ** 56) * data.get('Paragraph_Separator', 0)
	props += (2 ** 57) * data.get('Ellipsis', 0)
	props += (2 ** 58) * data.get('Semi_Colon', 0)
	props += (2 ** 59) * data.get('Colon', 0)
	props += (2 ** 60) * data.get('Comma', 0)
	props += (2 ** 61) * data.get('Exclamation_Mark', 0)
	props += (2 ** 62) * data.get('Question_Mark', 0)
	props += (2 ** 63) * data.get('Full_Stop', 0)
	return props

if __name__ == '__main__':
	for codepoint in ucd.CodeRange('000000..10FFFF'):
		try:
			data = unicode_chars[codepoint]
		except KeyError:
			data = {'CodePoint': codepoint}
		script = data.get('Script', 'Zzzz')
		title = data.get('TitleCase', codepoint)
		upper = data.get('UpperCase', codepoint)
		lower = data.get('LowerCase', codepoint)
		if title == null: title = codepoint
		if upper == null: upper = codepoint
		if lower == null: lower = codepoint
		print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %016x' % (
		      codepoint, script,
		      data.get('GeneralCategory', 'Cn')[0], data.get('GeneralCategory', 'Cn'),
		      upper, lower, title,
		      isdigit(data), isxdigit(data),
		      iscntrl(data), isspace(data), isblank(data), ispunct(data),
		      isprint(data), isgraph(data), isalnum(data), isalpha(data), isupper(data), islower(data),
		      properties(data)))
