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

script_map = {}

class CodePoint:
	def __init__(self, x):
		if isinstance(x, str):
			self.codepoint = int(x, 16)
		else:
			self.codepoint = x

	def __repr__(self):
		return '%06X' % self.codepoint

	def __str__(self):
		return '%06X' % self.codepoint

	def __iter__(self):
		yield self

	def __hash__(self):
		return self.codepoint

	def __eq__(self, other):
		return self.codepoint == other.codepoint

	def __ne__(self, other):
		return self.codepoint != other.codepoint

	def __lt__(self, other):
		return self.codepoint < other.codepoint

	def char(self):
		return unichr(self.codepoint)

class CodeRange:
	def __init__(self, x):
		f, l = x.split('..')
		self.first = CodePoint(f)
		self.last  = CodePoint(l)

	def __repr__(self):
		return '%s..%s' % (self.first, self.last)

	def __str__(self):
		return '%s..%s' % (self.first, self.last)

	def __iter__(self):
		for c in range(self.first.codepoint, self.last.codepoint + 1):
			yield CodePoint(c)

	def size(self):
		return self.last.codepoint - self.first.codepoint + 1

	def char(self):
		return unichr(self.first.codepoint)

def codepoint(x):
	if '..' in x[0]:
		return CodeRange(x[0]), x[1:]
	if ' ' in x:
		return [CodePoint(c) for c in x[0].split()], x[1:]
	if x[0] == '':
		return CodePoint('0000'), x[1:]
	return CodePoint(x[0]), x[1:]

def string(x):
	if x[0] == '':
		return None, x[1:]
	return x[0], x[1:]

def integer(x):
	return int(x[0]), x[1:]

def boolean(x):
	if x[0] == 'Y':
		return True, x[1:]
	return False, x[1:]

def script(x):
	return script_map[x[0]], x[1:]

def strlist(x):
	return x, []

data_items = {
	# Unicode Character Data:
	'emoji-data': [
		('Range', codepoint),
		('Property', string)
	],
	'Blocks': [
		('Range', codepoint),
		('Name', string)
	],
	'DerivedAge': [
		('Range', codepoint),
		('Age', string),
	],
	'DerivedCoreProperties': [
		('Range', codepoint),
		('Property', string),
	],
	'PropList': [
		('Range', codepoint),
		('Property', string),
	],
	'PropertyValueAliases': [
		('Property', string),
		('Key', string),
		('Value', string),
		('Aliases', strlist),
	],
	'Scripts': [
		('Range', codepoint),
		('Script', script),
	],
	'UnicodeData': [
		('CodePoint', codepoint),
		('Name', string),
		('GeneralCategory', string),
		('CanonicalCombiningClass', integer),
		('BidiClass', string),
		('DecompositionType', string),
		('DecompositionMapping', string),
		('NumericType', string),
		('NumericValue', string),
		('BidiMirrored', boolean),
		('UnicodeName', string),
		('ISOComment', string),
		('UpperCase', codepoint),
		('LowerCase', codepoint),
		('TitleCase', codepoint),
	],
	# ConScript Unicode Registry Data:
	'Klingon': [
		('CodePoint', codepoint),
		('Script', string),
		('GeneralCategory', string),
		('Name', string),
		('Transliteration', string),
	],
}

def parse_ucd_data(ucd_rootdir, dataset):
	keys  = data_items[dataset]
	first = None
	with open(os.path.join(ucd_rootdir, '%s.txt' % dataset)) as f:
		for line in f:
			line = line.replace('\n', '').split('#')[0]
			linedata = [' '.join(x.split()) for x in line.split(';')]
			if len(linedata) > 1:
				if linedata[1].endswith(', First>'):
					first = linedata
					continue

				if linedata[1].endswith(', Last>'):
					linedata[0] = '%s..%s' % (first[0], linedata[0])
					linedata[1] = linedata[1].replace(', Last>', '').replace('<', '')
					first = None

				data = {}
				for key, typemap in keys:
					data[key], linedata = typemap(linedata)
				yield data

def parse_property_mapping(ucd_rootdir, propname, reverse=False):
	ret = {}
	for data in parse_ucd_data(ucd_rootdir, 'PropertyValueAliases'):
		if data['Property'] == propname:
			if reverse:
				ret[data['Value']] = data['Key']
			else:
				ret[data['Key']] = data['Value']
	return ret

if __name__ == '__main__':
	try:
		items = sys.argv[3].split(',')
	except:
		items = None
	script_map = parse_property_mapping(sys.argv[1], 'sc', reverse=True)
	for entry in parse_ucd_data(sys.argv[1], sys.argv[2]):
		if items:
			print(','.join([str(entry[item]) for item in items]))
		else:
			print(entry)
else:
	script_map = parse_property_mapping('data/ucd', 'sc', reverse=True)
