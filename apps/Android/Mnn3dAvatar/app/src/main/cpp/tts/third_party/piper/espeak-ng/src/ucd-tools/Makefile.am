AUTOMAKE_OPTIONS = subdir-objects

localedir  = $(datadir)/locale
xdgdatadir = @XDGDATADIR@
mimedir    = $(xdgdatadir)/mime
xmldir     = $(mimedir)/packages

AM_LDFLAGS  = ${LTLIBINTL}
AM_CFLAGS   = -Isrc/include -Isrc
AM_CXXFLAGS = -Isrc/include -Isrc

ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS =
lib_LTLIBRARIES =
man1_MANS =
pkgdata_DATA =

noinst_bin_PROGRAMS =
noinst_LIBRARIES =

noinst_bindir =

EXTRA_DIST = config.rpath ChangeLog
CLEANFILES =
SUBDIRS =

############################# libtool #########################################

EXTRA_DIST += config.guess config.sub ltmain.sh

# Increment if the interface has changed and is not backward compatible
CURRENT=0

# Increment  if source files have changed
# Reset to 0 if the interface has changed
REVISION=0

# Increment  if the interface is backward compatible (superset)
# Reset to 0 if the interface is not backward compatible
AGE=0

LIBUCD_VERSION=$(CURRENT):$(REVISION):$(AGE)

############################# ChangeLog #######################################

ChangeLog:
	git log > ChangeLog

dist-hook: ChangeLog

.PHONY: ChangeLog

EXTRA_DIST += ChangeLog

############################# Unicode Data ####################################

EMOJI_VERSION=@EMOJI_VERSION@
UCD_VERSION=@UCD_VERSION@
UCD_ROOTDIR=data/ucd
UCD_SRCDIR=http://www.unicode.org/Public

data/emoji/emoji-data.txt:
	mkdir -pv data/emoji
	curl ${UCD_SRCDIR}/emoji/${EMOJI_VERSION}/emoji-data.txt -o $@

data/ucd/PropList.txt:
	mkdir -pv data/ucd
	curl ${UCD_SRCDIR}/${UCD_VERSION}/ucd/PropList.txt -o $@

data/ucd/DerivedCoreProperties.txt:
	mkdir -pv data/ucd
	curl ${UCD_SRCDIR}/${UCD_VERSION}/ucd/DerivedCoreProperties.txt -o $@

data/ucd/PropertyValueAliases.txt:
	mkdir -pv data/ucd
	curl ${UCD_SRCDIR}/${UCD_VERSION}/ucd/PropertyValueAliases.txt -o $@

data/ucd/Scripts.txt:
	mkdir -pv data/ucd
	curl ${UCD_SRCDIR}/${UCD_VERSION}/ucd/Scripts.txt -o $@

data/ucd/UnicodeData.txt:
	mkdir -pv data/ucd
	curl ${UCD_SRCDIR}/${UCD_VERSION}/ucd/UnicodeData.txt -o $@

############################# documentation ###################################

SUFFIXES=.html .md

.md.html: _layouts/webpage.html
	cat $< | sed -e 's/\.md)/.html)/g' | \
		kramdown --template _layouts/webpage.html > $@

html:
	doxygen docs/ucd-tools.conf

docs:	html \
	ChangeLog.html \
	README.html

############################# libucd ##########################################

tools/ucd.py: data/ucd/PropertyValueAliases.txt

tools/case.py: tools/ucd.py \
	data/ucd/UnicodeData.txt

tools/categories.py: tools/ucd.py \
	data/ucd/UnicodeData.txt

tools/scripts.py: tools/ucd.py \
	data/ucd/Scripts.txt

ucd-update: tools/case.py tools/categories.py tools/scripts.py
	tools/case.py ${UCD_ROOTDIR} ${UCD_VERSION} ${UCD_FLAGS} > src/case.c
	tools/categories.py ${UCD_ROOTDIR} ${UCD_VERSION} ${UCD_FLAGS} > src/categories.c
	tools/scripts.py ${UCD_ROOTDIR} ${UCD_VERSION} ${UCD_FLAGS} > src/scripts.c

libucd_includedir = $(includedir)/ucd
libucd_include_HEADERS = \
	src/include/ucd/ucd.h

lib_LTLIBRARIES       += src/libucd.la
src_libucd_la_LDFLAGS  = -version-info $(LIBUCD_VERSION)
src_libucd_la_SOURCES  = \
	src/case.c \
	src/categories.c \
	src/ctype.c \
	src/proplist.c \
	src/scripts.c \
	src/tostring.c

############################# tests ###########################################

noinst_bin_PROGRAMS     += tests/printcdata
tests_printcdata_SOURCES = tests/printcdata.c
tests_printcdata_LDADD   = src/libucd.la

noinst_bin_PROGRAMS       += tests/printucddata
tests_printucddata_SOURCES = tests/printucddata.c
tests_printucddata_LDADD   = src/libucd.la

noinst_bin_PROGRAMS           += tests/printucddata_cpp
tests_printucddata_cpp_SOURCES = tests/printucddata_cpp.cpp
tests_printucddata_cpp_LDADD   = src/libucd.la

tests/unicode-data.expected: tools/printdata.py tools/ucd.py \
	data/emoji/emoji-data.txt \
	data/espeak-ng/PropList.txt \
	data/ucd/UnicodeData.txt \
	data/ucd/PropList.txt \
	data/ucd/DerivedCoreProperties.txt \
	data/ucd/Scripts.txt
	tools/printdata.py ${UCD_ROOTDIR} ${UCD_FLAGS} > $@

tests/unicode-data-capi.actual: tests/printucddata
	tests/printucddata > $@

tests/unicode-data-cppapi.actual: tests/printucddata_cpp
	tests/printucddata_cpp > $@

tests/unicode-data-capi.diff: tests/unicode-data.expected tests/unicode-data-capi.actual
	diff -U0 tests/unicode-data.expected tests/unicode-data-capi.actual > tests/unicode-data-capi.diff

tests/unicode-data-cppapi.diff: tests/unicode-data.expected tests/unicode-data-cppapi.actual
	diff -U0 tests/unicode-data.expected tests/unicode-data-cppapi.actual > tests/unicode-data-cppapi.diff

check: tests/unicode-data-capi.diff tests/unicode-data-cppapi.diff
