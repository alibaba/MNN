AC_PREREQ([2.65])
AC_INIT([Unicode Character Database Tools], [11.0.0], [https://github.com/rhdunn/ucd-tools/issues], [ucd-tools], [https://github.com/rhdunn/ucd-tools])
AM_INIT_AUTOMAKE()

m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES])
AM_SILENT_RULES([yes])

AC_CONFIG_SRCDIR([src])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

LT_INIT

dnl ================================================================
dnl Program checks.
dnl ================================================================

AC_PROG_CC
AC_PROG_CXX
AC_PROG_MAKE_SET

dnl ================================================================
dnl library checks.
dnl ================================================================

AC_CHECK_HEADERS([stddef.h])   dnl C89
AC_CHECK_FUNCS([iswblank])     dnl C99

AC_TYPE_UINT8_T
AC_TYPE_UINT32_T

dnl ================================================================
dnl UCD data configuration.
dnl ================================================================

AC_ARG_WITH([unicode-version],
    [AS_HELP_STRING([--with-unicode-version], [Unicode version to support @<:@default=11.0.0@:>@])],
    [AS_IF([test x"$withval" != x],
           [UCD_VERSION="$withval"])],
    [UCD_VERSION="11.0.0"])

AC_SUBST(UCD_VERSION)

AC_ARG_WITH([emoji-version],
    [AS_HELP_STRING([--with-emoji-version], [Unicode emoji version to support @<:@default=11.0@:>@])],
    [AS_IF([test x"$withval" != x],
           [EMOJI_VERSION="$withval"])],
    [EMOJI_VERSION="11.0"])

AC_SUBST(EMOJI_VERSION)

dnl ================================================================
dnl Compiler warnings.
dnl
dnl Reference: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
dnl ================================================================

AX_CHECK_COMPILE_FLAG([-Wmissing-prototypes],    [CFLAGS="-Wmissing-prototypes $CFLAGS"])
AX_CHECK_COMPILE_FLAG([-Wreturn-type],           [CFLAGS="-Wreturn-type $CFLAGS"])

dnl ================================================================
dnl Generate output.
dnl ================================================================

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

AC_MSG_NOTICE([

    Configuration for Unicode Character Data Tools complete.

        Source code location:          ${srcdir}

        C Compiler:                    ${CC}
        C Compiler flags:              ${CFLAGS}

        C++ Compiler:                  ${CXX}
        C++ Compiler flags:            ${CXXFLAGS}

        Unicode version:               ${UCD_VERSION}
        Emoji version:                 ${EMOJI_VERSION}
])
