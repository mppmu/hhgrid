AC_INIT([hhgrid], [0.4])
AC_CONFIG_SRCDIR([src/hhgrid.h])
AC_CONFIG_MACRO_DIR([acinclude.d])

AC_ARG_ENABLE([fortran],
  [AS_HELP_STRING([--disable-fortran], [do not build Fortran support check programs @<:@check@:>@])],
  [case "${enableval}" 
     in yes | no) 
       ENABLE_FORTRAN ="${enableval}" ;;
     *) 
       AC_MSG_ERROR(bad value ${enableval} for --disable-fortran) ;;
   esac],
  [ENABLE_FORTRAN="yes"])
AM_CONDITIONAL([ENABLE_FORTRAN], [test "x$ENABLE_FORTRAN" = "xyes"])

AM_INIT_AUTOMAKE

AC_DISABLE_STATIC

LT_INIT
AC_PROG_FC
AC_PROG_CC
AC_PROG_CXX

AM_PATH_PYTHON([3])
AX_PYTHON_VERSION_ENSURE([3.11])
AX_PYTHON_CSPEC
AX_PYTHON_LSPEC


AX_CXX_COMPILE_STDCXX_11([noext]) # noext: use -std=c++11 rather than -std=gnu++11

AC_CHECK_HEADERS([stdlib.h assert.h])

AC_SUBST(PYTHON_CSPEC)
AC_SUBST(PYTHON_LSPEC)
AC_CONFIG_FILES(Makefile src/Makefile check/Makefile share/Makefile share/grids_sm/Makefile share/grids_eft/Makefile hhgrid-config)

AC_OUTPUT
