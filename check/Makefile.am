check_PROGRAMS = test_c_grid.x test_cpp_grid.x 
check_SCRIPTS = test_c_grid.sh test_cpp_grid.sh

AM_CPPFLAGS = -I$(top_srcdir)/src @PYTHON_CSPEC@
LDADD = ../src/libhhgrid.la $(PYTHON_LSPEC)

test_c_grid_x_SOURCES = test_c_grid.c
test_cpp_grid_x_SOURCES = test_cpp_grid.cpp

if ENABLE_FORTRAN
check_PROGRAMS += test_f90_grid.x
check_SCRIPTS += test_f90_grid.sh
test_f90_grid_x_SOURCES = test_f90_grid.f90
endif

TESTS = $(check_SCRIPTS)
