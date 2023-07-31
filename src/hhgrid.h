#ifndef _hhgrid_H_
#define _hhgrid_H_

#include <Python.h>

extern void python_initialize(void);
extern void python_decref(PyObject* grid);
extern void python_finalize(void);
extern void python_printinfo(void);

extern PyObject* grid_initialize(const char* grid_name);
extern double grid_virt(PyObject* grid, double s, double t);

#endif
