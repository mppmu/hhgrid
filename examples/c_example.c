
#include "hhgrid.h"

#include <stdio.h> // printf

int main()
{
    // Initialise python and grid
    python_initialize();
    python_printinfo();
    const char* grid_name = "Virt_EFT.grid";
    PyObject* pGrid = grid_initialize(grid_name);

    double s = 2.56513e6;
    double t = -482321.e0;
    double result = grid_virt(pGrid, s, t);

    printf("Sent: %f %f\n",s ,t);
    printf("Received: %f\n", result);

    python_decref(pGrid);
    python_finalize();

    return 0;
};
