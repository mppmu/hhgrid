
#include "hhgrid.h"

#include <stdio.h> // printf

int main()
{
    // Initialise python and grid
    python_initialize();
    python_printinfo();
    const char* grid_name = "grids_eft/Virt_full";
    combine_grids(grid_name, 2, 0, 0, 0, 0, 3, 0, 90400, 1.);
    python_finalize();

    return 0;
};
