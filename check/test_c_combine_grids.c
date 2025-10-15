
#include "hhgrid.h"

#include <stdio.h> // printf
#include <stdlib.h> // abs

int equal(double a, double b)
{
    if( fabs(a-b) > 1.0e-14 )
    {
        return 0;
    }
    return 1;
}

int test_point(PyObject* pGrid, double s, double t, double expected)
{
    double res;
    printf("Sending: %1.15f %1.15f\n",s,t);
    res = grid_virt(pGrid,s,t);
    printf("Received: %1.15f\n",res);
    if ( equal(res,expected) != 1 )
    {
        printf("Expected: %1.25f, Got: %1.25f",expected,res);
        printf("\nTESTS FAILED\n\n");

        // Destruct grid, terminate python
        python_decref(pGrid);
        python_finalize();

        return 0;
    }
    return 1;
}

int main()
{
    double expected;
    double s;
    double t;

    // Initialise python and grid
    python_initialize();
    python_printinfo();
    const char* grid_name = "grids_eft/Virt_full";
    combine_grids(grid_name, 1., 2., 3., 4., 5., 3., 0., 90400, 1.);

    const char* combined_grid_name = "grids_eft/Virt_full_+1.000000E+00_+2.000000E+00_+3.000000E+00_+4.000000E+00_+5.000000E+00_EFTcount3_usesmeft0_lhaid90400_renfac+1.000000E+00_combined.grid";

    PyObject* pGrid = grid_initialize(combined_grid_name);

    // Test point 1
    s = 250000.e0;
    t = -50000.e0;
    expected = 2.253828988312267245230913e-01; // Expected result from Virt_EFT.grid
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Destruct grid, terminate python
    python_decref(pGrid);
    python_finalize();

    printf("\nTESTS PASSED\n\n");

    return 0;
};
