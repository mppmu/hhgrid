
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
    printf("Sending: %f %f\n",s,t);
    res = grid_virt(pGrid,s,t);
    printf("Received: %f\n",res);
    if ( equal(res,expected) != 1 )
    {
        printf("Expected: %f, Got: %f",expected,res);
        printf("TESTS FAILED");

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
    const char* grid_name = "Virt_EFT.grid";
    PyObject* pGrid = grid_initialize(grid_name);

    // Test point 1
    s = 250000.e0;
    t = -50000.e0;
    expected = 3.3875286683507290375755305333882461127359e-04; // Expected result from Virt_EFT.grid
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Test point 2
    s = 66887.6e0;
    t = -9407.43e0;
    expected = 7.6844534557689569490555349384752759078765e-07;
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Test point 3
    s = 83055.9e0;
    t = -11438.e0;
    expected = 5.6368756265788111468190146879919666389469e-06;
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Test point 4
    s = 414983.e0;
    t = -59786.9e0;
    expected = 1.0669994178547940674728344845334504498169e-03;
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Test point 5
    s = 2.56513e6;
    t = -482321.e0;
    expected = 3.9170655780375776555679578905255766585469e-02;
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Evaluate point repeatedly (can detect synchronization issues with the grid)
    int i;
    for(i=0; i<10000; i++)
    {
        if (test_point(pGrid,s,t,expected) != 1)
        return 1;
    }

    // Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
    s = 275621.46328957588411867618560791015625e0;
    t = -243368.2897650574450381100177764892578125e0;
    expected = 4.3499409718481744750728790194216344389133e-04;
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
    s = 959959.10608519916422665119171142578125e0;
    t = -928446.149935275432653725147247314453125e0;
    expected = 5.9433502098979523395327895229911518981680e-03;
    if (test_point(pGrid,s,t,expected) != 1)
    return 1;

    // Destruct grid, terminate python
    python_decref(pGrid);
    python_finalize();

    printf("\nTESTS PASSED\n\n");

    return 0;
};
