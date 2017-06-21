
extern "C" {
#include "hhgrid.h"
}

#include <iostream> // std::cout, std::scientific
#include <string>   // std::string
#include <limits>   // std::numeric_limits
#include <cmath>    // std::abs

bool equal(double a, double b)
{
    if( std::abs(a-b) > std::numeric_limits<double>::epsilon() )
    {
        return false;
    }
    return true;
}

bool test_point(PyObject* pGrid, double s, double t, double expected)
{
    double res;
    std::cout << "Sending: " << s << " " << t << std::endl;
    res = grid_virt(pGrid,s,t);
    std::cout << "Received: " << res << std::endl;

    if ( !equal(res,expected) )
    {
        std::cout << "Expected: " << expected << ", Got: " << res << std::endl;
        std::cout << "TESTS FAILED" << std::endl;

        // Destruct grid, terminate python
        python_decref(pGrid);
        python_finalize();

        return false;
    }
    return true;
}

int main()
{
    double expected;
    double s;
    double t;

    // Initialise python and grid
    python_initialize();
    python_printinfo();
    std::string grid_name = "Virt_EFT.grid";
    PyObject* pGrid = grid_initialize(grid_name.c_str());

    // Test point 1
    s = 250000.e0;
    t = -50000.e0;
    expected = 3.3875286683507290375755305333882461127359e-04; // Expected result from Virt_EFT.grid
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Test point 2
    s = 66887.6e0;
    t = -9407.43e0;
    expected = 7.6844534557689569490555349384752759078765e-07;
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Test point 3
    s = 83055.9e0;
    t = -11438.e0;
    expected = 5.6368756265788111468190146879919666389469e-06;
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Test point 4
    s = 414983.e0;
    t = -59786.9e0;
    expected = 1.0669994178547940674728344845334504498169e-03;
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Test point 5
    s = 2.56513e6;
    t = -482321.e0;
    expected = 3.9170655780375776555679578905255766585469e-02;
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Evaluate point repeatedly (can detect synchronization issues with the grid)
    for(int i=0; i<10000; i++)
    {
        if(!test_point(pGrid,s,t,expected))
            return 1;
    }

    // Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
    s = 275621.46328957588411867618560791015625e0;
    t = -243368.2897650574450381100177764892578125e0;
    expected = 4.3499409718481744750728790194216344389133e-04;
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
    s = 959959.10608519916422665119171142578125e0;
    t = -928446.149935275432653725147247314453125e0;
    expected = 5.9433502098979523395327895229911518981680e-03;
    if (!test_point(pGrid,s,t,expected))
        return 1;

    // Destruct grid, terminate python
    python_decref(pGrid);
    python_finalize();

    std::cout << std::endl;
    std::cout << "TESTS PASSED" << std::endl;
    std::cout << std::endl;

    return 0;
};
