
extern "C" {
#include "hhgrid.h"
}

#include <iostream> // std::cout, std::scientific
#include <string>   // std::string

int main()
{
    // Initialise python and grid
    python_initialize();
    python_printinfo();
    std::string grid_name = "grids_sm/Virt_EFT.grid";
    PyObject* pGrid = grid_initialize(grid_name.c_str());

    // Evaluate point
    double s = 2.56513e6;
    double t = -482321.e0;
    double result = grid_virt(pGrid,s,t);

    std::cout << "Sent: " << s << " " << t << std::endl;
    std::cout << "Received: " << result << std::endl;

    // Destruct grid, terminate python
    python_decref(pGrid);
    python_finalize();

    return 0;
};
