#include <iostream>  // std::cout, std::scientific
#include <iomanip>   // std::setfill, std::setw, std::setprecision
#include <iterator>  // std::istreambuf_iterator
#include <string>    // std::string, std::stod
#include <fstream>   // std::ifstream, std::ofstream
#include <sstream>   // std::stringstream

#include <limits>    // std::numeric_limits (only used for testing)
#include <cmath>     // std::abs (only used for testing)

#if !defined(VERBOSE)
#define VERBOSE 1
#endif

/*
 * BOF - Functions required for calling grid
 */

double grid_virt(int seed, double s, double t)
{
    bool verbose;
    verbose = VERBOSE;

    // Use input seed to determine which FIFOs to use,
    // e.g. seed = 0 => FIFOs: pyInputPipe-0000, pyOutputPipe-0000
    std::stringstream fill;
    std::string pyin;
    std::string pyout;
    fill << std::setfill('0') << std::setw(4) << seed;
    pyin = "pyInputPipe-" + fill.str();
    pyout = "pyOutputPipe-" + fill.str();
    if(verbose)
    {
        std::cout << "Using FIFOs:" << std::endl;
        std::cout << pyin << std::endl;
        std::cout << pyout << std::endl;
        std::cout << "Input to grid_virt:" << std::endl;
        std::cout << s << std::endl;
        std::cout << t << std::endl;
    }

    // Build argument to be passed to python
    std::stringstream arg;
    arg << std::scientific << std::setprecision(40) <<  s << "," << t;
    if(verbose)
    {
        std::cout << "Will send the following string to python:" << std::endl;
        std::cout << arg.str() << std::endl;
    }

    // Send input to python script
    std::ofstream pyinstream(pyin.c_str(), std::ofstream::out);
    pyinstream << arg.str();
    pyinstream.close();

    // Receive result from python script
    std::ifstream pyoutstream(pyout.c_str(), std::ifstream::in);
    std::string res_string;
    res_string.reserve(500);
    res_string.assign((std::istreambuf_iterator<char>(pyoutstream)),std::istreambuf_iterator<char>());

    // Parse result of python grid
    if(verbose)
    {
        std::cout << "Got the following string from python:" << std::endl;
        std::cout << res_string << std::endl;
    }
    double res = std::stod(res_string);
    if(verbose)
    {
        std::cout << "Output of grid_virt:" << std::endl;
        std::cout << res << std::endl;
    }

    return res;
}

void kill_python(int seed)
{
    // Use input seed to determine which FIFOs to use,
    // e.g. seed = 0 => FIFOs: pyInputPipe-0000, pyOutputPipe-0000
    std::stringstream fill;
    std::string pyin;
    fill << std::setfill('0') << std::setw(4) << seed;
    pyin = "pyInputPipe-" + fill.str();

    std::ofstream pyinstream(pyin.c_str(), std::ofstream::out);
    pyinstream << "exit";
    pyinstream.close();
}

/*
 * EOF - Functions required for calling grid
 */

bool equal(double a, double b)
{
    if( std::abs(a-b) > std::numeric_limits<double>::epsilon() )
        return false;
    return true;
}

bool test_point(int seed, double s, double t, double expected)
{
    bool verbose;
    verbose = VERBOSE;

    double res;
    if (verbose)
    {
        std::cout << "Sending: " << s << " " << t << std::endl;
    }
    res = grid_virt(seed,s,t);
    if (verbose)
    {
        std::cout << "Received: " << res << std::endl;
    }
    if ( !equal(res,expected) )
    {
        std::cout << "Expected: " << expected << ", Got: " << res << std::endl;
        std::cout << "TESTS FAILED" << std::endl;
        return false;
    }
    return true;
}

int main()
{
    double expected;
    int seed;
    double s;
    double t;
    bool success;

    // Must match the seed used to launch grid.py
    seed = 0;

    // Test point 1
    s = 250000.e0;
    t = -50000.e0;
    expected = 3.3875286683507290375755305333882461127359e-04; // Expected result from Virt_EFT.grid
    if (!test_point(seed,s,t,expected))
        return 1;

    // Test point 2
    s = 66887.6e0;
    t = -9407.43e0;
    expected = 7.6844534557689569490555349384752759078765e-07;
    if (!test_point(seed,s,t,expected))
        return 1;

    // Test point 3
    s = 83055.9e0;
    t = -11438.e0;
    expected = 5.6368756265788111468190146879919666389469e-06;
    if (!test_point(seed,s,t,expected))
        return 1;

    // Test point 4
    s = 414983.e0;
    t = -59786.9e0;
    expected = 1.0669994178547940674728344845334504498169e-03;
    if (!test_point(seed,s,t,expected))
        return 1;

    // Test point 5
    s = 2.56513e6;
    t = -482321.e0;
    expected = 3.9170655780375776555679578905255766585469e-02;
    if (!test_point(seed,s,t,expected))
        return 1;

    // Evaluate point repeatedly (can detect synchronization issues with the grid)
    for(int i=0; i<10000; i++)
    {
        if(!test_point(seed,s,t,expected))
            return 1;
    }

    // Tell python program to exit
    kill_python(seed);

    std::cout << std::endl;
    std::cout << "TESTS PASSED" << std::endl;
    std::cout << std::endl;
    
    return 0;
}
