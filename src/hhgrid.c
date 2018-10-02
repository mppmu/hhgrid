
#include "hhgrid.h"

#include <python2.7/Python.h>
#include <stdlib.h> // setenv, strdup, strtok, NULL
#include <assert.h> // assert
#include <unistd.h> // access

// BOF - Helper Functions (mostly for Fortran)
void python_initialize()
{
    Py_Initialize();
};

void python_decref(PyObject* grid)
{
    Py_DECREF(grid);
};

void python_finalize()
{
    Py_Finalize();
};

void python_printinfo()
{
    const char* programFullPath = Py_GetProgramFullPath();
    const char* getVersion = Py_GetVersion();
    const char* getPythonHome = Py_GetPythonHome();
    const char* getPath = Py_GetPath();

    printf("== Python Parameters ==\n");
    printf("Py_GetProgramFullPath: %s\n", programFullPath);
    printf("Py_GetVersion: %s\n", getVersion);
    printf("Py_GetPythonHome: %s\n", getPythonHome);
    printf("Py_GetPath: %s\n", getPath);
    printf("\n");
};
// EOF - Helper Functions (mostly for Fortran)

PyObject* grid_initialize(const char* grid_name)
{
    int search_paths = 1;
    char* delims = ":";
    char* path_sep = "/";
    char* grid_file_path;
    size_t len_path_sep = strlen(path_sep);
    size_t len_grid_name = strlen(grid_name);
    char* events_file_path;
    char* events_name = "events.cdf";
    size_t len_events_name = strlen(events_name);
    char* creategrid_file_path;
    char* creategrid_name = "creategrid.py";
    size_t len_creategrid_name = strlen(creategrid_name);
    char* pythonpath = strdup(getenv("PYTHONPATH"));
    char* result = strtok( pythonpath, delims );
    size_t len_result = 0;

    // Check if grid_name is accessible as-is
    if( access(grid_name, F_OK) != -1 && access(grid_name, R_OK) != -1 )
    {
        grid_file_path = (char*) malloc(len_grid_name + 1); // +1 for null terminator
        memcpy(grid_file_path, grid_name, len_grid_name);
        search_paths = 0;
        setenv("PYTHONPATH", ".", 1); // Set PYTHONPATH to look here
    }

    // Else search PYTHONPATH for grid_name
    while( search_paths == 1 && result != NULL )
    {
        len_result = strlen(result);
        grid_file_path = (char*) malloc (len_result + len_path_sep + len_grid_name + 1); // +1 for null terminator
        memcpy(grid_file_path, result, len_result);
        memcpy(grid_file_path + len_result, path_sep, len_path_sep);
        memcpy(grid_file_path + len_result + len_path_sep, grid_name, len_grid_name + 1); // +1 for null terminator
        printf("Searching for %s in: %s ", grid_name, grid_file_path);
        if( access(grid_file_path, F_OK) != -1 && access(grid_file_path, R_OK) != -1 )
        {
            printf("found\n");
            search_paths = 0;
        } else
        {
            printf("not found\n");
            result = strtok( NULL, delims );
            if(result == NULL)
            {
                printf("ERROR: Failed to find grid");
                exit(1);
            }
        }
    }

    // Now check for other required files
    // events.cdf
    events_file_path = (char*) malloc (len_result + len_path_sep + len_events_name + 1); // +1 for null terminator
    memcpy(events_file_path, result, len_result);
    memcpy(events_file_path + len_result, path_sep, len_path_sep);
    memcpy(events_file_path + len_result + len_path_sep, events_name, len_events_name + 1); // +1 for null terminator
    printf("Searching for %s in: %s ", events_name, events_file_path);
    if( access(events_file_path, F_OK) != -1 && access(events_file_path, R_OK) != -1 )
    {
        printf("found\n");
    } else {
        printf("not found\n");
        printf("ERROR: Failed to find events.cdf");
        exit(1);
    }

    // creategrid.py
    creategrid_file_path = (char*) malloc (len_result + len_path_sep + len_creategrid_name + 1); // +1 for null terminator
    memcpy(creategrid_file_path, result, len_result);
    memcpy(creategrid_file_path + len_result, path_sep, len_path_sep);
    memcpy(creategrid_file_path + len_result + len_path_sep, creategrid_name, len_creategrid_name + 1); // +1 for null terminator
    printf("Searching for %s in: %s ", creategrid_name, creategrid_file_path);
    if( access(creategrid_file_path, F_OK) != -1 && access(creategrid_file_path, R_OK) != -1 )
    {
        printf("found\n");
    } else {
        printf("not found\n");
        printf("ERROR: Failed to find creategrid.py");
        exit(1);
    }

    printf("PYTHONPATH: %s\n", getenv("PYTHONPATH"));
    printf("Grid Path: %s\n", grid_file_path);
    printf("Events file Path: %s\n", events_file_path);
    printf("CreateGrid file Path: %s\n", creategrid_file_path);

    PyObject* pModule = PyImport_ImportModule("creategrid");
    if(pModule == NULL)
    {
        PyErr_Print();
        printf("Failed to load creategrid.py: please check that you have numpy and scipy installed");
    }
    assert(pModule != NULL);

    PyObject* pClass = PyObject_GetAttrString(pModule, "CreateGrid");
    if(pClass == NULL)
    {
        PyErr_Print();
        printf("Failed to locate CreateGrid class: please check that you have the latest version of creategrid.py and Python 2.7.x");
    }
    assert(pClass != NULL);

    PyObject* pGridName = PyString_FromString(grid_file_path);
    if(pGridName == NULL)
    {
        PyErr_Print();
        printf("Failed to create Python string from grid_file_path: please check that grid_file_path is a valid string");
    }
    assert(pGridName != NULL);

    PyObject* pGridNameTuple = PyTuple_Pack(1,pGridName);
    if(pGridNameTuple == NULL)
    {
        PyErr_Print();
        printf("Failed to create Python tuple: please check that your Python version is 2.7.x");
    }
    assert(pGridNameTuple != NULL);

    PyObject* pInstance = PyInstance_New(pClass, pGridNameTuple, NULL);
    if(pInstance == NULL)
    {
        PyErr_Print();
        printf("Failed to create instance of CreateGrid: please check that you have the latest version of creategrid.py and Python 2.7.x");
    }
    assert(pInstance != NULL);

    // Cleanup
    free(pythonpath);
    free(grid_file_path);
    Py_DECREF(pModule);
    Py_DECREF(pClass);
    Py_DECREF(pGridName);
    Py_DECREF(pGridNameTuple);

    return pInstance;
};

double grid_virt(PyObject* grid, double s, double t)
{

    PyObject* pResult = PyObject_CallMethod(grid, "GetAmplitude", "(ff)", s, t);
    assert(pResult != NULL);

    double result = PyFloat_AsDouble(pResult);

    // Cleanup
    Py_DECREF(pResult);

    return result;
};
