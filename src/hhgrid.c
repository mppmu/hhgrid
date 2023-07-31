
#include "hhgrid.h"

#include <Python.h>
#include <wchar.h>  // wchar_t, wstrtok
#include <stdlib.h> // setenv, strdup, strtok, NULL
#include <assert.h> // assert
#include <unistd.h> // access, getcwd

// BOF - Helper Functions (mostly for Fortran)
void python_initialize()
{
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        printf("ERROR: Failed to initialize Python interpreter\n");
        exit(1);
    }
};

void python_decref(PyObject* grid)
{
    Py_XDECREF(grid);
};

void python_finalize()
{
    Py_Finalize();
};

void python_printinfo()
{
    const wchar_t* programFullPath = Py_GetProgramFullPath();
    const char* getVersion = Py_GetVersion();
    const wchar_t* getPythonHome = Py_GetPythonHome();
    const wchar_t* getPath = Py_GetPath();

    printf("== Python Parameters ==\n");
    printf("Py_GetProgramFullPath: %ls\n", programFullPath);
    printf("Py_GetVersion: %s\n", getVersion);
    printf("Py_GetPythonHome: %ls\n", getPythonHome);
    printf("Py_GetPath: %ls\n", getPath);
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
    char* pythonpath = strdup(getenv("PYTHONPATH"));
    char* result = strtok( pythonpath, delims );

    char* events_file_path;
    char* events_name = "events.cdf";
    size_t len_events_name = strlen(events_name);

    char* creategrid_file_path;
    char* creategrid_name = "creategrid.py";
    size_t len_creategrid_name = strlen(creategrid_name);
    
    // Check if grid_name is accessible as-is
    if( access(grid_name, F_OK) != -1 && access(grid_name, R_OK) != -1 )
    {
        printf("Looking for %s in current directory. Found\n", grid_name);
        grid_file_path = (char*) malloc(len_grid_name + 1); // +1 for null terminator
        memcpy(grid_file_path, grid_name, len_grid_name + 1);
        search_paths = 0;
        setenv("PYTHONPATH", ".", 1); // Set PYTHONPATH to look here
    }

    // Else search PYTHONPATH for grid_name
    while( search_paths == 1 && result != NULL )
    {
        size_t len_result = strlen(result);
        grid_file_path = (char*) malloc (len_result + len_path_sep + len_grid_name + 1); // +1 for null terminator
        memcpy(grid_file_path, result, len_result);
        memcpy(grid_file_path + len_result, path_sep, len_path_sep);
        memcpy(grid_file_path + len_result + len_path_sep, grid_name, len_grid_name + 1); // +1 for null terminator
        printf("Searching for %s in: %s ", grid_name, grid_file_path);
        if( access(grid_file_path, F_OK) != -1 && access(grid_file_path, R_OK) != -1 )
        {
            printf("found\n");
            search_paths = 0;
            
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
        } else
        {
            printf("not found\n");
            result = strtok( NULL, delims );
            if(result == NULL)
            {
                printf("ERROR: Failed to find grid\n");
                exit(1);
            }
        }
    }

    printf("PYTHONPATH: %s\n", getenv("PYTHONPATH"));
    printf("Grid Path: %s\n", grid_file_path);
    printf("Events file Path: %s\n", events_file_path);
    printf("CreateGrid file Path: %s\n", creategrid_file_path);

    PyObject* pModule = PyImport_ImportModule("creategrid");
    if(pModule == NULL)
    {
        PyErr_Print();
        printf("ERROR: Failed to load creategrid.py: please check that you have numpy and scipy installed\n");
    }
    assert(pModule != NULL);

    PyObject* pClass = PyObject_GetAttrString(pModule, "CreateGrid");
    if(pClass == NULL)
    {
        PyErr_Print();
        printf("ERROR: Failed to locate CreateGrid class: please check that you have the latest version of creategrid.py and Python 3.x\n");
    }
    assert(pClass != NULL);
    
    if(!PyCallable_Check(pClass))
    {
        PyErr_Print();
        printf("ERROR: CreateGrid is not callable: please check that you have the latest version of creategrid.py and Python 3.x\n");
    }

    PyObject* pGridName = PyUnicode_FromString(grid_file_path);
    if(pGridName == NULL)
    {
        PyErr_Print();
        printf("ERROR: Failed to create Python string from grid_file_path: please check that grid_file_path is a valid string\n");
    }
    assert(pGridName != NULL);

    PyObject* pGridNameTuple = PyTuple_Pack(1,pGridName);
    if(pGridNameTuple == NULL)
    {
        PyErr_Print();
        printf("ERROR: Failed to create Python tuple: please check that your Python version is 3.x\n");
    }
    assert(pGridNameTuple != NULL);

    PyObject* pInstance = PyObject_CallObject(pClass, pGridNameTuple);
    if(pInstance == NULL)
    {
        PyErr_Print();
        printf("ERROR: Failed to create instance of CreateGrid: please check that you have the latest version of creategrid.py and Python 3.x\n");
    }
    assert(pInstance != NULL);

    // Cleanup
    free(pythonpath);
    free(grid_file_path);
    free(events_file_path);
    free(creategrid_file_path);
    Py_XDECREF(pModule);
    Py_XDECREF(pClass);
    Py_XDECREF(pGridName);
    Py_XDECREF(pGridNameTuple);

    return pInstance;
};

double grid_virt(PyObject* grid, double s, double t)
{
    PyObject* pResult = PyObject_CallMethod(grid, "GetAmplitude", "(ff)", s, t);
    if(pResult == NULL)
    {
        PyErr_Print();
        printf("ERROR: Failed to compute result\n");
    }
    assert(pResult != NULL);

    double result = PyFloat_AsDouble(pResult);

    // Cleanup
    Py_XDECREF(pResult);

    return result;
};
