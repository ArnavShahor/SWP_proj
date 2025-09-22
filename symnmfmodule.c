#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include "symnmf.h"

static PyObject *symnmf_wrapper(PyObject *self, PyObject *args);
static PyObject *sym_wrapper(PyObject *self, PyObject *args);
static PyObject *ddg_wrapper(PyObject *self, PyObject *args);
static PyObject *norm_wrapper(PyObject *self, PyObject *args);

static PyMethodDef symnmfMethods[] = {
    {"symnmf", (PyCFunction)symnmf_wrapper, METH_VARARGS, PyDoc_STR("Perform SymNMF factorization")},
    {"sym", (PyCFunction)sym_wrapper, METH_VARARGS, PyDoc_STR("Compute similarity matrix")},
    {"ddg", (PyCFunction)ddg_wrapper, METH_VARARGS, PyDoc_STR("Compute diagonal degree matrix")},
    {"norm", (PyCFunction)norm_wrapper, METH_VARARGS, PyDoc_STR("Compute normalized similarity matrix")},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    symnmfMethods};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m)
        return NULL;
    return m;
}

/**
 * Fills a C matrix with data from a Python list of lists.
 *
 * @param mat The pre-allocated matrix to fill.
 * @param py_matrix The Python list of lists containing the data.
 * @param n The number of rows.
 * @param d The number of columns.
 *
 * @return 0 on success, -1 on failure.
 */
static int fill_matrix_from_python(matrix mat, PyObject *py_matrix, int n, int d)
{
    for (int i = 0; i < n; i++)
    {
        PyObject *row = PyList_GetItem(py_matrix, i);
        if (row == NULL)
            return -1;

        for (int j = 0; j < d; j++)
        {
            PyObject *coord = PyList_GetItem(row, j);
            if (!PyNumber_Check(coord))
                return -1;

            mat[i][j] = PyFloat_AsDouble(coord);
            if (PyErr_Occurred())
                return -1;
        }
    }
    return 0;
}

/**
 * Converts a Python list of lists to a C matrix.
 *
 * @param py_matrix The Python list of lists to convert.
 * @param n Pointer to store the number of rows.
 * @param d Pointer to store the number of columns.
 *
 * @return A pointer to the allocated matrix, or NULL if conversion fails.
 */
static matrix pymat_to_matrix(PyObject *py_matrix, int *n, int *d)
{
    printf("DEBUG MODULE: pymat_to_matrix called\n");
    
    if (!PyList_Check(py_matrix))
    {
        printf("DEBUG MODULE: Input is not a Python list\n");
        return NULL;
    }

    *n = PyObject_Length(py_matrix);
    if (*n < 0)
    {
        printf("DEBUG MODULE: Failed to get length of Python matrix\n");
        return NULL;
    }
    printf("DEBUG MODULE: Matrix has %d rows\n", *n);

    PyObject *first_row = PyList_GetItem(py_matrix, 0);
    if (first_row == NULL)
    {
        printf("DEBUG MODULE: Failed to get first row\n");
        return NULL;
    }

    if (!PyList_Check(first_row))
    {
        printf("DEBUG MODULE: First row is not a Python list\n");
        return NULL;
    }

    *d = PyObject_Length(first_row);
    if (*d < 0)
    {
        printf("DEBUG MODULE: Failed to get length of first row\n");
        return NULL;
    }
    printf("DEBUG MODULE: Matrix has %d columns\n", *d);

    matrix mat = alloc_matrix(*n, *d);
    if (mat == NULL)
    {
        printf("DEBUG MODULE: Failed to allocate matrix\n");
        return NULL;
    }
    printf("DEBUG MODULE: Matrix allocated successfully\n");

    if (fill_matrix_from_python(mat, py_matrix, *n, *d) != 0)
    {
        printf("DEBUG MODULE: Failed to fill matrix from Python data\n");
        free_matrix(mat);
        return NULL;
    }

    printf("DEBUG MODULE: Matrix filled successfully\n");
    return mat;
}

/**
 * Converts a C matrix to a Python list of lists.
 *
 * @param mat The C matrix to convert.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return A Python list of lists, or NULL if conversion fails.
 */
static PyObject *matrix_to_pymat(matrix mat, int rows, int cols)
{
    PyObject *py_matrix = PyList_New(rows);
    if (!py_matrix)
        return NULL;

    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_New(cols);
        if (!row)
            return NULL;

        for (int j = 0; j < cols; j++)
        {
            PyObject *coord = PyFloat_FromDouble(mat[i][j]);
            if (!coord)
                return NULL;
            if (PyList_SetItem(row, j, coord) < 0)
                return NULL;
        }
        if (PyList_SetItem(py_matrix, i, row) < 0)
            return NULL;
    }

    return py_matrix;
}

/**
 * Python wrapper function for the SymNMF algorithm.
 *
 * @param self The module object (unused).
 * @param args Python tuple containing H and W matrices.
 *
 * @return Python list of lists representing the final H matrix, or NULL on error.
 */
static PyObject *symnmf_wrapper(PyObject *self, PyObject *args)
{
    PyObject *py_H, *py_W;

    printf("DEBUG MODULE: symnmf_wrapper called\n");

    if (!PyArg_ParseTuple(args, "OO", &py_H, &py_W))
    {
        printf("DEBUG MODULE: Failed to parse arguments\n");
        return NULL;
    }

    printf("DEBUG MODULE: Arguments parsed successfully\n");

    int n, k;
    matrix H = pymat_to_matrix(py_H, &n, &k);
    if (!H)
    {
        printf("DEBUG MODULE: Failed to convert H matrix from Python\n");
        return NULL;
    }

    printf("DEBUG MODULE: H matrix converted - size %dx%d\n", n, k);

    matrix W = pymat_to_matrix(py_W, &n, &n);
    if (!W)
    {
        printf("DEBUG MODULE: Failed to convert W matrix from Python\n");
        free_matrix(H);
        return NULL;
    }

    printf("DEBUG MODULE: W matrix converted - size %dx%d\n", n, n);
    printf("DEBUG MODULE: Calling symnmf_c function\n");

    matrix result = symnmf_c(H, W, n, k);
    free_matrix(H);
    free_matrix(W);

    if (!result)
    {
        printf("DEBUG MODULE: symnmf_c returned NULL\n");
        return NULL;
    }

    printf("DEBUG MODULE: symnmf_c completed, converting result to Python\n");

    PyObject *py_result = matrix_to_pymat(result, n, k);
    free_matrix(result);

    if (!py_result)
    {
        printf("DEBUG MODULE: Failed to convert result to Python format\n");
        return NULL;
    }

    printf("DEBUG MODULE: symnmf_wrapper completed successfully\n");
    return py_result;
}

/**
 * Unified wrapper function for matrix computation functions (sym, ddg, norm).
 *
 * @param args Python tuple containing the datapoints matrix.
 * @param func Pointer to the C function to call (sym_c, ddg_c, or norm_c).
 *
 * @return Python list of lists representing the computed matrix, or NULL on error.
 */
static PyObject *unified_matrix_wrapper(PyObject *args, matrix (*func)(matrix, int, int))
{
    int n, d;
    PyObject *py_datapoints;

    if (!PyArg_ParseTuple(args, "O", &py_datapoints))
        return NULL;

    matrix datapoints = pymat_to_matrix(py_datapoints, &n, &d);
    if (!datapoints)
        return NULL;

    matrix result = func(datapoints, n, d);
    free_matrix(datapoints);

    if (!result)
        return NULL;

    PyObject *py_result = matrix_to_pymat(result, n, n);
    free_matrix(result);
    if (!py_result)
        return NULL;

    return py_result;
}

/**
 * Python wrapper function for similarity matrix computation.
 *
 * @param self The module object (unused).
 * @param args Python tuple containing the datapoints matrix.
 *
 * @return Python list of lists representing the similarity matrix, or NULL on error.
 */
static PyObject *sym_wrapper(PyObject *self, PyObject *args)
{
    return unified_matrix_wrapper(args, sym_c);
}

/**
 * Python wrapper function for diagonal degree matrix computation.
 *
 * @param self The module object (unused).
 * @param args Python tuple containing the datapoints matrix.
 *
 * @return Python list of lists representing the diagonal degree matrix, or NULL on error.
 */
static PyObject *ddg_wrapper(PyObject *self, PyObject *args)
{
    return unified_matrix_wrapper(args, ddg_c);
}

/**
 * Python wrapper function for normalized similarity matrix computation.
 *
 * @param self The module object (unused).
 * @param args Python tuple containing the datapoints matrix.
 *
 * @return Python list of lists representing the normalized similarity matrix, or NULL on error.
 */
static PyObject *norm_wrapper(PyObject *self, PyObject *args)
{
    return unified_matrix_wrapper(args, norm_c);
}