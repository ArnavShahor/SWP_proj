#define PY_SSIZE_T_CLEAN
#include <Python.h>
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

static matrix pymat_to_matrix(PyObject *py_matrix, int *n, int *d)
{
    *n = PyObject_Length(py_matrix);
    if (*n < 0)
        return NULL;

    PyObject *first_row = PyList_GetItem(py_matrix, 0);
    if (first_row == NULL)
        return NULL;

    *d = PyObject_Length(first_row);
    if (*d < 0)
        return NULL;

    matrix mat = alloc_matrix(*n, *d);
    if (mat == NULL)
        return NULL;

    if (fill_matrix_from_python(mat, py_matrix, *n, *d) != 0)
    {
        free_matrix(mat);
        return NULL;
    }

    return mat;
}

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

static PyObject *symnmf_wrapper(PyObject *self, PyObject *args)
{
    PyObject *py_H, *py_W;

    if (!PyArg_ParseTuple(args, "OO", &py_H, &py_W))
        return NULL;

    int n, k;
    matrix H = pymat_to_matrix(py_H, &n, &k);
    if (!H)
        return NULL;

    matrix W = pymat_to_matrix(py_W, &n, &k);
    if (!W)
    {
        free_matrix(H);
        return NULL;
    }

    matrix result = symnmf_func(H, W, n, k);
    free_matrix(H);
    free_matrix(W);

    if (!result)
        return NULL;

    PyObject *py_result = matrix_to_pymat(result, n, k);
    free_matrix(result);

    return py_result;
}

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

static PyObject *sym_wrapper(PyObject *self, PyObject *args)
{
    return unified_matrix_wrapper(args, sym_func);
}

static PyObject *ddg_wrapper(PyObject *self, PyObject *args)
{
    return unified_matrix_wrapper(args, ddg_func);
}

static PyObject *norm_wrapper(PyObject *self, PyObject *args)
{
    return unified_matrix_wrapper(args, norm_func);
}