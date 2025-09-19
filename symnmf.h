#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_ITERATIONS 300
#define EPSILON 1e-4
#define BETA 0.5

typedef double **matrix;

#define SUCCESS 0
#define ERROR 1
#define GENERIC_ERROR_MSG "An Error Has Occurred"

/* TODO should we filter header file and keep only module-related functions? */

/* Error handling */
void print_error(void);

/* Matrix memory management */
matrix alloc_matrix(int rows, int cols);
void free_matrix(matrix mat);

/* File I/O functions */
int count_dimensions(const char *filename);
int count_lines(const char *filename);
matrix parse_file(const char *filename, int *n, int *d);

/* Utility functions */
double euclidean_distance_squared(double *p1, double *p2, int d);
void copy_matrix(matrix src, matrix dst, int rows, int cols);
void print_matrix(matrix mat, int rows, int cols);

/* Matrix operations */
matrix matrix_multiply(matrix A, matrix B, int n1, int m1, int m2);
matrix transpose_matrix(matrix A, int rows, int cols);
double F_norm_squared(matrix A, matrix B, int rows, int cols);

/* Degree calculations */
double *calculate_degrees_array(matrix A, int n);
matrix create_diagonal_matrix(double *degrees, int n);
void normalize_degrees(double *degrees, int n);
matrix create_normalized_matrix(matrix A, double *degrees, int n);

/* Core SymNMF algorithms */
matrix sym_func(matrix datapoints, int n, int d);
matrix ddg_func(matrix datapoints, int n, int d);
matrix norm_func(matrix datapoints, int n, int d);

/* SymNMF iteration functions */
matrix compute_H_denominator(matrix H, int n, int k);
matrix update_H_iteration(matrix H, matrix W, int n, int k);
matrix symnmf_func(matrix H, matrix W, int n, int k);

#endif /* SYMNMF_H */
