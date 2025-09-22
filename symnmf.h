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

/* Matrix memory management */
matrix alloc_matrix(int rows, int cols);
void free_matrix(matrix mat);

/* File parsing (needed by main program) */
matrix parse_file(const char *filename, int *n, int *d);

/* Main program helpers */
matrix execute_goal(const char *goal, matrix datapoints, int n, int d);
void print_matrix(matrix mat, int rows, int cols);

/* Core SymNMF algorithms - Public API */
matrix sym_c(matrix datapoints, int n, int d);
matrix ddg_c(matrix datapoints, int n, int d);
matrix norm_c(matrix datapoints, int n, int d);
matrix symnmf_c(matrix H, matrix W, int n, int k);

#endif /* SYMNMF_H */
