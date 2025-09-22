#include <math.h>
#include <stdio.h>
#include <string.h>
#include "symnmf.h"

/**
 * Allocates memory for a 2D matrix with specified dimnomsions.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return A pointer to the allocated matrix, or NULL if allocation fails.
 */
matrix alloc_matrix(int rows, int cols)
{
    int i;
    matrix mat;
    double *data;

    mat = (matrix)malloc(rows * sizeof(double *));
    if (!mat)
        return NULL;

    data = (double *)calloc(rows * cols, sizeof(double));
    if (!data)
    {
        free(mat);
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        mat[i] = data + i * cols;
    }

    return mat;
}

/**
 * Frees the memory allocated for a matrix.
 *
 * @param mat The matrix to free.
 *
 * @return void
 */
void free_matrix(matrix mat)
{
    if (mat)
    {
        free(mat[0]);
        free(mat);
    }
}

/**
 * Counts the number of dimensions (columns) in the first row of a CSV file.
 *
 * @param filename The path to the CSV file.
 *
 * @return The number of dimensions, or -1 if file cannot be opened.
 */
int count_dimensions(const char *filename)
{
    FILE *file;
    int count, found_digit, c;

    file = fopen(filename, "r");
    if (!file)
        return -1;

    count = 0;
    found_digit = 0;
    while ((c = fgetc(file)) != EOF)
    {
        if (c == ',' || c == '\n')
        {
            if (found_digit)
            {
                count++;
                found_digit = 0;
            }
            if (c == '\n')
                break;
        }
        else if ((c >= '0' && c <= '9') || c == '.' || c == '-')
            found_digit = 1;
    }
    if (found_digit)
        count++;

    fclose(file);
    return count;
}

/**
 * Counts the number of lines in a file.
 *
 * @param filename The path to the file.
 *
 * @return The number of lines, or -1 if file cannot be opened.
 */
int count_lines(const char *filename)
{
    FILE *file;
    int count;
    int c;

    file = fopen(filename, "r");
    if (!file)
        return -1;

    count = 0;

    while ((c = fgetc(file)) != EOF)
        if (c == '\n')
            count++;

    fclose(file);
    return count;
}

/**
 * Reads matrix data from an open file.
 *
 * @param file Open file pointer.
 * @param result Pre-allocated matrix to fill.
 * @param n Number of rows.
 * @param d Number of columns.
 *
 * @return 1 on success, 0 on failure.
 */
int read_matrix_data(FILE *file, matrix result, int n, int d)
{
    int i, j;
    
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            double value;
            char delim;
            
            if (fscanf(file, "%lf%c", &value, &delim) != 2 || 
                !(delim == ',' || delim == '\n'))
                return 0;
            
            result[i][j] = value;
        }
    }
    return 1;
}

/**
 * Parses a CSV file and creates a matrix from its contents.
 *
 * @param filename The path to the CSV file.
 * @param n Pointer to store the number of rows.
 * @param d Pointer to store the number of columns.
 *
 * @return A matrix containing the file data, or NULL if parsing fails.
 */
matrix parse_file(const char *filename, int *n, int *d)
{
    matrix result;
    FILE *file;

    *d = count_dimensions(filename);
    *n = count_lines(filename);
    if (*d <= 0 || *n <= 0)
        return NULL;

    result = alloc_matrix(*n, *d);
    if (!result)
        return NULL;

    file = fopen(filename, "r");
    if (!file)
    {
        free_matrix(result);
        return NULL;
    }

    if (!read_matrix_data(file, result, *n, *d))
    {
        fclose(file);
        free_matrix(result);
        return NULL;
    }
    fclose(file);
    return result;
}

/**
 * Calculates the squared Euclidean distance between two points.
 *
 * @param p1 The first point vector.
 * @param p2 The second point vector.
 * @param d The dimensionality of the points.
 *
 * @return The squared Euclidean distance.
 */
double euclidean_distance_squared(double *p1, double *p2, int d)
{
    double sum, diff;
    int i;

    sum = 0.0;
    for (i = 0; i < d; i++)
    {
        diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * Computes the similarity matrix A from datapoints using Gaussian kernel.
 *
 * @param datapoints The input data matrix.
 * @param n The number of datapoints.
 * @param d The dimensionality of each datapoint.
 *
 * @return The similarity matrix A, or NULL if allocation fails.
 */
matrix sym_c(matrix datapoints, int n, int d)
{
    matrix A;
    int i, j;
    double dist_sq, similarity;

    A = alloc_matrix(n, n);
    if (!A)
        return NULL;

    for (i = 0; i < n; i++)
    {
        A[i][i] = 0.0;
        for (j = i + 1; j < n; j++)
        {
            dist_sq = euclidean_distance_squared(datapoints[i], datapoints[j], d);
            similarity = exp(-dist_sq / 2.0);
            A[i][j] = A[j][i] = similarity;
        }
    }

    return A;
}

/**
 * Calculates the degree of each vertex by summing the row values.
 *
 * @param A The similarity matrix.
 * @param n The size of the matrix.
 *
 * @return An array containing the degrees, or NULL if allocation fails.
 */
double *calculate_degrees_array(matrix A, int n)
{
    double *degrees;
    int i, j;

    degrees = (double *)malloc(n * sizeof(double));
    if (!degrees)
        return NULL;

    for (i = 0; i < n; i++)
    {
        degrees[i] = 0.0;
        for (j = 0; j < n; j++)
        {
            degrees[i] += A[i][j];
        }
    }

    return degrees;
}

/**
 * Creates a diagonal matrix from a degrees array.
 *
 * @param degrees Array of degree values.
 * @param n The size of the matrix.
 *
 * @return A diagonal matrix with degrees on the diagonal, or NULL if allocation fails.
 */
matrix create_diagonal_matrix(double *degrees, int n)
{
    matrix D;
    int i;

    D = alloc_matrix(n, n);
    if (!D)
        return NULL;

    for (i = 0; i < n; i++)
    {
        D[i][i] = degrees[i];
    }

    return D;
}

/**
 * Computes the diagonal degree matrix D from datapoints.
 *
 * @param datapoints The input data matrix.
 * @param n The number of datapoints.
 * @param d The dimensionality of each datapoint.
 *
 * @return The diagonal degree matrix D, or NULL if computation fails.
 */
matrix ddg_c(matrix datapoints, int n, int d)
{
    matrix A;
    double *degrees;
    matrix D;

    A = sym_c(datapoints, n, d);
    if (!A)
        return NULL;

    degrees = calculate_degrees_array(A, n);
    if (!degrees)
    {
        free_matrix(A);
        return NULL;
    }

    D = create_diagonal_matrix(degrees, n);

    free_matrix(A);
    free(degrees);
    return D;
}

/**
 * Normalizes degrees by raising each to the power of -0.5.
 *
 * @param degrees Array of degree values to normalize.
 * @param n The number of degrees.
 *
 * @return void
 */
void normalize_degrees(double *degrees, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        degrees[i] = pow(degrees[i], -0.5);
    }
}

/**
 * Creates a normalized similarity matrix from similarity matrix and normalized degrees.
 *
 * @param A The similarity matrix.
 * @param degrees Array of normalized degree values.
 * @param n The size of the matrix.
 *
 * @return The normalized similarity matrix W, or NULL if allocation fails.
 */
matrix create_normalized_matrix(matrix A, double *degrees, int n)
{
    matrix W;
    int i, j;

    W = alloc_matrix(n, n);
    if (!W)
        return NULL;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            W[i][j] = A[i][j] * degrees[i] * degrees[j];
        }
    }

    return W;
}

/**
 * Computes the normalized similarity matrix W from datapoints.
 *
 * @param datapoints The input data matrix.
 * @param n The number of datapoints.
 * @param d The dimensionality of each datapoint.
 *
 * @return The normalized similarity matrix W, or NULL if computation fails.
 */
matrix norm_c(matrix datapoints, int n, int d)
{
    matrix A;
    double *degrees;
    matrix W;

    A = sym_c(datapoints, n, d);
    if (!A)
        return NULL;

    degrees = calculate_degrees_array(A, n);
    if (!degrees)
    {
        free_matrix(A);
        return NULL;
    }

    normalize_degrees(degrees, n);
    W = create_normalized_matrix(A, degrees, n);

    free_matrix(A);
    free(degrees);
    return W;
}

/**
 * Multiplies two matrices.
 *
 * @param A The first matrix.
 * @param B The second matrix.
 * @param n1 Number of rows in matrix A.
 * @param m1 Number of columns in matrix A (must equal rows in B).
 * @param m2 Number of columns in matrix B.
 *
 * @return The product matrix C = A * B, or NULL if allocation fails.
 */
matrix matrix_multiply(matrix A, matrix B, int n1, int m1, int m2)
{
    matrix C;
    int i, j, k;

    C = alloc_matrix(n1, m2);
    if (!C)
        return NULL;

    for (i = 0; i < n1; i++)
    {
        for (j = 0; j < m2; j++)
        {
            for (k = 0; k < m1; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

/**
 * Transposes a matrix.
 *
 * @param A The matrix to transpose.
 * @param rows The number of rows in matrix A.
 * @param cols The number of columns in matrix A.
 *
 * @return The transposed matrix AT, or NULL if allocation fails.
 */
matrix transpose_matrix(matrix A, int rows, int cols)
{
    matrix AT;
    int i, j;

    AT = alloc_matrix(cols, rows);
    if (!AT)
        return NULL;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            AT[j][i] = A[i][j];
        }
    }

    return AT;
}

/**
 * Calculates the squared F norm of the difference between two matrices.
 *
 * @param A The first matrix.
 * @param B The second matrix.
 * @param rows The number of rows in both matrices.
 * @param cols The number of columns in both matrices.
 *
 * @return The squared F norm ||A - B||².
 */
double F_norm_squared(matrix A, matrix B, int rows, int cols)
{
    double sum, diff;
    int i, j;

    sum = 0.0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/**
 * Copies the contents of one matrix to another.
 *
 * @param src The source matrix to copy from.
 * @param dst The destination matrix to copy to.
 * @param rows The number of rows to copy.
 * @param cols The number of columns to copy.
 *
 * @return void
 */
void copy_matrix(matrix src, matrix dst, int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            dst[i][j] = src[i][j];
        }
    }
}

/**
 * Computes the denominator matrix for H update iteration (H * H^T * H).
 *
 * @param H The current H matrix.
 * @param n The number of rows in H.
 * @param k The number of columns in H.
 *
 * @return The denominator matrix H * H^T * H, or NULL if computation fails.
 */
matrix compute_H_denominator(matrix H, int n, int k)
{
    matrix Ht;
    matrix H_Ht;
    matrix H_Ht_H;
    int i, j;

    Ht = transpose_matrix(H, n, k);
    if (!Ht)
        return NULL;

    H_Ht = matrix_multiply(H, Ht, n, k, n);
    if (!H_Ht)
    {
        free_matrix(Ht);
        return NULL;
    }

    H_Ht_H = matrix_multiply(H_Ht, H, n, n, k);

    /* Add epsilon squared only to zero entries to prevent division by zero */
    for (i = 0; i < n; i++)
        for (j = 0; j < k; j++)
            if (H_Ht_H[i][j] == 0.0)
                H_Ht_H[i][j] = EPSILON * EPSILON;

    free_matrix(Ht);
    free_matrix(H_Ht);
    return H_Ht_H;
}

/**
 * Performs one iteration of H matrix update using multiplicative update rule.
 *
 * @param H The current H matrix.
 * @param W The normalized similarity matrix.
 * @param n The number of rows in H.
 * @param k The number of columns in H.
 *
 * @return The updated H matrix for next iteration, or NULL if computation fails.
 */
matrix update_H_iteration(matrix H, matrix W, int n, int k)
{
    matrix numerator, denominator, new_H;
    int i, j;
    
    numerator = matrix_multiply(W, H, n, n, k);
    if (!numerator)
        return NULL;
    denominator = compute_H_denominator(H, n, k);
    if (!denominator)
    {
        free_matrix(numerator);
        return NULL;
    }
    new_H = alloc_matrix(n, k);
    if (!new_H)
    {
        free_matrix(numerator);
        free_matrix(denominator);
        return NULL;
    }

    for (i = 0; i < n; i++)
        for (j = 0; j < k; j++)
            new_H[i][j] = H[i][j] * (1 - BETA + BETA * (numerator[i][j] / denominator[i][j]));

    free_matrix(numerator);
    free_matrix(denominator);
    return new_H;
}

/**
 * Performs the SymNMF algorithm to factorize W into H * H^T.
 *
 * @param H The initial H matrix.
 * @param W The normalized similarity matrix.
 * @param n The number of datapoints.
 * @param k The number of clusters.
 *
 * @return The final H matrix after convergence, or NULL if computation fails.
 */
matrix symnmf_c(matrix H, matrix W, int n, int k)
{
    matrix current_H, new_H;
    int iter;
    double norm;

    current_H = H;

    for (iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        new_H = update_H_iteration(current_H, W, n, k);
        if (!new_H)
        {
            if (current_H != H)
                free_matrix(current_H);
            return NULL;
        }
        norm = F_norm_squared(new_H, current_H, n, k);

        if (current_H != H)
            free_matrix(current_H);
        current_H = new_H;

        if (norm < EPSILON)
            break;
    }
    return current_H;
}

/**
 * Prints a matrix to stdout in CSV format.
 *
 * @param mat The matrix to print.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return void
 */
void print_matrix(matrix mat, int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%.4f", mat[i][j]);
            if (j < cols - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

/**
 * Executes the specified goal algorithm on datapoints.
 *
 * @param goal The algorithm to execute ("sym", "ddg", or "norm").
 * @param datapoints The input data matrix.
 * @param n Number of datapoints.
 * @param d Dimensionality of datapoints.
 *
 * @return The result matrix, or NULL if goal is invalid or computation fails.
 */
matrix execute_goal(const char *goal, matrix datapoints, int n, int d)
{
    if (strcmp(goal, "sym") == 0)
        return sym_c(datapoints, n, d);
    else if (strcmp(goal, "ddg") == 0)
        return ddg_c(datapoints, n, d);
    else if (strcmp(goal, "norm") == 0)
        return norm_c(datapoints, n, d);
    else
        return NULL;
}

int main(int argc, char *argv[])
{
    char *goal;
    char *filename;
    int n, d;
    matrix datapoints;
    matrix result;

    if (argc != 3)
    {
        printf("%s\n", GENERIC_ERROR_MSG);
        return ERROR;
    }

    goal = argv[1];
    filename = argv[2];

    datapoints = parse_file(filename, &n, &d);
    if (!datapoints)
    {
        printf("%s\n", GENERIC_ERROR_MSG);
        return ERROR;
    }

    result = execute_goal(goal, datapoints, n, d);
    if (!result)
    {
        printf("%s\n", GENERIC_ERROR_MSG);
        free_matrix(datapoints);
        return ERROR;
    }

    print_matrix(result, n, n);
    free_matrix(datapoints);
    free_matrix(result);
    return SUCCESS;
}
