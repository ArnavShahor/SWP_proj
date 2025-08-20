#include <math.h>
#include <string.h>
#include "symnmf.h"

/**
 * Prints a generic error message to stdout.
 *
 * @return void
 */
void print_error()
{
    printf("%s\n", GENERIC_ERROR_MSG);
}

/**
 * Allocates memory for a 2D matrix with specified dimensions.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 *
 * @return A pointer to the allocated matrix, or NULL if allocation fails.
 */
matrix alloc_matrix(int rows, int cols)
{
    matrix mat = (matrix)malloc(rows * sizeof(double *));
    if (!mat)
        return NULL;

    double *data = (double *)calloc(rows * cols, sizeof(double));
    if (!data)
    {
        free(mat);
        return NULL;
    }

    for (int i = 0; i < rows; i++)
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
    FILE *file = fopen(filename, "r");
    if (!file)
        return -1;

    int count = 0;
    int c;
    int found_digit = 0;

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
        {
            found_digit = 1;
        }
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
    // TODO check that we don't miscount the last linebreaks as a point
    FILE *file = fopen(filename, "r");
    if (!file)
        return -1;

    int count = 0;
    int c;

    while ((c = fgetc(file)) != EOF)
        if (c == '\n')
            count++;

    fclose(file);
    return count;
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
    *d = count_dimensions(filename);
    *n = count_lines(filename);

    if (*d <= 0 || *n <= 0)
        return NULL;

    matrix result = alloc_matrix(*n, *d);
    if (!result)
        return NULL;

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        free_matrix(result);
        return NULL;
    }

    for (int i = 0; i < *n; i++)
    {
        for (int j = 0; j < *d; j++)
        {
            double value;
            char delim;

            if (fscanf(file, "%lf%c", &value, &delim) != 2 || !(delim == ',' || delim == '\n'))
            {
                fclose(file);
                free_matrix(result);
                return NULL;
            }

            result[i][j] = value;
        }
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
    double sum = 0.0;
    for (int i = 0; i < d; i++)
    {
        double diff = p1[i] - p2[i];
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
matrix sym_func(matrix datapoints, int n, int d)
{
    matrix A = alloc_matrix(n, n);
    if (!A)
        return NULL;

    for (int i = 0; i < n; i++)
    {
        A[i][i] = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            double dist_sq = euclidean_distance_squared(datapoints[i], datapoints[j], d);
            double similarity = exp(-dist_sq / 2.0);
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
    double *degrees = (double *)malloc(n * sizeof(double));
    if (!degrees)
        return NULL;

    for (int i = 0; i < n; i++)
    {
        degrees[i] = 0.0;
        for (int j = 0; j < n; j++)
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
    matrix D = alloc_matrix(n, n);
    if (!D)
        return NULL;

    for (int i = 0; i < n; i++)
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
matrix ddg_func(matrix datapoints, int n, int d)
{
    matrix A = sym_func(datapoints, n, d);
    if (!A)
        return NULL;

    double *degrees = calculate_degrees_array(A, n);
    if (!degrees)
    {
        free_matrix(A);
        return NULL;
    }

    matrix D = create_diagonal_matrix(degrees, n);

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
    for (int i = 0; i < n; i++)
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
    matrix W = alloc_matrix(n, n);
    if (!W)
        return NULL;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
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
matrix norm_func(matrix datapoints, int n, int d)
{
    matrix A = sym_func(datapoints, n, d);
    if (!A)
        return NULL;

    double *degrees = calculate_degrees_array(A, n);
    if (!degrees)
    {
        free_matrix(A);
        return NULL;
    }

    normalize_degrees(degrees, n);
    matrix W = create_normalized_matrix(A, degrees, n);

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
    matrix C = alloc_matrix(n1, m2);
    if (!C)
        return NULL;

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m2; j++)
        {
            for (int k = 0; k < m1; k++)
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
    matrix AT = alloc_matrix(cols, rows);
    if (!AT)
        return NULL;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
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
    double sum = 0.0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double diff = A[i][j] - B[i][j];
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
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
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
    // TODO should we add epsilon to all of the entries?

    matrix Ht = transpose_matrix(H, n, k);
    if (!Ht)
        return NULL;

    matrix H_Ht = matrix_multiply(H, Ht, n, k, n);
    if (!H_Ht)
    {
        free_matrix(Ht);
        return NULL;
    }

    matrix H_Ht_H = matrix_multiply(H_Ht, H, n, n, k);

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
    matrix numerator = matrix_multiply(W, H, n, n, k);
    if (!numerator)
        return NULL;

    matrix denominator = compute_H_denominator(H, n, k);
    if (!denominator)
    {
        free_matrix(numerator);
        return NULL;
    }

    matrix new_H = alloc_matrix(n, k);
    if (!new_H)
    {
        free_matrix(numerator);
        free_matrix(denominator);
        return NULL;
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
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
matrix symnmf_func(matrix H, matrix W, int n, int k)
{
    // TODO use origin H instead of copy
    matrix current_H = alloc_matrix(n, k);
    if (!current_H)
        return NULL;

    copy_matrix(H, current_H, n, k);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        matrix new_H = update_H_iteration(current_H, W, n, k);
        if (!new_H)
        {
            free_matrix(current_H);
            return NULL;
        }

        double norm = F_norm_squared(new_H, current_H, n, k);
        free_matrix(current_H);
        current_H = new_H;

        if (norm < EPSILON)
        {
            break;
        }
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
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
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

int main(int argc, char *argv[])
{
    // TODO delete tests from main
    if (argc == 2 && strcmp(argv[1], "test") == 0)
    {
        // Test mode for file parsing
        const char *test_file = "tests/input_1.txt";
        int n, d;

        printf("Testing file parsing with %s\n", test_file);

        d = count_dimensions(test_file);
        printf("Dimensions: %d\n", d);

        n = count_lines(test_file);
        printf("Lines: %d\n", n);

        matrix data = parse_file(test_file, &n, &d);
        if (!data)
        {
            print_error();
            return ERROR;
        }

        printf("Successfully parsed %dx%d matrix\n", n, d);
        printf("First 5 rows:\n");
        for (int i = 0; i < 5 && i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                printf("%.4f", data[i][j]);
                if (j < d - 1)
                    printf(",");
            }
            printf("\n");
        }

        free_matrix(data);
        printf("Test completed successfully!\n");
        return SUCCESS;
    }

    if (argc == 2 && strcmp(argv[1], "small") == 0)
    {
        // Test with specific small matrix A
        printf("Testing with hardcoded 6x6 matrix A:\n");
        printf("A = [0 1 0 0 1 0]\n");
        printf("    [1 0 1 0 1 0]\n");
        printf("    [0 1 0 1 0 0]\n");
        printf("    [0 0 1 0 1 1]\n");
        printf("    [1 1 0 1 0 0]\n");
        printf("    [0 0 0 1 0 0]\n\n");

        int n = 6;
        matrix A = alloc_matrix(n, n);
        if (!A)
        {
            print_error();
            return ERROR;
        }

        // Initialize the matrix A with your values
        int A_values[6][6] = {
            {0, 1, 0, 0, 1, 0},
            {1, 0, 1, 0, 1, 0},
            {0, 1, 0, 1, 0, 0},
            {0, 0, 1, 0, 1, 1},
            {1, 1, 0, 1, 0, 0},
            {0, 0, 0, 1, 0, 0}};

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i][j] = (double)A_values[i][j];
            }
        }

        printf("=== Matrix A ===\n");
        print_matrix(A, n, n);
        printf("\n");

        // Test degree calculation
        printf("=== Degrees Calculation (section 1.2) ===\n");
        double *degrees = calculate_degrees_array(A, n);
        if (!degrees)
        {
            print_error();
            free_matrix(A);
            return ERROR;
        }

        printf("Degrees for each node: ");
        for (int i = 0; i < n; i++)
        {
            printf("%.0f ", degrees[i]);
        }
        printf("\n");

        // Create diagonal degree matrix D
        matrix D = create_diagonal_matrix(degrees, n);
        if (!D)
        {
            print_error();
            free_matrix(A);
            free(degrees);
            return ERROR;
        }

        printf("\nDiagonal Degree Matrix D:\n");
        print_matrix(D, n, n);
        printf("\n");

        // Test normalization
        printf("=== Normalized Matrix W (section 1.3) ===\n");
        double *norm_degrees = calculate_degrees_array(A, n);
        if (!norm_degrees)
        {
            print_error();
            free_matrix(A);
            free_matrix(D);
            free(degrees);
            return ERROR;
        }

        normalize_degrees(norm_degrees, n);
        printf("D^(-1/2) values: ");
        for (int i = 0; i < n; i++)
        {
            printf("%.4f ", norm_degrees[i]);
        }
        printf("\n");

        matrix W = create_normalized_matrix(A, norm_degrees, n);
        if (!W)
        {
            print_error();
            free_matrix(A);
            free_matrix(D);
            free(degrees);
            free(norm_degrees);
            return ERROR;
        }

        printf("\nNormalized Similarity Matrix W:\n");
        print_matrix(W, n, n);
        printf("\n");

        // Cleanup
        free_matrix(A);
        free_matrix(D);
        free_matrix(W);
        free(degrees);
        free(norm_degrees);

        printf("Small matrix test completed successfully!\n");
        return SUCCESS;
    }

    if (argc == 2 && strcmp(argv[1], "matrices") == 0)
    {
        // Test matrix calculations (sections 1.1-1.3)
        const char *test_file = "tests/input_1.txt";
        int n, d;

        printf("Testing matrix calculations with %s\n\n", test_file);

        // Parse data
        matrix data = parse_file(test_file, &n, &d);
        if (!data)
        {
            print_error();
            return ERROR;
        }

        printf("Loaded %dx%d datapoints matrix\n\n", n, d);

        // Test 1.1: Similarity Matrix A
        printf("=== Testing Similarity Matrix A (section 1.1) ===\n");
        matrix A = sym_func(data, n, d);
        if (!A)
        {
            print_error();
            free_matrix(data);
            return ERROR;
        }

        printf("Similarity matrix A (%dx%d):\n", n, n);
        printf("First 5x5 submatrix:\n");
        for (int i = 0; i < 5 && i < n; i++)
        {
            for (int j = 0; j < 5 && j < n; j++)
            {
                printf("%.4f ", A[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // Test 1.2: Diagonal Degree Matrix D
        printf("=== Testing Diagonal Degree Matrix D (section 1.2) ===\n");
        double *degrees = calculate_degrees_array(A, n);
        if (!degrees)
        {
            print_error();
            free_matrix(data);
            free_matrix(A);
            return ERROR;
        }

        matrix D = create_diagonal_matrix(degrees, n);
        if (!D)
        {
            print_error();
            free_matrix(data);
            free_matrix(A);
            free(degrees);
            return ERROR;
        }

        printf("Diagonal degree matrix D (%dx%d):\n", n, n);
        printf("First 5 diagonal elements: ");
        for (int i = 0; i < 5 && i < n; i++)
        {
            printf("%.4f ", D[i][i]);
        }
        printf("\n");
        printf("Degrees array (first 5): ");
        for (int i = 0; i < 5 && i < n; i++)
        {
            printf("%.4f ", degrees[i]);
        }
        printf("\n\n");

        // Test 1.3: Normalized Similarity Matrix W
        printf("=== Testing Normalized Similarity Matrix W (section 1.3) ===\n");
        double *norm_degrees = calculate_degrees_array(A, n);
        if (!norm_degrees)
        {
            print_error();
            free_matrix(data);
            free_matrix(A);
            free_matrix(D);
            free(degrees);
            return ERROR;
        }

        normalize_degrees(norm_degrees, n);
        matrix W = create_normalized_matrix(A, norm_degrees, n);
        if (!W)
        {
            print_error();
            free_matrix(data);
            free_matrix(A);
            free_matrix(D);
            free(degrees);
            free(norm_degrees);
            return ERROR;
        }

        printf("Normalized similarity matrix W (%dx%d):\n", n, n);
        printf("First 5x5 submatrix:\n");
        for (int i = 0; i < 5 && i < n; i++)
        {
            for (int j = 0; j < 5 && j < n; j++)
            {
                printf("%.4f ", W[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("D^(-1/2) values (first 5): ");
        for (int i = 0; i < 5 && i < n; i++)
        {
            printf("%.4f ", norm_degrees[i]);
        }
        printf("\n\n");

        // Cleanup
        free_matrix(data);
        free_matrix(A);
        free_matrix(D);
        free_matrix(W);
        free(degrees);
        free(norm_degrees);

        printf("Matrix calculations test completed successfully!\n");
        return SUCCESS;
    }

    if (argc != 3)
    {
        print_error();
        return ERROR;
    }

    char *goal = argv[1];
    char *filename = argv[2];

    int n, d;
    matrix datapoints = parse_file(filename, &n, &d);
    if (!datapoints)
    {
        print_error();
        return ERROR;
    }

    matrix result = NULL;

    if (strcmp(goal, "sym") == 0)
        result = sym_func(datapoints, n, d);
    else if (strcmp(goal, "ddg") == 0)
        result = ddg_func(datapoints, n, d);
    else if (strcmp(goal, "norm") == 0)
        result = norm_func(datapoints, n, d);
    else
    {
        print_error();
        free_matrix(datapoints);
        return ERROR;
    }

    if (result)
        print_matrix(result, n, n);
    else
    {
        print_error();
        free_matrix(datapoints);
        return ERROR;
    }

    free_matrix(datapoints);
    free_matrix(result);
    return SUCCESS;
}