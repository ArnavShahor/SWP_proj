#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITERATIONS 300
#define EPSILON 1e-4
#define BETA 0.5

typedef double **matrix;

const int SUCCESS = 0;
const int ERROR = 1;
const char *GENERIC_ERROR_MSG = "An Error Has Occurred";

void print_error()
{
    printf("%s\n", GENERIC_ERROR_MSG);
}

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

void free_matrix(matrix mat)
{
    if (mat)
    {
        free(mat[0]);
        free(mat);
    }
}

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

int count_lines(const char *filename)
{
    // TODO check that we don't misscount the last linebreaks as a point
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

matrix calc_sym_mat(matrix datapoints, int n, int d)
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

matrix create_diagonal_matrix(matrix A, int n) {
    matrix D = alloc_matrix(n, n);
    if (!D)
        return NULL;

    for (int i = 0; i < n; i++) {
        double degree = 0.0;
        for (int j = 0; j < n; j++) {
            degree += A[i][j];
        }
        D[i][i] = degree;
    }

    return D;
}

matrix ddg_func(matrix datapoints, int n, int d) {
    matrix A = calc_sym_mat(datapoints, n, d);
    if (!A)
        return NULL;

    matrix D = create_diagonal_matrix(A, n);

    free_matrix(A);
    return D;
}

double* calculate_degrees_array(matrix A, int n) {
    double *degrees = (double *)malloc(n * sizeof(double));
    if (!degrees)
        return NULL;

    for (int i = 0; i < n; i++) {
        degrees[i] = 0.0;
        for (int j = 0; j < n; j++) {
            degrees[i] += A[i][j];
        }
    }

    return degrees;
}
void normalize_degrees(double *degrees, int n) {
    for (int i = 0; i < n; i++) {
        degrees[i] = pow(degrees[i], -0.5);
    }
}

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

matrix norm_func(matrix datapoints, int n, int d) {
    matrix A = calc_sym_mat(datapoints, n, d);
    if (!A)
        return NULL;

    double *degrees = calculate_degrees_array(A, n);
    if (!degrees) {
        free_matrix(A);
        return NULL;
    }

    normalize_degrees(degrees, n);
    matrix W = create_normalized_matrix(A, degrees, n);

    free_matrix(A);
    free(degrees);
    return W;
}

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

double frobenius_norm(matrix A, matrix B, int rows, int cols)
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

matrix compute_H_numerator(matrix W, matrix H, int n, int k)
{
    return matrix_multiply(W, H, n, n, k);
}

matrix compute_H_denominator(matrix H, int n, int k)
{
    matrix HT = transpose_matrix(H, n, k);
    if (!HT)
        return NULL;

    matrix HHT = matrix_multiply(H, HT, n, k, n);
    if (!HHT)
    {
        free_matrix(HT);
        return NULL;
    }

    matrix HHTH = matrix_multiply(HHT, H, n, n, k);

    free_matrix(HT);
    free_matrix(HHT);
    return HHTH;
}

matrix apply_H_update_rule(matrix H, matrix num, matrix denom, int n, int k)
{
    matrix new_H = alloc_matrix(n, k);
    if (!new_H)
        return NULL;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            new_H[i][j] = H[i][j] * (1 - BETA + BETA * (num[i][j] / denom[i][j]));
        }
    }

    return new_H;
}

matrix update_H_iteration(matrix H, matrix W, int n, int k)
{
    matrix numerator = compute_H_numerator(W, H, n, k);
    if (!numerator)
        return NULL;

    matrix denominator = compute_H_denominator(H, n, k);
    if (!denominator)
    {
        free_matrix(numerator);
        return NULL;
    }

    matrix new_H = apply_H_update_rule(H, numerator, denominator, n, k);

    free_matrix(numerator);
    free_matrix(denominator);
    return new_H;
}

matrix symnmf_func(matrix H, matrix W, int n, int k)
{
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

        double norm = frobenius_norm(new_H, current_H, n, k);
        free_matrix(current_H);
        current_H = new_H;

        if (norm < EPSILON)
        {
            break;
        }
    }

    return current_H;
}

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
    if (argc == 2 && strcmp(argv[1], "test") == 0)
    {
        // Test mode
        const char *test_file = "tests/input_1.txt";
        int n, d;

        printf("Testing file parsing with %s\n", test_file);

        // Test dimension counting
        d = count_dimensions(test_file);
        printf("Dimensions: %d\n", d);

        // Test line counting
        n = count_lines(test_file);
        printf("Lines: %d\n", n);

        // Test parsing
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
    {
        result = calc_sym_mat(datapoints, n, d);
        if (result)
        {
            print_matrix(result, n, n);
        }
    }
    else if (strcmp(goal, "ddg") == 0)
    {
        result = ddg_func(datapoints, n, d);
        if (result)
        {
            print_matrix(result, n, n);
        }
    }
    else if (strcmp(goal, "norm") == 0)
    {
        result = norm_func(datapoints, n, d);
        if (result)
        {
            print_matrix(result, n, n);
        }
    }
    else
    {
        print_error();
        free_matrix(datapoints);
        return ERROR;
    }

    if (!result)
    {
        print_error();
        free_matrix(datapoints);
        return ERROR;
    }

    free_matrix(datapoints);
    free_matrix(result);
    return SUCCESS;
}