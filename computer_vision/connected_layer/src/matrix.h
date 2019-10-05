// Include guards and C++ compatibility
#ifndef MATRIX_H
#define MATRIX_H
#ifdef __cplusplus
extern "C" {
#endif

// A matrix has size rows x cols
// and some data stored as an array of floats
// storage is row-major order:
// https://en.wikipedia.org/wiki/Row-_and_column-major_order
typedef struct matrix{
    int rows, cols;
    float *data;
    int shallow;
} matrix;


// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix make_matrix(int rows, int cols);

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix random_matrix(int rows, int cols, float s);

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(matrix m);


// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
matrix copy_matrix(matrix m);

// Perform matrix multiplication a*b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix matmul(matrix a, matrix b);

// Perform the hammard product of two matrices (element-wise multiplication)
// matrix a, b: operands
// returns: result of hammard product
matrix mathamm(matrix a, matrix b);

// Perform y = ax + y
// float a: scalar for matrix x
// matrix x: left operand to the scaled addition
// matrix y: unscaled right operand, also stores result
void axpy_matrix(float a, matrix x, matrix y);

// In-place, element-wise scaling of matrix
// float s: scaling factor
// matrix m: matrix to be scaled
void scal_matrix(float s, matrix m);

// Print a matrix
void print_matrix(matrix m);

// You won't need these
matrix solve_system(matrix M, matrix b);
matrix matrix_invert(matrix m);
matrix transpose_matrix(matrix m);
void test_matrix();

#ifdef __cplusplus
}
#endif
#endif
