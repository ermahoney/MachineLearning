import numpy as np

#Compute A+B
def problem1 (A, B):
    return A + B
#Compute AB-C
def problem2 (A, B, C):
    return np.dot(A,B) - C
#Compute element-wise product for A and B + C transpose
def problem3 (A, B, C):
    return (A*B) + np.transpose(C)
#Compute dot of x transpose S and dot it by y
def problem4 (x, S, y):
    return np.dot(np.dot(np.transpose(x),S),y)
#Given matrix A, return a vector with the same number of rows as A but that contains all ones
def problem5 (A):
    return np.ones((A.shape[0], 1))
#Given matrix A, return a matrix with the same shape and contents as A except that the diagonal terms are all zero
def problem6 (A):
    return np.fill_diagonal(A, 0)
#Given square matrix A and (scalar) α, compute A + αI
def problem7 (A, alpha):
    rows, cols = A.shape
    return A + alpha * np.eye(rows, cols, dtype = A.dtype)
#Given matrix A and integers i,j, return the ith column of the jth row of A
def problem8 (A, i, j):
    return A[j][i]
#Given matrix A and integer i, return the sum of all the entries in the ith row
def problem9 (A, i):
    return np.sum(A[i])
#Given matrix A and scalars c,d, compute the arithmetic mean over all entries of A that are between c and d
def problem10 (A, c, d):
    lessThanD = A[np.nonzero(A < d)]
    greaterThanCLessThanD = lessThanD[np.nonzero(lessThanD > c)]
    return np.mean(greaterThanCLessThanD)
#Given an (n×n) matrix A and integer k, return an (n×k) matrix containing the right-eigenvectors of A corresponding to the k eigenvalues of A with the largest magnitude
def problem11 (A, k):
    eigen = np.linalg.eig(A)[1]
    rows, cols = A.shape
    cols = cols - k
    return eigen[:,cols:]
#Compute A^-1x
def problem12 (A, x):
    return np.linalg.solve(A, x)
#Given an n-vector x and a non-negative integer k, return a n ×k matrix consisting of k copies of x
def problem13 (x, k):
    return np.atleast_2d(np.repeat(x, k, 0))
#Given a matrix A with n rows, return a matrix that results from randomly permuting the rows in A
def problem14 (A):
    return np.random.permutation(A)
