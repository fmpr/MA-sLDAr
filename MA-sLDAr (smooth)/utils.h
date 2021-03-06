#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
//double lgamma(double x);
void make_directory(char* name);
int argmax(double* x, int n);
int map_idx(int row, int col, int dim); 
double dotprod(double *a, double *b, const int&n);
void matrixprod(double *a, double **A, double *res, int n);
void matrixprod(double** A, double* a, double* res, int n);
void addmatrix(double **A, double *a, double *b, const int &n, double factor);
void addmatrix2(double **A, double *a, double *b, const int &n, double factor);
bool inverse(double **A, double **res, const int &n);
void printmatrix(double **A, double n);
#endif)

