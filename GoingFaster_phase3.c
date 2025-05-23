#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void fill(double* x, int n) {

    int i;
    for (i=0, n=n*n; i<n; i++, x++)
        *x = ((double) (1 + rand() % 12345)) / ((double) (1 + rand() % 6789));
}

void matrix_mult_index (int n, double* a, double* b, double* c) {
  int i, j, k;
  for (i=0; i<n; i++)
    for (j=0; j<n; j++) {
      c[i*n+j] =0;
      for(k = 0; k < n; k++)
        c[i*n+j] += a[i*n+k] * b[k*n+j];
    }
}

void matrix_mult_ptr_reg (int n, double* a, double* b, double* c) {
    register double cij;
    register double *at, *bt;
    register int i, j, k;
    for (i=0; i<n; i++, a+=n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for(k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt+=n)
                cij += *at * *bt;
            *c = cij;
        }
}

void matrix_mult_ptr_no_reg (int n, double* a, double* b, double* c) {
    double cij;
    double *at, *bt;
    int i, j, k;
    for (i=0; i<n; i++, a+=n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for(k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt+=n)
                cij += *at * *bt;
            *c = cij;
        }
}


void matrix_mult_transpose(int n, double* a, double* b, double* c){
    double *Bt  = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;
    register int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(b_tp + j * n + i) = *(bp + i * n + j);
        }
    }

    register double cij;
    register double *at, *bt;
    register int k;

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for (k = 0, at = a, bt = &Bt[j*n]; k < n; k++, at++, bt++)
                cij += (*at) * (*bt);

            *c = cij;
        }
    }

    _aligned_free(Bt);
}

void matrix_mult_block(int n, int block_size, double* a, double* b, double* c){
    register double cij;
    register double *a_block, *b_block, *cij_block;
    register double *ai, *bj;
    register int i, j, k, x, y, z;

    for (i = 0; i < n * n; i++) {
        c[i] = 0;
    }

    for (i = 0; i < n; i += block_size) {
        for (j = 0; j < n; j += block_size) {
            cij_block = c + i * n + j;

            for (k = 0; k < n; k += block_size) {
                a_block = a + i * n + k;
                b_block = b + k * n + j;
                for (x = 0; x < block_size; x++) {

                    for (y = 0; y < block_size; y++) {
                        cij = *(cij_block + x * n + y);
                        ai = a_block + x * n;
                        bj = b_block + y;
                        for (z = 0; z < block_size; z++)
                            cij += *(ai + z) * *(bj + z * n);

                        *(cij_block + x * n + y) = cij;
                    }
                }
            }
        }
    }
}

void matrix_mult_transpose_unroll2(int n, double* a, double* b, double* c) {
    double *Bt = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;
    register int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(b_tp + j * n + i) = *(bp + i * n + j);
        }
    }

    register double cij0, cij1;
    register double *at, *bt0, *bt1;
    register int k;

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j += 2, c += 2) {
            cij0 = 0;
            cij1 = 0;
            for (k = 0, at = a, bt0 = &Bt[j * n], bt1 = bt0 + n; k < n; k++, at++, bt0++, bt1++) {
                cij0 += (*at) * (*bt0);
                cij1 += (*at) * (*bt1);
            }
            *c = cij0;
            *(c + 1) = cij1;
        }
    }

    _aligned_free(Bt);
}

void matrix_mult_transpose_unroll4(int n, double* a, double* b, double* c) {
    double *Bt = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;
    register int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(b_tp + j * n + i) = *(bp + i * n + j);
        }
    }

    register double cij0, cij1, cij2, cij3;
    register double *at, *bt0, *bt1, *bt2, *bt3;
    register int k;

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j += 4, c += 4) {
            cij0 = 0;
            cij1 = 0;
            cij2 = 0;
            cij3 = 0;
            for (k = 0, at = a, bt0 = &Bt[j * n], bt1 = bt0 + n, bt2 = bt1 + n, bt3 = bt2 + n; k < n; k++, at++, bt0++, bt1++, bt2++, bt3++) {
                cij0 += (*at) * (*bt0);
                cij1 += (*at) * (*bt1);
                cij2 += (*at) * (*bt2);
                cij3 += (*at) * (*bt3);
            }

            *c = cij0;
            *(c + 1) = cij1;
            *(c + 2) = cij2;
            *(c + 3) = cij3;
        }
    }

    _aligned_free(Bt);
}

void matrix_mult_transpose_unroll8(int n, double* a, double* b, double* c) {
    double *Bt = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;
    register int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(b_tp + j * n + i) = *(bp + i * n + j);
        }
    }

    register double cij0, cij1, cij2, cij3, cij4, cij5, cij6, cij7;
    register double *at, *bt0, *bt1, *bt2, *bt3, *bt4, *bt5, *bt6, *bt7;
    register int k;

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j += 8, c += 8) {
            cij0 = 0;
            cij1 = 0;
            cij2 = 0;
            cij3 = 0;
            cij4 = 0;
            cij5 = 0;
            cij6= 0;
            cij7 = 0;
            for (k = 0, at = a, bt0 = &Bt[j * n], bt1 = bt0 + n, bt2 = bt1 + n, bt3 = bt2 + n, bt4 = bt3 + n, bt5 = bt4 + n, bt6 = bt5 + n, bt7 = bt6 + n;
             k < n; k++, at++, bt0++, bt1++, bt2++, bt3++, bt4++, bt5++, bt6++, bt7++) {
                cij0 += (*at) * (*bt0);
                cij1 += (*at) * (*bt1);
                cij2 += (*at) * (*bt2);
                cij3 += (*at) * (*bt3);
                cij4 += (*at) * (*bt4);
                cij5 += (*at) * (*bt5);
                cij6 += (*at) * (*bt6);
                cij7 += (*at) * (*bt7);
            }

            *c = cij0;
            *(c + 1) = cij1;
            *(c + 2) = cij2;
            *(c + 3) = cij3;
            *(c + 4) = cij4;
            *(c + 5) = cij5;
            *(c + 6) = cij6;
            *(c + 7) = cij7;
        }
    }

    _aligned_free(Bt);
}





int main(){

    clock_t t0, t1;
    int n, ref;

    do{
        printf("Input size of matrix, n = ");
        scanf("%d", &n);

        ref = 0;

        double *A  = (double*)_aligned_malloc(n * n * sizeof(double), 64);
        double *B  = (double*)_aligned_malloc(n * n * sizeof(double), 64);
        double *C1 = (double*)_aligned_malloc(n * n * sizeof(double), 64);
        double *C2 = (double*)_aligned_malloc(n * n * sizeof(double), 64);

        if(A == NULL || B == NULL || C1 == NULL || C2 == NULL){
            printf("Memory Allocation Error\n\n");
            return(-1);
        }

        unsigned int seed = time(NULL);
        printf("\nseed = %u\n", seed);

        srand(seed);
        fill(A, n);
        fill(B, n);


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_index (y/n)? ");
        if(getchar() == 'y'){
            ref = 1;
            t0 = clock();
            matrix_mult_index(n, A, B, C1);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_index = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_ptr_reg (y/n)? ");
        if(getchar() == 'y'){
            ref++;
            t0 = clock();
            matrix_mult_ptr_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_reg = %0.2f s\n", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_ptr_no_reg (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_ptr_no_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_no_reg = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_block (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;

            int block_size;
            printf("\n\tInput size of block = ");
            scanf("%d", &block_size);

            t0 = clock();
            matrix_mult_block(n, block_size, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_block = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_unroll2 (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_unroll2(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_unroll2 = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_unroll4 (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_unroll4(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_unroll4 = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_unroll8 (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_unroll8(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_unroll8 = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }


        printf("\n\n\nEnd Of Execution\n\n");

        if(ref == 2){
            int i;
            double *c1, *c2;
            printf("\n\nStart of Compare: ");
            for(i=0, c1=C1, c2=C2, n=n*n; i<n; i++, c1++, c2++){
                if((fabs((*c1 - *c2) / *c1) > 1E-10))
                    break;
                if(i % (n/20) == 0)
                    printf(".");
            }

            if(i != n)
                printf(" Ooops, Error Found @ %d: %0.3f vs %0.3f\n\n",i, *c1, *c2);
            else
                printf(" OK, OK, Matrixes are equivalent.\n\n");
        }
        else
            printf("\n\nNo Compare due to No Reference or No Data.\n\n");

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C1);
        _aligned_free(C2);

        fflush(stdin);
        printf("\n\nDo you want to continue (y/n)? ");

    } while(getchar() == 'y');

    return 0;

}
