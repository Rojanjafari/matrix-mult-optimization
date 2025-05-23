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

void matrix_mult_transpose_vector(int n, double* a, double* b, double* c) {
    typedef double v4df __attribute__ ((vector_size (32), aligned(32)));
    v4df ctij;
    register v4df cij;
    register v4df *at, *bt;
    register int i, j, k;

    double *Bt  = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(b_tp + j * n + i) = *(bp + i * n + j);
        }
    }

    for (i=0; i<n; i++, a+=n){
        for (j = 0, bt = (v4df *) Bt; j < n; j++, c++){
            cij = (v4df) {0, 0, 0, 0};
            for(k = 0, at = (v4df *) a; k < n; k+=4, at++, bt++)
                cij += *at * *bt;

            ctij = cij;
            *c = ctij[0]+ctij[1]+ctij[2]+ctij[3];
        }
    }
    _aligned_free(Bt);
}

void matrix_mult_transpose_vector_unroll4(int n, double* a, double* b, double* c){
    typedef double v4df __attribute__ ((vector_size (32), aligned(32)));
    v4df ctij;
    register v4df cij;
    register v4df *at, *bt;
    register int i, j, k;

    double *Bt  = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(b_tp + j * n + i) = *(bp + i * n + j);
        }
    }

    for (i=0; i<n; i++, a+=n){
        for (j = 0, bt = (v4df *) Bt; j < n; j++, c++){
            v4df cij0 = (v4df) {0, 0, 0, 0};
            v4df cij1 = (v4df) {0, 0, 0, 0};
            v4df cij2 = (v4df) {0, 0, 0, 0};
            v4df cij3 = (v4df) {0, 0, 0, 0};

            for(k = 0, at = (v4df *) a; k < n; k+=16, at+=4, bt+=4) {
                cij0 += at[0] * bt[0];
                cij1 += at[1] * bt[1];
                cij2 += at[2] * bt[2];
                cij3 += at[3] * bt[3];
            }

            cij = cij0 + cij1 + cij2 + cij3;
            ctij = cij;
            *c = ctij[0]+ctij[1]+ctij[2]+ctij[3];
        }
    }
    _aligned_free(Bt);
}

void matrix_mult_vector_unroll4_block(int n, double* a, double* b, double* c) {
    typedef double v4df __attribute__ ((vector_size (32)));
    const int BLOCK_SIZE = 128;

    register int i, j, k, ii, jj, kk, i_end, j_end, k_end;

    v4df ctij0, ctij1, ctij2, ctij3;
    register v4df cij0, cij1, cij2, cij3;

    double *Bt = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    register double *bp = b;
    register double *b_tp = Bt;
    for (i = 0; i < n; i += BLOCK_SIZE) {
        for (j = 0; j < n; j += BLOCK_SIZE) {
            i_end = i + BLOCK_SIZE < n ? i + BLOCK_SIZE : n;
            j_end = j + BLOCK_SIZE < n ? j + BLOCK_SIZE : n;

            for (ii = i; ii < i_end; ii++) {
                for (jj = j; jj < j_end; jj++) {
                    *(b_tp + jj * n + ii) = *(bp + ii * n + jj);
                }
            }
        }
    }



    for (i = 0; i < n * n; i++) {
        c[i] = 0.0;
    }

    for (ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (kk = 0; kk < n; kk += BLOCK_SIZE) {
                i_end = ii + BLOCK_SIZE < n ? ii + BLOCK_SIZE : n;
                j_end = jj + BLOCK_SIZE < n ? jj + BLOCK_SIZE : n;
                k_end = kk + BLOCK_SIZE < n ? kk + BLOCK_SIZE : n;

                for (i = ii; i < i_end; i += 4) {
                    for (j = jj; j < j_end; j++) {
                        ctij0 = (v4df){0, 0, 0, 0};
                        ctij1 = (v4df){0, 0, 0, 0};
                        ctij2 = (v4df){0, 0, 0, 0};
                        ctij3 = (v4df){0, 0, 0, 0};

                        register v4df *a0 = (v4df*)(a + i * n + kk);
                        register v4df *a1 = (v4df*)(a + (i+1) * n + kk);
                        register v4df *a2 = (v4df*)(a + (i+2) * n + kk);
                        register v4df *a3 = (v4df*)(a + (i+3) * n + kk);
                        register v4df *bt = (v4df*)(Bt + j * n + kk);

                        for (k = kk; k < k_end; k += 16, a0 += 4, a1 += 4, a2 += 4, a3 += 4, bt += 4) {
                            ctij0 += a0[0] * bt[0];
                            ctij0 += a0[1] * bt[1];
                            ctij0 += a0[2] * bt[2];
                            ctij0 += a0[3] * bt[3];

                            ctij1 += a1[0] * bt[0];
                            ctij1 += a1[1] * bt[1];
                            ctij1 += a1[2] * bt[2];
                            ctij1 += a1[3] * bt[3];

                            ctij2 += a2[0] * bt[0];
                            ctij2 += a2[1] * bt[1];
                            ctij2 += a2[2] * bt[2];
                            ctij2 += a2[3] * bt[3];

                            ctij3 += a3[0] * bt[0];
                            ctij3 += a3[1] * bt[1];
                            ctij3 += a3[2] * bt[2];
                            ctij3 += a3[3] * bt[3];
                        }

                        cij0 = ctij0;
                        cij1 = ctij1;
                        cij2 = ctij2;
                        cij3 = ctij3;

                        *(c + i * n + j) += cij0[0] + cij0[1] + cij0[2] + cij0[3];
                        *(c + (i+1) * n + j) += cij1[0] + cij1[1] + cij1[2] + cij1[3];
                        *(c + (i+2) * n + j) += cij2[0] + cij2[1] + cij2[2] + cij2[3];
                        *(c + (i+3) * n + j) += cij3[0] + cij3[1] + cij3[2] + cij3[3];
                    }
                }
            }
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

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_vector (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_vector(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_vector = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to run matrix_mult_transpose_vector_unroll4 (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose_vector_unroll4(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose_vector_unroll4 = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
        }

        fflush(stdin);
        printf("\n\nDo you want to matrix_mult_vector_unroll4_block (y/n)? ");
        if(getchar() == 'y'){
            if(++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_vector_unroll4_block(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_vector_unroll4_block = %0.2f s", (float)(t1-t0)/CLOCKS_PER_SEC);
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
