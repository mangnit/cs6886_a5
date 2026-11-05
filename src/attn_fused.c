/*
 * attn_fused.c -- Q3c: row-wise fusion (the N x N matrix is never stored).
 *
 * For each query row i, the three steps run back to back on one row of scores
 * s (N floats, 16 KB at N = 4096, so it stays in L1/L2):
 *   1. s = q_i K^T * scale, with Kt = K^T and the k-j loop order of attn_loops.c
 *   2. softmax of s (the division by the sum is deferred to step 3)
 *   3. O_i = (sum_j s_j V_j) / sum, with the j-k loop order of attn_loops.c
 * Only O(N) scratch memory is used instead of the N x N matrix S.
 * Keep the driver (main) below unchanged.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): row-wise fused version.                                       */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *Kt = malloc((size_t)d * N * sizeof(float));
    float *s = malloc((size_t)N * sizeof(float));

    /* Kt = K^T, so that step 1 reads K along contiguous rows of Kt */
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    for (int i = 0; i < N; i++) {
        float *restrict srow = s;
        float *restrict orow = O + (size_t)i * d;

        /* 1. s = q_i K^T * scale */
        for (int j = 0; j < N; j++)
            srow[j] = 0.0f;
        for (int k = 0; k < d; k++) {
            const float q = Q[(size_t)i * d + k] * scale;
            const float *restrict kt = Kt + (size_t)k * N;
            for (int j = 0; j < N; j++)
                srow[j] += q * kt[j];
        }

        /* 2. softmax of s, without the final division */
        float max = srow[0];
        for (int j = 1; j < N; j++)
            if (srow[j] > max)
                max = srow[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            srow[j] = expf(srow[j] - max);
            sum += srow[j];
        }

        /* 3. O_i = (s V) / sum */
        for (int k = 0; k < d; k++)
            orow[k] = 0.0f;
        for (int j = 0; j < N; j++) {
            const float p = srow[j];
            const float *restrict v = V + (size_t)j * d;
            for (int k = 0; k < d; k++)
                orow[k] += p * v[k];
        }
        const float inv = 1.0f / sum;
        for (int k = 0; k < d; k++)
            orow[k] *= inv;
    }

    free(s);
    free(Kt);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
