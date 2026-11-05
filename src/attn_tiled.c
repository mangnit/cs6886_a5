/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Same three steps as attn_loops.c (S is still stored), with tiled matmuls:
 *   1. S = Q K^T: K^T is packed into panels of NR columns (d x NR floats,
 *      contiguous). Each MR x NR tile of S is accumulated in registers over the
 *      whole k loop, so one load of a panel row feeds MR rows and one Q value
 *      feeds NR columns. A panel (8 KB at d = 64) stays in L1 while all IB rows
 *      of a Q block (IB x d, in L2) pass over it.
 *   2. softmax: as in attn_loops.c.
 *   3. O = S V: each MR x DC tile of O is accumulated in registers over a block
 *      of JB rows of V; that V block (JB x d) stays in L1/L2 while all IB rows
 *      of the block use it.
 * Tile sizes are compile-time constants so the compiler can unroll the tiles
 * and keep the accumulators in vector registers.
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
/* attention(): cache- and register-tiled version.                            */
/* ------------------------------------------------------------------------- */
#ifndef MR
#define MR 4   /* rows per register tile */
#endif
#ifndef NR
#define NR 32  /* columns of S per register tile (step 1) */
#endif
#ifndef DC
#define DC 32  /* columns of O per register tile (step 3) */
#endif
#ifndef IB
#define IB 128 /* rows per cache block */
#endif
#ifndef JB
#define JB 256 /* rows of V per cache block (step 3) */
#endif

static int min_int(int a, int b) { return a < b ? a : b; }

/* s[0:MR][0:nc] = scale * q[0:MR] . panel, accumulated in an MR x NR register tile */
static void qk_tile(const float *restrict q, const float *restrict panel, int d,
                    float *restrict s, int ldS, int nc, float scale)
{
    float acc[MR][NR] = {{0}};
    for (int k = 0; k < d; k++)
        for (int r = 0; r < MR; r++)
            for (int c = 0; c < NR; c++)
                acc[r][c] += q[(size_t)r * d + k] * panel[(size_t)k * NR + c];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < nc; c++)
            s[(size_t)r * ldS + c] = acc[r][c] * scale;
}

/* same for a single leftover row */
static void qk_row(const float *restrict q, const float *restrict panel, int d,
                   float *restrict s, int nc, float scale)
{
    float acc[NR] = {0};
    for (int k = 0; k < d; k++)
        for (int c = 0; c < NR; c++)
            acc[c] += q[k] * panel[(size_t)k * NR + c];
    for (int c = 0; c < nc; c++)
        s[c] = acc[c] * scale;
}

/* o[0:MR][0:DC] += s[0:MR][0:nj] v[0:nj][0:DC], accumulated in an MR x DC register tile */
static void sv_tile(const float *restrict s, int ldS, const float *restrict v, int d, int nj,
                    float *restrict o)
{
    float acc[MR][DC];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < DC; c++)
            acc[r][c] = o[(size_t)r * d + c];
    for (int j = 0; j < nj; j++)
        for (int r = 0; r < MR; r++)
            for (int c = 0; c < DC; c++)
                acc[r][c] += s[(size_t)r * ldS + j] * v[(size_t)j * d + c];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < DC; c++)
            o[(size_t)r * d + c] = acc[r][c];
}

/* o[0:nr][0:nc] += s[0:nr][0:nj] v[0:nj][0:nc] for leftover rows and columns */
static void sv_edge(const float *s, int ldS, const float *v, int d, int nj, float *o, int nr, int nc)
{
    for (int r = 0; r < nr; r++)
        for (int j = 0; j < nj; j++) {
            const float a = s[(size_t)r * ldS + j];
            for (int c = 0; c < nc; c++)
                o[(size_t)r * d + c] += a * v[(size_t)j * d + c];
        }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int np = (N + NR - 1) / NR; /* number of K^T panels */
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kp = calloc((size_t)np * d * NR, sizeof(float));

    /* 0. pack K^T into panels: Kp[p][k][c] = K[p*NR + c][k], zero-padded past N */
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kp[((size_t)(j / NR) * d + k) * NR + j % NR] = K[(size_t)j * d + k];

    /* 1. S = Q K^T * scale, tiled */
    for (int i0 = 0; i0 < N; i0 += IB) {
        const int i1 = min_int(i0 + IB, N);
        for (int p = 0; p < np; p++) {
            const float *panel = Kp + (size_t)p * d * NR;
            const int j = p * NR, nc = min_int(NR, N - j);
            int i = i0;
            for (; i + MR <= i1; i += MR)
                qk_tile(Q + (size_t)i * d, panel, d, S + (size_t)i * N + j, N, nc, scale);
            for (; i < i1; i++)
                qk_row(Q + (size_t)i * d, panel, d, S + (size_t)i * N + j, nc, scale);
        }
    }

    /* 2. softmax of each row of S */
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; j++)
            row[j] = row[j] / sum;
    }

    /* 3. O = S V, tiled */
    memset(O, 0, (size_t)N * d * sizeof(float));
    for (int i0 = 0; i0 < N; i0 += IB) {
        const int i1 = min_int(i0 + IB, N);
        for (int j0 = 0; j0 < N; j0 += JB) {
            const int nj = min_int(JB, N - j0);
            const float *vb = V + (size_t)j0 * d;
            int i = i0;
            for (; i + MR <= i1; i += MR) {
                const float *sb = S + (size_t)i * N + j0;
                float *ob = O + (size_t)i * d;
                int c = 0;
                for (; c + DC <= d; c += DC)
                    sv_tile(sb, N, vb + c, d, nj, ob + c);
                if (c < d)
                    sv_edge(sb, N, vb + c, d, nj, ob + c, MR, d - c);
            }
            if (i < i1)
                sv_edge(S + (size_t)i * N + j0, N, vb, d, nj, O + (size_t)i * d, i1 - i, d);
        }
    }

    free(Kp);
    free(S);
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
