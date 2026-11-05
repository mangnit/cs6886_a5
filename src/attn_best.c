/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
 *
 * Fused, blocked attention in the style of FlashAttention, with AVX-512
 * intrinsics:
 *   - Q (pre-scaled by log2(e)/sqrt(d)), K and V are packed once into 64-byte
 *     aligned buffers, zero-padded so that N is a multiple of the block sizes
 *     and d a multiple of 16; the kernels then have no tails. K^T is packed in
 *     panels of 64 keys (dp x 64 floats, contiguous).
 *   - Query rows are processed in blocks of BR and keys in blocks of BC. For
 *     each key block:
 *       a. the BR x BC score tile is computed with 4 x 64 register tiles,
 *       b. the online softmax updates each row's running max m and sum l and
 *          rescales its partial output by 2^(m_old - m_new),
 *       c. the probabilities are multiplied by the V block into the BR x dp
 *          output accumulator with 4 x 64 register tiles.
 *     The N x N matrix is never stored, and each K and V block is reused by
 *     all BR rows while it is in L1.
 *   - exp is computed as 2^x (log2(e) is folded into Q) with a degree-6
 *     polynomial and vscalefps, 16 lanes at a time; no libm call.
 * Keep the driver (main) below unchanged.
 */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
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
/* attention(): blocked, fused AVX-512 version.                               */
/* ------------------------------------------------------------------------- */
#ifndef BR
#define BR 32 /* query rows per block (multiple of 4; BR and BC must divide the larger of the two) */
#endif
#ifndef BC
#define BC 64 /* keys per block (multiple of 64) */
#endif

#define LOG2E 1.44269504088896341f

/* zeroed, 64-byte aligned buffer of n floats */
static float *alloc64(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float) + 64) != 0)
        return NULL;
    memset(p, 0, n * sizeof(float));
    return p;
}

/* 2^x for 16 lanes: 2^x = 2^n * 2^f with n = round(x) and f = x - n in [-0.5, 0.5].
 * 2^f = e^(f ln 2) by its degree-6 Taylor series (relative error < 2e-7).
 * x is clamped to -126 so that the result is never denormal. */
static inline __m512 exp2_ps(__m512 x)
{
    x = _mm512_max_ps(x, _mm512_set1_ps(-126.0f));
    const __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(x, n);
    __m512 p = _mm512_set1_ps(1.5403530e-4f);                  /* ln2^6 / 720 */
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3333558e-3f)); /* ln2^5 / 120 */
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291e-3f)); /* ln2^4 / 24 */
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504109e-2f)); /* ln2^3 / 6 */
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022651e-1f)); /* ln2^2 / 2 */
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718e-1f)); /* ln2 */
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

/* s[0:4][0:64] = q[0:4] . panel; q rows have stride dp, panel is dp x 64, s has stride BC */
static inline void score_4x64(const float *restrict q, const float *restrict panel, int dp,
                              float *restrict s)
{
    __m512 c[4][4];
    for (int r = 0; r < 4; r++)
        for (int x = 0; x < 4; x++)
            c[r][x] = _mm512_setzero_ps();
    for (int k = 0; k < dp; k++) {
        __m512 b[4];
        for (int x = 0; x < 4; x++)
            b[x] = _mm512_load_ps(panel + (size_t)k * 64 + 16 * x);
        for (int r = 0; r < 4; r++) {
            const __m512 a = _mm512_set1_ps(q[(size_t)r * dp + k]);
            for (int x = 0; x < 4; x++)
                c[r][x] = _mm512_fmadd_ps(a, b[x], c[r][x]);
        }
    }
    for (int r = 0; r < 4; r++)
        for (int x = 0; x < 4; x++)
            _mm512_store_ps(s + r * BC + 16 * x, c[r][x]);
}

/* acc[0:4][0:16W] += p[0:4][0:BC] . v[0:BC][0:16W]; p has stride BC, v and acc stride dp */
#define DEFINE_PV(NAME, W)                                                                    \
    static inline void NAME(const float *restrict p, const float *restrict v, int dp,         \
                            float *restrict acc)                                              \
    {                                                                                         \
        __m512 c[4][W];                                                                       \
        for (int r = 0; r < 4; r++)                                                           \
            for (int x = 0; x < W; x++)                                                       \
                c[r][x] = _mm512_load_ps(acc + (size_t)r * dp + 16 * x);                      \
        for (int j = 0; j < BC; j++) {                                                        \
            __m512 b[W];                                                                      \
            for (int x = 0; x < W; x++)                                                       \
                b[x] = _mm512_load_ps(v + (size_t)j * dp + 16 * x);                           \
            for (int r = 0; r < 4; r++) {                                                     \
                const __m512 a = _mm512_set1_ps(p[r * BC + j]);                               \
                for (int x = 0; x < W; x++)                                                   \
                    c[r][x] = _mm512_fmadd_ps(a, b[x], c[r][x]);                              \
            }                                                                                 \
        }                                                                                     \
        for (int r = 0; r < 4; r++)                                                           \
            for (int x = 0; x < W; x++)                                                       \
                _mm512_store_ps(acc + (size_t)r * dp + 16 * x, c[r][x]);                      \
    }
DEFINE_PV(pv_4x64, 4)
DEFINE_PV(pv_4x16, 1)

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const int blk = BR > BC ? BR : BC;
    const int Np = (N + blk - 1) / blk * blk; /* padded N */
    const int dp = (d + 15) & ~15;            /* padded d */
    const float qscale = LOG2E / sqrtf((float)d);
    float *Qp = alloc64((size_t)Np * dp);
    float *Kp = alloc64((size_t)Np * dp); /* Kp[j/64][k][j%64] = K[j][k] */
    float *Vp = alloc64((size_t)Np * dp);
    float *S = alloc64((size_t)BR * BC);
    float *acc = alloc64((size_t)BR * dp);
    float m[BR], l[BR];

    /* pack Q (scaled), K^T (in panels of 64 keys) and V */
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++) {
            Qp[(size_t)j * dp + k] = Q[(size_t)j * d + k] * qscale;
            Kp[((size_t)(j / 64) * dp + k) * 64 + j % 64] = K[(size_t)j * d + k];
            Vp[(size_t)j * dp + k] = V[(size_t)j * d + k];
        }

    for (int i0 = 0; i0 < N; i0 += BR) {
        for (int r = 0; r < BR; r++) {
            m[r] = -INFINITY;
            l[r] = 0.0f;
        }
        memset(acc, 0, (size_t)BR * dp * sizeof(float));

        for (int j0 = 0; j0 < N; j0 += BC) {
            /* a. score tile S = Qp[i0:i0+BR] Kp[j0:j0+BC]^T (already in log2 units) */
            for (int g = 0; g < BR; g += 4)
                for (int jj = 0; jj < BC; jj += 64)
                    score_4x64(Qp + (size_t)(i0 + g) * dp, Kp + (size_t)(j0 + jj) * dp, dp,
                               S + g * BC + jj);
            if (j0 + BC > N) /* padded keys must get probability 0 */
                for (int r = 0; r < BR; r++)
                    for (int j = N - j0; j < BC; j++)
                        S[r * BC + j] = -INFINITY;

            /* b. online softmax: S becomes 2^(S - m_new), acc and l are rescaled */
            for (int r = 0; r < BR; r++) {
                float *srow = S + r * BC;
                __m512 mx = _mm512_load_ps(srow);
                for (int j = 16; j < BC; j += 16)
                    mx = _mm512_max_ps(mx, _mm512_load_ps(srow + j));
                const float bmax = _mm512_reduce_max_ps(mx);
                const float m_new = bmax > m[r] ? bmax : m[r];
                const __m512 mv = _mm512_set1_ps(m_new);
                __m512 sum = _mm512_setzero_ps();
                for (int j = 0; j < BC; j += 16) {
                    const __m512 p = exp2_ps(_mm512_sub_ps(_mm512_load_ps(srow + j), mv));
                    _mm512_store_ps(srow + j, p);
                    sum = _mm512_add_ps(sum, p);
                }
                if (m_new != m[r]) {
                    const float alpha = _mm512_cvtss_f32(exp2_ps(_mm512_set1_ps(m[r] - m_new)));
                    const __m512 av = _mm512_set1_ps(alpha);
                    for (int k = 0; k < dp; k += 16) {
                        float *a = acc + (size_t)r * dp + k;
                        _mm512_store_ps(a, _mm512_mul_ps(_mm512_load_ps(a), av));
                    }
                    l[r] *= alpha;
                    m[r] = m_new;
                }
                l[r] += _mm512_reduce_add_ps(sum);
            }

            /* c. acc += P V[j0:j0+BC] */
            const float *vb = Vp + (size_t)j0 * dp;
            for (int g = 0; g < BR; g += 4) {
                int c = 0;
                for (; c + 64 <= dp; c += 64)
                    pv_4x64(S + g * BC, vb + c, dp, acc + (size_t)g * dp + c);
                for (; c < dp; c += 16)
                    pv_4x16(S + g * BC, vb + c, dp, acc + (size_t)g * dp + c);
            }
        }

        /* O = acc / l for the real (unpadded) rows and columns */
        for (int r = 0; r < BR && i0 + r < N; r++) {
            const float inv = 1.0f / l[r];
            for (int k = 0; k < d; k++)
                O[(size_t)(i0 + r) * d + k] = acc[(size_t)r * dp + k] * inv;
        }
    }

    free(acc);
    free(S);
    free(Vp);
    free(Kp);
    free(Qp);
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
