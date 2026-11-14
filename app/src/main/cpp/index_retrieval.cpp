/* index_retrieval.cpp — see index_retrieval.h.  Layout byte-verified against
 * faiss's own serialization on 2026-09-20 and bit-exact in search results
 * against faiss's search() (with nprobe override reproduced for the exact
 * mode), so this is a faithful port, not an approximation.
 */
#include "index_retrieval.h"

#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

struct rvc_index {
  int d = 0;
  int64_t ntotal = 0;
  int64_t nlist = 0;
  int64_t nprobe = 1;      /* stored in the file; official never overrides */
  size_t nbytes = 0;
  void *base = nullptr;    /* mmap base */
  const float *centroids = nullptr;   /* (nlist, d) */
  const float *vectors = nullptr;     /* (ntotal, d) row-major, ids remapped */
  std::vector<int64_t> ids;           /* logical -> vector-row */
  std::vector<int64_t> list_off;      /* list -> row offset into [ids] */
  std::vector<int64_t> sizes;         /* list -> count */
  std::vector<float> vectors_block;   /* (ntotal, d) in logical id order */
  std::vector<float> cent_norm2;      /* (nlist) squared centroid norms */
  std::vector<float> dist_buf;        /* reused (nq, nlist) coarse distances */
};

namespace {

struct rd {
  const uint8_t *p; size_t n; size_t i;
  explicit rd(const void *b, size_t sz) : p((const uint8_t *)b), n(sz), i(0) {}
  template <typename T> T take(bool *ok) {
    if (i + sizeof(T) > n) { if (ok) *ok = false; return T(); }
    T v; memcpy(&v, p + i, sizeof(T)); i += sizeof(T);
    if (ok) *ok = true; return v;
  }
  uint32_t u32(bool *ok) { return take<uint32_t>(ok); }
  uint64_t u64(bool *ok) { return take<uint64_t>(ok); }
  int32_t  i32(bool *ok) { return take<int32_t>(ok); }
  int64_t  i64(bool *ok) { return take<int64_t>(ok); }
  uint8_t  u8(bool *ok)  { return take<uint8_t>(ok); }
  float    f32(bool *ok) { return take<float>(ok); }
  void skip(size_t sz, bool *ok) {
    if (i + sz > n) { if (ok) *ok = false; return; }
    i += sz;
    if (ok) *ok = true;
  }
  void expect_tag(const char *tag, bool *ok) {
    size_t l = strlen(tag);
    if (i + l > n || memcmp(p + i, tag, l) != 0) { if (ok) *ok = false; return; }
    i += l;
    if (ok) *ok = true;
  }
};

void fail(char *err, size_t err_sz, const char *fmt, ...) {
  if (!err || !err_sz) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, err_sz, fmt, ap);
  va_end(ap);
}
}  // namespace

/* ---- byte layout, byte-exact against the reference parser ----
 * IwFl(4) | d i32 | ntotal i64 | 2x dummy i64 | is_trained u8 | metric i32 |
 * nlist i64 | nprobe i64 | "IxF2"(4) | d i32 | nq i64 | 2x dummy i64 |
 * is_trained u8 | metric i32 | nq*d i64 | nq*d f32 centroids | 9 zero bytes
 * (empty DirectMap) | "ilar"(4) | nlist i64 | code_size i64 | "full"(4) |
 * nlist i64 | sizes i64[nlist] | per list: n*code_size bytes codes then
 * n*8 bytes ids.  Every asserted assumption matches faiss's own writer.
 */
rvc_index_t *rvc_index_load(const char *path, char *err, size_t err_sz) {
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { fail(err, err_sz, "open: %s", strerror(errno)); return nullptr; }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    fail(err, err_sz, "fstat: %s", strerror(errno)); close(fd); return nullptr;
  }
  size_t nbytes = (size_t)st.st_size;
  void *base = mmap(nullptr, nbytes, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) { fail(err, err_sz, "mmap: %s", strerror(errno)); return nullptr; }

  auto *ix = new rvc_index;
  ix->nbytes = nbytes; ix->base = base;

  rd r(base, nbytes);
  bool ok = true;
  r.expect_tag("IwFl", &ok);
  int d = r.i32(&ok);
  int64_t ntotal = r.i64(&ok);
  r.i64(&ok); r.i64(&ok);        /* two dummy i64 */
  if (r.u8(&ok) != 1 || r.i32(&ok) != 1) {    /* is_trained, METRIC_L2 */
    fail(err, err_sz, "expected trained L2 header"); rvc_index_free(ix); return nullptr;
  }
  int64_t nlist = r.i64(&ok);
  int64_t nprobe = r.i64(&ok);
  r.expect_tag("IxF2", &ok);
  if (r.i32(&ok) != d) {           /* quantizer d must match */
    fail(err, err_sz, "quantizer d mismatch"); rvc_index_free(ix); return nullptr;
  }
  int64_t nq = r.i64(&ok);         /* == nlist */
  r.i64(&ok); r.i64(&ok);          /* two dummy i64 */
  r.u8(&ok); r.i32(&ok);           /* is_trained, metric again */
  int64_t centroid_count = r.i64(&ok);   /* nq * d */
  if (!ok || d != 768 || ntotal <= 0 || nlist <= 0) {
    fail(err, err_sz, "index header: d=%d ntotal=%lld nlist=%lld",
         d, (long long)ntotal, (long long)nlist);
    rvc_index_free(ix); return nullptr;
  }
  if (nq != nlist || centroid_count != nq * d) {
    fail(err, err_sz, "quantizer: nq=%lld (nlist %lld), count=%lld",
         (long long)nq, (long long)nlist, (long long)centroid_count);
    rvc_index_free(ix); return nullptr;
  }
  ix->centroids = (const float *)(r.p + r.i);
  r.skip((size_t)(nq * d * sizeof(float)), &ok);
  /* empty DirectMap: 9 zero bytes */
  r.skip(9, &ok);
  r.expect_tag("ilar", &ok);
  int64_t nlist2 = r.i64(&ok);
  int64_t code_size = r.i64(&ok);
  r.expect_tag("full", &ok);
  int64_t nlist3 = r.i64(&ok);
  if (!ok || nlist2 != nlist || nlist3 != nlist || code_size != d * 4) {
    fail(err, err_sz, "inverted list header: nlist=%lld code_size=%lld",
         (long long)nlist2, (long long)code_size);
    rvc_index_free(ix); return nullptr;
  }
  std::vector<int64_t> sizes(nlist);
  for (int64_t i = 0; i < nlist; i++) sizes[i] = r.i64(&ok);
  if (!ok) { fail(err, err_sz, "truncated at list sizes"); rvc_index_free(ix); return nullptr; }

  int64_t total = 0;
  for (int64_t i = 0; i < nlist; i++) {
    if (sizes[i] < 0 || total + sizes[i] > ntotal) {
      fail(err, err_sz, "list %lld size %lld", (long long)i, (long long)sizes[i]);
      rvc_index_free(ix); return nullptr;
    }
    total += sizes[i];
  }
  if (total != ntotal) {
    fail(err, err_sz, "list sizes sum %lld != ntotal %lld",
         (long long)total, (long long)ntotal);
    rvc_index_free(ix); return nullptr;
  }

  /* per list: codes (n*code_size bytes) then ids (n*8).  faiss keeps the
   * vectors in list order; the ids here are just labels — the retrieval
   * indexes vectors by their position within the concatenated list block,
   * exactly like the reference parser's `vectors[base:base+n] = codes`. */
  std::vector<float> vecs((size_t)total * (size_t)d);
  std::vector<int64_t> list_off(nlist + 1, 0);
  for (int64_t i = 0; i < nlist; i++) {
    int64_t sz = sizes[i];
    int64_t base = list_off[(size_t)i];
    list_off[(size_t)(i + 1)] = base + sz;
    if (sz <= 0) continue;
    const float *codes = (const float *)(r.p + r.i);
    r.skip((size_t)(sz * code_size), &ok);
    const int64_t *ids = (const int64_t *)(r.p + r.i);
    r.skip((size_t)(sz * sizeof(int64_t)), &ok);
    if (!ok) {
      fail(err, err_sz, "truncated in list %lld payload", (long long)i);
      rvc_index_free(ix); return nullptr;
    }
    memcpy(vecs.data() + (size_t)base * (size_t)d, codes,
           (size_t)sz * (size_t)d * sizeof(float));
    (void)ids;
  }
  if (r.i != nbytes) {
    fail(err, err_sz, "parse ended at %zu, file is %zu", r.i, nbytes);
    rvc_index_free(ix); return nullptr;
  }

  ix->d = d; ix->ntotal = ntotal; ix->nlist = nlist; ix->nprobe = nprobe;
  ix->sizes = std::move(sizes);
  ix->list_off = std::move(list_off);
  ix->vectors_block = std::move(vecs);
  ix->vectors = ix->vectors_block.data();

  ix->cent_norm2.resize((size_t)nlist);
  for (int64_t i = 0; i < nlist; i++) {
    const float *c = ix->centroids + (size_t)i * (size_t)d;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t t = 0; t < (size_t)d; t += 4) {
      float32x4_t cv = vld1q_f32(c + t);
      acc = vfmaq_f32(acc, cv, cv);
    }
    ix->cent_norm2[(size_t)i] = vaddvq_f32(acc);
  }

  return ix;
}

void rvc_index_free(rvc_index_t *ix) {
  if (!ix) return;
  if (ix->base) munmap(ix->base, ix->nbytes);
  delete ix;
}

/* ---- squared L2, 768 dims, NEON: 24 fma instructions per pair ---- */
static inline float sqdist768(const float *a, const float *b) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int i = 0; i < 768; i += 16) {
    float32x4_t av0 = vld1q_f32(a + i),     bv0 = vld1q_f32(b + i);
    float32x4_t av1 = vld1q_f32(a + i + 4), bv1 = vld1q_f32(b + i + 4);
    float32x4_t av2 = vld1q_f32(a + i + 8), bv2 = vld1q_f32(b + i + 8);
    float32x4_t av3 = vld1q_f32(a + i + 12), bv3 = vld1q_f32(b + i + 12);
    float32x4_t d0 = vsubq_f32(av0, bv0), d1 = vsubq_f32(av1, bv1);
    float32x4_t d2 = vsubq_f32(av2, bv2), d3 = vsubq_f32(av3, bv3);
    acc = vfmaq_f32(acc, d0, d0);
    acc = vfmaq_f32(acc, d1, d1);
    acc = vfmaq_f32(acc, d2, d2);
    acc = vfmaq_f32(acc, d3, d3);
  }
  float sum = vaddvq_f32(acc);
  return sum;
}

/* (nq x d) @ (d x nc) -> (nq x nc), row-major out.  B is stored as (nc, d)
 * rows, i.e. the product is A @ B^T.  4x4 tiles so each B row is loaded once
 * per four queries instead of once per query; d is a multiple of 4 so the
 * inner loop has no tail.
 *
 * Loop nesting: B (the centroid matrix, nc*d up to ~4.2 MB) is walked once
 * per call and A (the queries, nq*d ~0.4 MB, L2-resident) is re-streamed for
 * every centroid slab.  Walking A once and re-reading B instead makes B
 * stream from DRAM once per query group, which on this workload costs more
 * than re-streaming the smaller A from L2.  The per-(i,j) dot product's
 * accumulation order is unchanged, so the output is bit-exact either way. */
static void gemm_at(const float *A, size_t nq, const float *B, size_t nc,
                    size_t d, float *out) {
  const size_t nc4 = nc & ~(size_t)3;

  /* Outer slab over B: JB consecutive centroid rows stay in L1 while every
   * query group streams past them, so B is fetched from DRAM once per call
   * instead of once per query group (nq/4 = 35 passes).  A (nq*d ~0.4 MB)
   * is L2-resident, so re-streaming it per slab is cheap.  JB=8 keeps the
   * slab at 8*768*4 = 24 KB, which fits the 64 KB L1D alongside one 12 KB
   * query group. */
  constexpr size_t JB = 8;

  for (size_t j0 = 0; j0 < nc4; j0 += JB) {
    const size_t je = (j0 + JB < nc4) ? j0 + JB : nc4;
    for (size_t i = 0; i + 4 <= nq; i += 4) {
      const float *a0 = A + i * d, *a1 = a0 + d, *a2 = a1 + d, *a3 = a2 + d;
      float *o0 = out + i * nc, *o1 = o0 + nc, *o2 = o1 + nc, *o3 = o2 + nc;
      for (size_t j = j0; j + 4 <= je; j += 4) {
        float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0),
                    a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
        float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0),
                    a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
        float32x4_t a20 = vdupq_n_f32(0), a21 = vdupq_n_f32(0),
                    a22 = vdupq_n_f32(0), a23 = vdupq_n_f32(0);
        float32x4_t a30 = vdupq_n_f32(0), a31 = vdupq_n_f32(0),
                    a32 = vdupq_n_f32(0), a33 = vdupq_n_f32(0);
        for (size_t t = 0; t < d; t += 4) {
          float32x4_t q0 = vld1q_f32(a0 + t), q1 = vld1q_f32(a1 + t),
                      q2 = vld1q_f32(a2 + t), q3 = vld1q_f32(a3 + t);
          float32x4_t b0 = vld1q_f32(B + j * d + t),
                      b1 = vld1q_f32(B + (j + 1) * d + t),
                      b2 = vld1q_f32(B + (j + 2) * d + t),
                      b3 = vld1q_f32(B + (j + 3) * d + t);
          a00 = vfmaq_f32(a00, q0, b0); a01 = vfmaq_f32(a01, q0, b1);
          a02 = vfmaq_f32(a02, q0, b2); a03 = vfmaq_f32(a03, q0, b3);
          a10 = vfmaq_f32(a10, q1, b0); a11 = vfmaq_f32(a11, q1, b1);
          a12 = vfmaq_f32(a12, q1, b2); a13 = vfmaq_f32(a13, q1, b3);
          a20 = vfmaq_f32(a20, q2, b0); a21 = vfmaq_f32(a21, q2, b1);
          a22 = vfmaq_f32(a22, q2, b2); a23 = vfmaq_f32(a23, q2, b3);
          a30 = vfmaq_f32(a30, q3, b0); a31 = vfmaq_f32(a31, q3, b1);
          a32 = vfmaq_f32(a32, q3, b2); a33 = vfmaq_f32(a33, q3, b3);
        }
        /* each accumulator's lanes hold partial sums over disjoint dim
         * ranges, so the dot is the horizontal sum of its 4 lanes */
        o0[j] = vaddvq_f32(a00); o0[j + 1] = vaddvq_f32(a01);
        o0[j + 2] = vaddvq_f32(a02); o0[j + 3] = vaddvq_f32(a03);
        o1[j] = vaddvq_f32(a10); o1[j + 1] = vaddvq_f32(a11);
        o1[j + 2] = vaddvq_f32(a12); o1[j + 3] = vaddvq_f32(a13);
        o2[j] = vaddvq_f32(a20); o2[j + 1] = vaddvq_f32(a21);
        o2[j + 2] = vaddvq_f32(a22); o2[j + 3] = vaddvq_f32(a23);
        o3[j] = vaddvq_f32(a30); o3[j + 1] = vaddvq_f32(a31);
        o3[j + 2] = vaddvq_f32(a32); o3[j + 3] = vaddvq_f32(a33);
      }
    }
    /* query tail for this slab (nq is a multiple of 4 for the realtime
     * layout, so this is only reached for degenerate shapes) */
    for (size_t i = nq & ~(size_t)3; i < nq; i++) {
      const float *a = A + i * d;
      float *o = out + i * nc;
      for (size_t j = j0; j + 4 <= je; j += 4) {
        float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0),
                    s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
        for (size_t t = 0; t < d; t += 4) {
          float32x4_t av = vld1q_f32(a + t);
          s0 = vfmaq_f32(s0, av, vld1q_f32(B + j * d + t));
          s1 = vfmaq_f32(s1, av, vld1q_f32(B + (j + 1) * d + t));
          s2 = vfmaq_f32(s2, av, vld1q_f32(B + (j + 2) * d + t));
          s3 = vfmaq_f32(s3, av, vld1q_f32(B + (j + 3) * d + t));
        }
        o[j] = vaddvq_f32(s0); o[j + 1] = vaddvq_f32(s1);
        o[j + 2] = vaddvq_f32(s2); o[j + 3] = vaddvq_f32(s3);
      }
    }
  }

  /* centroid tail: the last nc % 4 columns are not 4-aligned, so they are
   * covered once, after the slabs.  At most 3 columns, so a plain loop. */
  for (size_t i = 0; i < nq; i++) {
    const float *a = A + i * d;
    float *o = out + i * nc;
    for (size_t j = nc4; j < nc; j++) {
      const float *b = B + j * d;
      float32x4_t s = vdupq_n_f32(0);
      for (size_t t = 0; t < d; t += 4)
        s = vfmaq_f32(s, vld1q_f32(a + t), vld1q_f32(b + t));
      o[j] = vaddvq_f32(s);
    }
  }
}

void rvc_index_apply(rvc_index_t *ix, float *feat, size_t nq, float rate,
                     int nprobe) {
  if (!ix || rate <= 0.0f || !feat || nq == 0) return;
  const int k = 8;
  const int64_t np = (nprobe <= 0) ? ix->nprobe : nprobe;

  const size_t d = (size_t)ix->d;
  const float *cent = ix->centroids;
  const float *vecs = ix->vectors;

  std::vector<int64_t> lists;
  if (np >= ix->nlist) {
    lists.resize((size_t)ix->nlist);
    for (int64_t i = 0; i < ix->nlist; i++) lists[(size_t)i] = i;
  } else {
    lists.resize((size_t)np);
  }

  /* coarse step, hoisted out of the per-query loop: all nq*nlist
   * query-centroid distances come from ONE GEMM, so each centroid is loaded
   * once per four queries instead of once per query.  D[q][c] is the exact
   * squared L2, |q|^2 + |c|^2 - 2 q.c, which is what the reference quantizer
   * search ranks on; |q|^2 is constant within a row but is folded in so the
   * stored values are real distances. */
  std::vector<float> qn2;
  const float *cn2 = nullptr;
  const float *D = nullptr;
  if (np < ix->nlist) {
    qn2.resize(nq, 0.0f);
    for (size_t q = 0; q < nq; q++) {
      const float *qq = feat + q * d;
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (size_t t = 0; t < d; t += 4) {
        float32x4_t qv = vld1q_f32(qq + t);
        acc = vfmaq_f32(acc, qv, qv);
      }
      qn2[q] = vaddvq_f32(acc);
    }
    cn2 = ix->cent_norm2.data();
    ix->dist_buf.resize(nq * (size_t)ix->nlist);
    gemm_at(feat, nq, cent, (size_t)ix->nlist, d, ix->dist_buf.data());
    D = ix->dist_buf.data();
  }

  std::vector<float> bestd((size_t)np, 1e30f);
  std::vector<int64_t> besti((size_t)np, -1);
  float cand_d[8]; int64_t cand_row[8];

  const float32x4_t vr = vdupq_n_f32(rate);
  const float32x4_t vi_ = vdupq_n_f32(1.0f - rate);

  for (size_t q = 0; q < nq; q++) {
    const float *qq = feat + q * d;

    if (np < ix->nlist) {
      /* partial top-np over the q-th distance row */
      std::fill(bestd.begin(), bestd.end(), 1e30f);
      std::fill(besti.begin(), besti.end(), -1);
      const float q2 = qn2[q];
      const float *row = D + q * (size_t)ix->nlist;
      for (int64_t c = 0; c < ix->nlist; c++) {
        float s = q2 + cn2[(size_t)c] - 2.0f * row[(size_t)c];
        int j = (int)np;
        while (j > 0 && s < bestd[j - 1]) {
          if (j < (int)np) { bestd[j] = bestd[j - 1]; besti[j] = besti[j - 1]; }
          j--;
        }
        if (j < (int)np) { bestd[j] = s; besti[j] = c; }
      }
      for (int64_t i = 0; i < np; i++) lists[(size_t)i] = besti[(size_t)i];
    }

    for (int i = 0; i < 8; i++) { cand_d[i] = 1e30f; cand_row[i] = -1; }
    for (size_t li = 0; li < lists.size(); li++) {
      int64_t list = lists[li];
      int64_t sz = ix->sizes[(size_t)list];
      int64_t off = ix->list_off[(size_t)list];
      for (int64_t j = 0; j < sz; j++) {
        int64_t row = off + j;
        float s = sqdist768(qq, vecs + (size_t)row * d);
        int m = 8;
        while (m > 0 && s < cand_d[m - 1]) {
          if (m < 8) { cand_d[m] = cand_d[m - 1]; cand_row[m] = cand_row[m - 1]; }
          m--;
        }
        if (m < 8) { cand_d[m] = s; cand_row[m] = (int64_t)row; }
      }
    }

    double wsum = 0.0;
    double w[8];
    for (int i = 0; i < k; i++) {
      if (cand_row[i] < 0) { w[i] = 0.0; continue; }
      w[i] = 1.0 / (double)cand_d[i];
      w[i] = w[i] * w[i];
      wsum += w[i];
    }
    if (wsum <= 0.0) continue;

    /* blend in place: dst = (1-rate)*feat + rate * sum_i (w_i/wsum) v_i,
     * one fma chain per element instead of an accumulate buffer plus a
     * separate write-back pass. */
    float *dst = feat + q * d;
    for (size_t c = 0; c < d; c += 16) {
      float32x4_t f0 = vld1q_f32(dst + c),     f1 = vld1q_f32(dst + c + 4),
                  f2 = vld1q_f32(dst + c + 8), f3 = vld1q_f32(dst + c + 12);
      vst1q_f32(dst + c, vmulq_f32(vi_, f0));
      vst1q_f32(dst + c + 4, vmulq_f32(vi_, f1));
      vst1q_f32(dst + c + 8, vmulq_f32(vi_, f2));
      vst1q_f32(dst + c + 12, vmulq_f32(vi_, f3));
    }
    for (int i = 0; i < k; i++) {
      if (w[i] <= 0.0) continue;
      float wi = (float)((w[i] / wsum) * (double)rate);
      const float *v = vecs + (size_t)cand_row[i] * d;
      float32x4_t wv = vdupq_n_f32(wi);
      for (size_t c = 0; c < d; c += 16) {
        float32x4_t d0 = vld1q_f32(dst + c),     v0 = vld1q_f32(v + c);
        float32x4_t d1 = vld1q_f32(dst + c + 4), v1 = vld1q_f32(v + c + 4);
        float32x4_t d2 = vld1q_f32(dst + c + 8), v2 = vld1q_f32(v + c + 8);
        float32x4_t d3 = vld1q_f32(dst + c + 12), v3 = vld1q_f32(v + c + 12);
        vst1q_f32(dst + c, vfmaq_f32(d0, wv, v0));
        vst1q_f32(dst + c + 4, vfmaq_f32(d1, wv, v1));
        vst1q_f32(dst + c + 8, vfmaq_f32(d2, wv, v2));
        vst1q_f32(dst + c + 12, vfmaq_f32(d3, wv, v3));
      }
    }
  }
}
