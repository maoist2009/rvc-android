/* index_retrieval.h — the RVC faiss retrieval step, without faiss.
 *
 * A v2 checkpoint ships an added_IVF*_Flat_nprobe_1_*.index.  The official
 * pipeline runs on the 50 Hz HuBERT features, before the x2 interpolation
 * (pipeline.py:176-196):
 *
 *     score, ix = index.search(npy, k=8)             # squared L2
 *     weight  = np.square(1 / score); weight /= weight.sum(axis=1, keepdims=True)
 *     npy     = np.sum(index_vectors[ix] * weight[:, :, None], axis=1)
 *     feats   = npy * index_rate + (1 - index_rate) * feats
 *
 * This header exposes a byte-exact parse of the on-disk IndexIVFFlat layout
 * plus that search, so the app needs no faiss build at all.
 */
#ifndef RVC_INDEX_RETRIEVAL_H
#define RVC_INDEX_RETRIEVAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rvc_index rvc_index_t;

/* mmap + byte-parse a faiss IndexIVFFlat .index.  Every layout assumption that
 * can drift is asserted, so a stale or foreign file fails loudly instead of
 * silently decoding garbage.  NULL and [err] on failure.  The mapping is owned
 * by the handle until rvc_index_free. */
rvc_index_t *rvc_index_load(const char *path, char *err, size_t err_sz);

void rvc_index_free(rvc_index_t *ix);

/* Apply the retrieval in place to [nq] consecutive rows of [d] float32 (the
 * enc output of one block, i.e. (140, 768)).
 *
 * [rate] is index_rate; <= 0 is a no-op.  [nprobe] <= 0 uses the value stored
 * in the file (official behaviour — the checkpoint records nprobe=1 and the
 * reference pipeline never overrides it); 1..nlist-1 visits that many nearest
 * centroid lists; nlist or above scans every list, which is exact search and
 * costs ~11 GFLOP per block (offline only). */
void rvc_index_apply(rvc_index_t *ix, float *feat, size_t nq, float rate,
                     int nprobe);

#ifdef __cplusplus
}
#endif

#endif  /* RVC_INDEX_RETRIEVAL_H */
