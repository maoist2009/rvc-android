/* rvc_pipeline.h — the official-realtime RVC conversion path, in-process.
 *
 * Port of run_rvc_npu.py::run_pipeline_ctx: 280-frame encoder context, 250 ms
 * block stride, SOLA overlap resolution.  The QNN graphs run on the NPU
 * through qnn_executor; everything else (mel front end, RMVPE post-proc,
 * GroupNorm+GELU, sine gate, SOLA, RMS match) is host code ported from the
 * numpy original.
 */
#ifndef RVC_PIPELINE_H
#define RVC_PIPELINE_H

#include <stddef.h>

#include "dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int sr;                 /* the voice's output rate (40000 for a v2 40k voice) */
  int upp;                /* prod(upsample_rates) == sr / 100 */
  int t_ph;               /* encoder context frames @100 Hz (280) */
  int t_out;              /* emitted frames per block @100 Hz (30) */
  int flow_head;          /* frames the flow keeps as history (226) */
  int dec_head;           /* z slice start for the decoder (24) */
  int t_flow;             /* flow width (54) */
  int n_spk;              /* speaker count of the checkpoint */
  int sid;                /* speaker id to render */
  float l_w, l_b;         /* the decoder's l_linear weight/bias */
  float sine_amp;         /* SineGen amplitude */
  float index_rate;       /* retrieval blend weight; <= 0 disables retrieval */
  int nprobe;             /* inverted-list probe count; <= 0 = use the file's */
} rvc_voice_t;

typedef struct {
  double ms_total, ms_init, ms_highpass, ms_f0, ms_c0, ms_gn, ms_r, ms_enc;
  double ms_synth, ms_sola, ms_rms, ms_index;
  long n_blocks;
  long n_in, n_out;
  long graph_calls;
  double rtf;             /* ms_total / output duration in ms */
  /* live pipeline state, published every feed: when no block comes out,
   * these say which gate stopped it. */
  int next_block, next_prep;
  long n_real;
  long f0_windows, f0_folded;
  unsigned long long q_next, q_done;
  int q_fail;
  int q_workers;
  char dbg[640];
} rvc_stats_t;

typedef struct rvc_pipeline rvc_pipeline_t;

rvc_pipeline_t *rvc_pipeline_create(void);

/* Load the retrieval index for [voice].  [models_dir] is searched for an
 * index.path override file, else retrieval.index.  Returns "" on success
 * (including when retrieval is disabled), else the error text. */
const char *rvc_pipeline_load_index(rvc_pipeline_t *p, const char *models_dir,
                                    const rvc_voice_t *voice);

/* Loads the seven QNN contexts and the conv0 GroupNorm parameters.
 * models_dir holds (flat): hubert_c0_s23.bin, hubert_r_2063_s23.bin,
 * hubert_enc_140_s23.bin, rmvpe_e2e_s23.bin, encflow.bin, flowreverse.bin,
 * decnsf_ext.bin, conv0_gn.f32 (512 weights then 512 biases, raw float32).
 * lib_dir is the app's nativeLibraryDir — qnn_init points ADSP_LIBRARY_PATH
 * there so the HTP stub finds libQnnHtpV79Skel.so. */
int rvc_pipeline_init(rvc_pipeline_t *p, const char *models_dir,
                      const char *lib_dir, const rvc_voice_t *voice);

/* Converts n samples of 16 kHz mono audio. Writes at most out_cap floats to
 * out and returns the count, or -1 on failure (rvc_pipeline_last_error). */
long rvc_pipeline_convert(rvc_pipeline_t *p, const float *x16k, size_t n,
                          float *out, size_t out_cap, rvc_stats_t *st);

/* Stats of the most recent convert (zero before the first call). */
const rvc_stats_t *rvc_pipeline_stats(rvc_pipeline_t *p);

/* ---------- streaming (realtime) ----------
 *
 * Same model as convert(), but fed incrementally: input arrives in arbitrary
 * chunks and every block whose look-ahead window has filled is emitted as
 * soon as it is computable.  Block i covers real input [4000i, 4000i+4800)
 * and needs input up to 4000i + ~7000 (the RMVPE window of its last f0
 * frame), so the first ~0.45 s of audio is buffered before any output; after
 * that each 250 ms of input yields one 10 000-sample block at 40 kHz.
 *
 * f0 windows and conv0 chunks are cached incrementally and the tail chunk is
 * re-run as input grows, so the per-block cost is the encoder+synth, not the
 * whole prefix. */
int rvc_pipeline_start_stream(rvc_pipeline_t *p);

/* Appends n samples of 16 kHz mono audio and writes every block that has
 * become computable into out (voice.upp * 25 floats per block). Returns the
 * total sample count written, or -1 on failure (rvc_pipeline_last_error). */
long rvc_pipeline_feed_stream(rvc_pipeline_t *p, const float *x, size_t n,
                              float *out, size_t out_cap, rvc_stats_t *st);

/* Zero-pads the input so the final block(s) are computable and emits them.
 * Returns the additional samples written, or -1. */
long rvc_pipeline_end_stream(rvc_pipeline_t *p, float *out, size_t out_cap,
                             rvc_stats_t *st);

void rvc_pipeline_destroy(rvc_pipeline_t *p);
const char *rvc_pipeline_last_error(rvc_pipeline_t *p);

#ifdef __cplusplus
}
#endif

#endif /* RVC_PIPELINE_H */
