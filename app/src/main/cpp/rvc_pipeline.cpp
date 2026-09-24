/* rvc_pipeline.cpp — official-realtime RVC conversion, in-process.
 *
 * Port of run_rvc_npu.py::run_pipeline_ctx: 280-frame encoder context, 250 ms
 * block stride, SOLA overlap resolution.  NPU graphs go through qnn_executor;
 * all host-side math is ported from the numpy original (dsp.cpp holds the
 * mel/FFT front end and the RMS machinery).
 */
#include "rvc_pipeline.h"
#include "index_retrieval.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <dlfcn.h>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <random>
#include <signal.h>
#include <string>
#include <thread>
#include <ucontext.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "dsp.h"
#include "qnn_executor.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* fixed constants (run_rvc_npu.py, matching the compiled graphs) */
namespace {
constexpr int SR16               = 16000;
constexpr int HUBERT_WIN_LEN     = 44960;   /* official realtime window @16k */
constexpr int HUBERT_T1_LEN      = 8991;    /* conv0(HUBERT_WIN_LEN) length */
constexpr int HUBERT_T2_LEN      = 140;     /* == t_ph / 2 */
constexpr int HUBERT_C0_CH       = 512;
constexpr int HUBERT_C0_LEN      = 10240;
constexpr int HUBERT_C0_STRIDE   = 10230;
constexpr int HUBERT_C0_T1       = 2047;
constexpr int HUBERT_C0_T1_STRIDE = 2046;   /* HUBERT_C0_STRIDE / 5 */
constexpr int HUBERT_CTX_T1_HOP  = 800;     /* CTX_STRIDE * 16 / 5... see below */
constexpr int HUBERT_R_STRIDE    = 64;
constexpr int HUBERT_R_HALO      = 79;
constexpr int HUBERT_R_N         = 32;      /* largest compiled chunk */
constexpr int HUBERT_R_LEN       = 2063;    /* 64*31 + 79 */
constexpr int HUBERT_DIM         = 768;
constexpr float HUBERT_GN_EPS    = 1e-5f;

constexpr int CTX_FRAMES  = 280;
constexpr int EMIT_FRAMES = 30;
constexpr int CTX_STRIDE  = 25;
constexpr int PAD_FRAMES  = 250;            /* 2.5 s of history @100 Hz */
constexpr int PAD_SAMPLES = 40000;
constexpr int F0_CTX_HOP  = 1600;           /* 100 ms @16k */
constexpr int F0_WIN      = 5120;           /* RMVPE window @16k */
constexpr int ENC_DIM     = 192;
constexpr int G_DIM       = 256;

constexpr float NOISE_SCALE = 0.66666f;
constexpr float NOISE_STD   = 0.003f;
constexpr float RMVPE_THRED = 0.03f;

constexpr const char *G_C0      = "c0";
constexpr const char *G_R       = "r";
constexpr const char *G_ENC     = "enc";
constexpr const char *G_RMVPE   = "rmvpe";
constexpr const char *G_ENCFLOW = "encflow";
constexpr const char *G_FLOW    = "flow";
constexpr const char *G_DEC     = "dec";

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* one conv0 chunk of the streaming c0 cache: t1 columns [2046c, 2046c+2047),
 * (512, 2047) channel-major.  conv0 is a pure strided conv, so a chunk's
 * columns are a function of its input window alone and stay valid as the
 * input grows — except the tail chunk, whose zero-padded right edge must be
 * refreshed (last_sig tracks when it was last run). */
struct c0_chunk_t {
  size_t c = 0;
  size_t last_sig = 0;
  size_t pending = 0;    /* need_sig of the in-flight refresh, 0 if none */
  std::vector<float> buf;
};

/* One graph call lifted off the calling thread.  The caller fills the input
 * description, submits, and collects later — the tensors it points at must
 * stay alive until the job runs, which the rotating slot pools guarantee. */
struct npu_job {
  uint64_t ticket = 0;
  std::string tag;
  std::vector<std::string> name_str;
  std::vector<const char *> names;
  std::vector<int> dtypes;
  std::vector<const void *> data;
  std::vector<size_t> elems;
  double *ms_acc = nullptr;
  float *out[16] = {};
  size_t out_elems[16] = {};
  size_t n_out = 0;
  int rc = -1;
  ~npu_job() {
    for (size_t i = 0; i < n_out; i++)
      if (out[i]) free(out[i]);
  }
};

/* Per-block synthesis scratch.  Two of these exist (block parity): while the
 * NPU is still reading block i's encflow/flow/dec inputs the host is already
 * building block i+1's. */
struct ctx_bufs {
  std::vector<float> m_p, logs_p, g, z_p, mask, z, gate;
  std::vector<int32_t> pitch;
  std::vector<double> f0;
  int32_t sid = 0;
};

/* A pool of reusable graph-input buffers.  acquire() waits for whatever job
 * last consumed the slot it is about to hand out, so a slot is never
 * overwritten while the NPU might still be reading it. */
template <size_t N>
struct slot_pool {
  std::vector<float> buf[N];
  uint64_t ticket[N] = {};
  size_t next = 0;
  float *acquire(rvc_pipeline *p, size_t want);
  void submitted(uint64_t t);
};
}  // namespace

/* The tombstone path is SELinux-denied on this device (crash_dump64 cannot
 * write), so a fatal signal would otherwise vanish without a trace.  This
 * handler dumps the faulting PC and a backtrace to a file the app owns. */
static char g_crash_path[256] = "";
static void rvc_crash_handler(int sig, siginfo_t *si, void *uc) {
  char buf[512];
  int n = 0;
  unsigned long pc = (unsigned long)((ucontext_t *)uc)->uc_mcontext.pc;
  n += snprintf(buf + n, sizeof(buf) - n, "\n*** signal %d code %d addr %p pc 0x%lx\n",
                sig, si->si_code, si->si_addr, pc);
  int (*bt)(void **, int) = (int (*)(void **, int))dlsym(RTLD_DEFAULT, "backtrace");
  if (bt) {
    void *frames[24];
    int nf = bt(frames, 24);
    for (int i = 0; i < nf && n < (int)sizeof(buf) - 24; i++)
      n += snprintf(buf + n, sizeof(buf) - n, "  #%d %p\n", i, frames[i]);
  }
  int fd = open(g_crash_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd >= 0) { write(fd, buf, (size_t)n); close(fd); }
  signal(sig, SIG_DFL);
  raise(sig);
}
static void rvc_install_crash_handler(const char *crash_path) {
  if (crash_path && *crash_path) snprintf(g_crash_path, sizeof(g_crash_path), "%s", crash_path);
  if (!*g_crash_path) return;
  int fd = open(g_crash_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd >= 0) close(fd);   /* start each run with an empty trail */
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = rvc_crash_handler;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
  for (int s : {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL}) sigaction(s, &sa, nullptr);
}

struct rvc_pipeline {
  rvc_dsp_t dsp{};
  rvc_voice_t voice{};
  float gn_w[HUBERT_C0_CH]{};
  float gn_b[HUBERT_C0_CH]{};
  bool inited = false;
  std::mt19937_64 rng{0};
  std::normal_distribution<double> gauss{0.0, 1.0};
  long graph_calls = 0;
  rvc_stats_t last_stats{};
  char err[1024]{};

  double randn() { return gauss(rng); }

  /* ---- streaming state (start_stream resets, feed/end use) ---- */
  bool stream_on = false;
  std::vector<float> sig;          /* PAD_SAMPLES zeros + raw 16k input */
  size_t n_real = 0;               /* real samples in sig (beyond the pad) */
  std::vector<float> hpv;          /* highpass of sig[hp_lo, sig.size()) */
  size_t hp_lo = 0;
  std::vector<double> f0_obs;      /* (n_windows * EMIT_FRAMES) */
  std::vector<double> f0_best;     /* per-frame selection cost */
  std::vector<double> f0_track;    /* per-frame selected f0 */
  size_t f0_frames = 0;            /* frames with a selected value */
  size_t f0_windows = 0;           /* RMVPE windows computed */
  std::vector<double> f0_cum;      /* f0_cum[g+1] = sum(f0_track[0..g)) */
  std::vector<c0_chunk_t> chunks;  /* cached conv0 outputs, t1-major chunks */
  std::vector<float> c0_buf;       /* (512, 8991) gathered window */
  std::vector<float> audio;        /* (t_out*upp) */
  std::vector<float> seg;
  std::vector<float> sola_buf;     /* 4*upp tail */
  std::vector<float> fade_in;
  /* reused scratch, so a block never allocates (LMK-unfriendly at 4 MB) */
  std::vector<double> sola_cv;     /* SOLA correlation window */
  std::vector<double> sola_cumsq;
  int next_block = 0;
  int next_prep = 0;        /* next block prep may run (≥ next_block) */
  double pad_phase = 0.0;
  /* streaming stats accumulators */
  double st_total = 0, st_hp = 0, st_f0 = 0, st_c0 = 0, st_gn = 0, st_r = 0;
  double st_enc = 0, st_synth = 0, st_sola = 0, st_index = 0;
  long st_blocks = 0, st_in = 0, st_out = 0;

  /* ---- async NPU execution (streaming only; the batch path stays sync) ----
   *
   * One worker thread runs every graph call, in FIFO submission order, while
   * the feeding thread does all the host-side work (mel, GroupNorm, slicing,
   * SOLA).  Overlapping them is what pushes RTF below the single-threaded
   * budget: rmvpe is 60 ms of NPU time that can run during block i+1's mel
   * and block i's GroupNorm. */
  std::thread npu_thr;
  std::mutex qmu;
  std::condition_variable qcv_work, qcv_done;
  std::deque<std::shared_ptr<npu_job>> pending;   /* submitted, not yet taken */
  std::deque<std::shared_ptr<npu_job>> done;      /* run, not yet collected */
  uint64_t q_next = 0, q_done = 0;
  bool q_stop = false, q_fail = false;
  std::atomic<int> q_workers{0};   /* >1 means a stale worker is still alive */
  std::string q_err;
  char stall[128] = "";   /* why the last feed emitted nothing */
  char qcur[48] = "";     /* job the worker is currently executing */
  /* the worker appends one line per job here before calling qnn_exec: a hung
   * or crashed run leaves the exact graph call that stopped returning, which
   * the in-memory trace cannot survive (the feed never returns to publish). */
  FILE *qlog = nullptr;
  /* submit/worker/collect trace, ring of 64 — the queue's state is otherwise
   * impossible to reconstruct after the fact. */
  struct qevt { char op; uint64_t a, b; };
  qevt qtrace[64];
  uint64_t qtrace_n = 0;
  void qtrace_push(char op, uint64_t a, uint64_t b) {
    qtrace[qtrace_n % 64] = {op, a, b};
    qtrace_n++;
  }

  slot_pool<8> mel_pool;    /* rmvpe inputs, 16 KB each */
  slot_pool<4> c0_pool;     /* conv0 chunk inputs, 40 KB each */
  slot_pool<5> rseg_pool;   /* R-slice inputs, 4.2 MB each */

  /* output-stage scratch, doubled so block i+1's host work cannot clobber
   * tensors an in-flight block i graph still reads */
  std::vector<float> gn_out[2];
  std::vector<float> cnn[2];
  std::vector<float> feat[2];
  std::vector<float> raw_feat[2];     /* (HUBERT_T2_LEN, HUBERT_DIM) pre-x2 */
  ctx_bufs ctx[2];

  rvc_index_t *index = nullptr;       /* optional; NULL = skip retrieval */

  /* prep → output handoff: tickets the output pass must wait for */
  std::deque<uint64_t> f0_job_tickets;   /* [w] = ticket of window w's rmvpe */
  std::deque<std::pair<size_t, uint64_t>> c0_pending;  /* (chunk, ticket) */
  std::vector<uint64_t> blk_f0_last;     /* last rmvpe ticket needed by block i */
  std::vector<uint64_t> blk_c0_last;     /* last c0 ticket submitted for block i */
  size_t f0_folded = 0;                  /* windows already folded into track */

  ~rvc_pipeline() { stop_npu(); if (qlog) fclose(qlog); rvc_index_free(index); }
  void stop_npu();
};

rvc_pipeline_t *rvc_pipeline_create(void) { return new rvc_pipeline(); }
void rvc_pipeline_destroy(rvc_pipeline_t *p) {
  if (p) p->stop_npu();   /* never leave the worker on freed memory */
  delete p;
}
const char *rvc_pipeline_last_error(rvc_pipeline_t *p) { return p->err; }

const char *rvc_pipeline_load_index(rvc_pipeline_t *p, const char *models_dir,
                                    const rvc_voice_t *voice) {
  rvc_index_free(p->index);
  p->index = nullptr;
  if (!voice || voice->index_rate <= 0.0f) return "";
  std::string idx = std::string(models_dir) + "/index.path";
  FILE *f = fopen(idx.c_str(), "r");
  if (f) {
    char buf[1024]; size_t got = fread(buf, 1, sizeof(buf) - 1, f);
    buf[got] = 0;
    while (got && (buf[got - 1] == '\n' || buf[got - 1] == '\r')) buf[--got] = 0;
    fclose(f);
    if (buf[0] == '/') idx = buf;
    else idx = std::string(models_dir) + "/" + buf;
  } else {
    idx = std::string(models_dir) + "/retrieval.index";
  }
  char ix_err[256] = "";
  p->index = rvc_index_load(idx.c_str(), ix_err, sizeof(ix_err));
  if (!p->index) {
    snprintf(p->err, sizeof(p->err), "retrieval index: %s",
             ix_err[0] ? ix_err : "not found");
    return p->err;
  }
  return "";
}
const rvc_stats_t *rvc_pipeline_stats(rvc_pipeline_t *p) { return &p->last_stats; }

/* defined out here (not in the anonymous namespace) because it is a member
 * of the globally-declared struct; it joins the NPU worker started by
 * start_stream so a stream that never ends still shuts its thread down. */
void rvc_pipeline::stop_npu(void) {
  if (npu_thr.joinable()) {
    {
      std::lock_guard<std::mutex> lk(qmu);
      q_stop = true;
    }
    qcv_work.notify_all();
    npu_thr.join();
  }
}

namespace {

/* ----------------------------------------------------------------- RMVPE -- */

/* RMVPE bin -> cents, with the 4-bin pad so the 9-bin window needs no bounds
 * check (infer/rmvpe.py:538). padded_index ranges over [0, 368). */
static inline double cents_mapping(int padded_index) {
  return 20.0 * (double)(padded_index - 4) + 1997.3794084376191;
}

/* rmvpe_logits_to_f0: logits (32,360) float32 -> 30 Hz values (0 = unvoiced). */
void rmvpe_logits_to_f0(const float *logits, double *f0_out) {
  const int BINS = 360;
  std::vector<double> sal((size_t)32 * BINS);
  for (size_t i = 0; i < sal.size(); i++) sal[i] = (double)logits[i];

  for (int t = 0; t < EMIT_FRAMES; t++) {
    const double *row = sal.data() + (size_t)t * BINS;
    int c = 0;
    double peak = row[0];
    for (int b = 1; b < BINS; b++)
      if (row[b] > peak) { peak = row[b]; c = b; }
    c += 4;   /* index into the padded grid */

    double wsum = 0.0, csum = 0.0;
    for (int k = -4; k <= 4; k++) {
      double w = row[c + k - 4];   /* raw row index in [0, 360) */
      wsum += w;
      csum += w * cents_mapping(c + k);
    }
    double cents = (wsum > 0.0) ? csum / wsum : 0.0;
    if (peak <= (double)RMVPE_THRED) cents = 0.0;   /* unvoiced */

    double hz = 10.0 * pow(2.0, cents / 1200.0);
    f0_out[t] = (hz == 10.0) ? 0.0 : hz;   /* 10 Hz is the "unvoiced" anchor */
  }
}

/* f0_hz_to_pitch_token: (n) Hz -> (n) int32 embedding index in 1..255. */
void f0_to_pitch(const double *f0, int32_t *out, size_t n) {
  const double F_MIN_MEL = 1127.0 * log1p(50.0 / 700.0);
  const double F_MAX_MEL = 1127.0 * log1p(1100.0 / 700.0);
  for (size_t i = 0; i < n; i++) {
    double m = 1127.0 * log1p(f0[i] / 700.0);
    if (m > 0.0) m = (m - F_MIN_MEL) * 254.0 / (F_MAX_MEL - F_MIN_MEL) + 1.0;
    double r = rint(m);
    if (r < 1.0) r = 1.0;
    if (r > 255.0) r = 255.0;
    out[i] = (int32_t)r;
  }
}

/* np.interp over the unvoiced (== 0) frames, in place. */
void interp_unvoiced(double *f0, size_t n) {
  std::vector<double> xp, fp;
  for (size_t i = 0; i < n; i++)
    if (f0[i] != 0.0) { xp.push_back((double)i); fp.push_back(f0[i]); }
  if (xp.empty()) return;
  for (size_t i = 0; i < n; i++)
    if (f0[i] == 0.0) f0[i] = rvc_np_interp((double)i, xp.data(), fp.data(), xp.size());
}

/* (32,128) time-major mel -> (1,128,32) channel-major, as the graph wants. */
void mel_to_graph(const float *mel_32x128, float *out_1_128_32) {
  for (int m = 0; m < 128; m++)
    for (int f = 0; f < 32; f++)
      out_1_128_32[m * 32 + f] = mel_32x128[f * 128 + m];
}

/* ------------------------------------------------------- GroupNorm+GELU --- */

/* HubertGroupNormConvLayer: GroupNorm(512,512) over the window's own time axis
 * (a per-channel instance norm), affine, then the exact erf GELU — evaluated
 * in b = x/sqrt2 space so the 1/sqrt2 folds into the affine.
 * src is (ch, t1_total) row-major; the window is columns [col, col+t_len). */
#if defined(__ARM_NEON)

/* GELU(z) = 0.5 z (1 + erf(z/sqrt2)); in a = z/sqrt2 space that is
 * a*(1+erf(a))/sqrt2.  erf uses the Abramowitz-Stegun 7.1.26 rational form
 * (max error 1.5e-7) with a range-reduced exp (-u = 2^-n * exp(-r)), so the
 * whole evaluation is branch free and 4 lanes wide.  Valid for |a| <= 6,
 * clamped here: GELU(a) is indistinguishable from a / 0 there (erf(6) = 1 -
 * 2e-17), and groupnormed activations live well inside the range. */
static inline float32x4_t gelu_neon(float32x4_t a) {
  const float p       = 0.3275911f;
  const float a1      = 0.254829592f;
  const float a2      = -0.284496736f;
  const float a3      = 1.421413741f;
  const float a4      = -1.453152027f;
  const float a5      = 1.061405429f;
  const float inv_ln2 = 1.4426950408889634f;
  const float ln2_hi  = 0.693145751953125f;
  const float ln2_lo  = 1.428606765330187e-06f;

  float32x4_t ac = vminq_f32(vmaxq_f32(a, vdupq_n_f32(-6.0f)), vdupq_n_f32(6.0f));
  float32x4_t abs_a = vabsq_f32(ac);
  float32x4_t u = vmulq_f32(ac, ac);

  /* exp(-u) = 2^-n * exp(-r), n = round(u/ln2) >= 0, |r| <= ln2/2 */
  int32x4_t n = vcvtnq_s32_f32(vmulq_n_f32(u, inv_ln2));
  float32x4_t fn = vcvtq_f32_s32(n);
  float32x4_t r = vfmsq_f32(vfmsq_f32(u, fn, vdupq_n_f32(ln2_hi)), fn,
                            vdupq_n_f32(ln2_lo));
  /* exp(-r) ~ 1 - r + r^2/2 - r^3/6 + r^4/24 - r^5/120 (Horner, alt. signs) */
  float32x4_t e = vdupq_n_f32(1.0f / 120.0f);
  e = vfmsq_f32(vdupq_n_f32(1.0f / 24.0f), r, e);
  e = vfmsq_f32(vdupq_n_f32(1.0f / 6.0f), r, e);
  e = vfmsq_f32(vdupq_n_f32(0.5f), r, e);
  e = vfmsq_f32(vdupq_n_f32(1.0f), r, e);
  e = vfmsq_f32(vdupq_n_f32(1.0f), r, e);
  /* multiply by 2^-n by shifting the float exponent (n <= 52, no denormals) */
  float32x4_t em = vreinterpretq_f32_s32(
      vsubq_s32(vreinterpretq_s32_f32(e), vshlq_n_s32(n, 23)));

  /* t = 1/(1 + p*|a|) to 2^-26 via two reciprocal refinements */
  float32x4_t den = vfmaq_f32(vdupq_n_f32(1.0f), abs_a, vdupq_n_f32(p));
  float32x4_t t = vrecpeq_f32(den);
  t = vmulq_f32(t, vrecpsq_f32(den, t));
  t = vmulq_f32(t, vrecpsq_f32(den, t));

  float32x4_t poly = vmlaq_f32(vdupq_n_f32(a4), t, vdupq_n_f32(a5));
  poly = vmlaq_f32(vdupq_n_f32(a3), t, poly);
  poly = vmlaq_f32(vdupq_n_f32(a2), t, poly);
  poly = vmlaq_f32(vdupq_n_f32(a1), t, poly);
  poly = vmulq_f32(t, poly);                 /* P(t) = a1 t + ... + a5 t^5 */

  float32x4_t one = vdupq_n_f32(1.0f);
  uint32x4_t sbit = vandq_u32(vreinterpretq_u32_f32(a), vdupq_n_u32(0x80000000u));
  float32x4_t s = vreinterpretq_f32_u32(vorrq_u32(sbit, vreinterpretq_u32_f32(one)));
  float32x4_t one_plus_erf = vfmsq_f32(vaddq_f32(one, s), s,
                                       vmulq_f32(poly, em));
  /* the unclamped a keeps the exact linear tails: erf saturates to +-1, so
   * GELU(a) -> a*sqrt2 / 0 beyond the clamp. */
  return vmulq_n_f32(vmulq_f32(a, one_plus_erf), 0.70710678118654752f);
}

void groupnorm_gelu(rvc_pipeline_t *p, const float *src, size_t t1_total,
                    size_t col, float *out, int t_len) {
  const double inv_sqrt2 = 0.7071067811865475;
  for (int c = 0; c < HUBERT_C0_CH; c++) {
    const float *x = src + (size_t)c * t1_total + col;
    float *y = out + (size_t)c * t_len;

    /* pass 1: mean/var in float32x4 (relative error ~1e-5, below the fp32
     * noise floor of the surrounding network) */
    float32x4_t vsum = vdupq_n_f32(0.0f), vsumsq = vdupq_n_f32(0.0f);
    int t = 0;
    for (; t + 4 <= t_len; t += 4) {
      float32x4_t v = vld1q_f32(x + t);
      vsum = vaddq_f32(vsum, v);
      vsumsq = vfmaq_f32(vsumsq, v, v);
    }
    double sum = 0.0, sumsq = 0.0;
    {
      float32x2_t lo = vget_low_f32(vsum), hi = vget_high_f32(vsum);
      sum = vaddv_f32(lo) + vaddv_f32(hi);
      float32x2_t lo2 = vget_low_f32(vsumsq), hi2 = vget_high_f32(vsumsq);
      sumsq = vaddv_f32(lo2) + vaddv_f32(hi2);
    }
    for (; t < t_len; t++) {
      double v = x[t];
      sum += v;
      sumsq += v * v;
    }

    double mean = sum / (double)t_len;
    double var = sumsq / (double)t_len - mean * mean;
    if (var < 0.0) var = 0.0;
    float scale = (float)((double)p->gn_w[c] * inv_sqrt2 /
                          sqrt(var + (double)HUBERT_GN_EPS));
    float shift = (float)((double)p->gn_b[c] * inv_sqrt2);

    /* pass 2: affine + GELU, 4 lanes at a time */
    float32x4_t vmean = vdupq_n_f32((float)mean);
    float32x4_t vs = vdupq_n_f32(scale);
    float32x4_t vb = vdupq_n_f32(shift);
    int t2 = 0;
    for (; t2 + 4 <= t_len; t2 += 4) {
      float32x4_t vx = vld1q_f32(x + t2);
      float32x4_t va = vfmaq_f32(vb, vsubq_f32(vx, vmean), vs);
      vst1q_f32(y + t2, gelu_neon(va));
    }
    for (; t2 < t_len; t2++) {
      float a = (x[t2] - (float)mean) * scale + shift;
      y[t2] = a * (1.0f + erff(a)) * (float)inv_sqrt2;
    }
  }
}

#else  /* portable scalar path */

void groupnorm_gelu(rvc_pipeline_t *p, const float *src, size_t t1_total,
                    size_t col, float *out, int t_len) {
  const double inv_sqrt2 = 0.7071067811865475;
  for (int c = 0; c < HUBERT_C0_CH; c++) {
    const float *x = src + (size_t)c * t1_total + col;
    float *y = out + (size_t)c * t_len;
    double sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < t_len; t++) { double v = x[t]; sum += v; sumsq += v * v; }
    double mean = sum / (double)t_len;
    double var = sumsq / (double)t_len - mean * mean;
    if (var < 0.0) var = 0.0;
    float scale = (float)((double)p->gn_w[c] * inv_sqrt2 /
                          sqrt(var + (double)HUBERT_GN_EPS));
    float shift = (float)((double)p->gn_b[c] * inv_sqrt2);
    for (int t = 0; t < t_len; t++) {
      float a = (x[t] - (float)mean) * scale + shift;
      y[t] = a * (1.0f + erff(a)) * (float)inv_sqrt2;
    }
  }
}

#endif

/* --------------------------------------------------------- sine gate ------ */

/* precompute_sine_gate: f0 (T) Hz -> (T*upp) float32.  float64 throughout:
 * the phase argument reaches ~2.8e6 samples where float32 resolves ±0.25. */
void precompute_sine_gate(rvc_pipeline_t *p, const double *f0, int T, int upp,
                          int sr, double l_w, double l_b, double sine_amp,
                          double phase_offset, float *gate) {
  std::vector<double> cum_prev(T);
  double acc = 0.0;
  for (int t = 0; t < T; t++) { cum_prev[t] = acc; acc += f0[t]; }
  const double two_pi_over_sr = 2.0 * M_PI / (double)sr;
  size_t g = 0;
  for (int t = 0; t < T; t++) {
    double uv = f0[t] > 0.0 ? 1.0 : 0.0;
    double noise_amp = uv * (double)NOISE_STD + (1.0 - uv) * (sine_amp / 3.0);
    for (int i = 0; i < upp; i++) {
      double total = cum_prev[t] * (double)upp + f0[t] * (double)(i + 1) +
                     phase_offset;
      double sine = sine_amp * sin(two_pi_over_sr * total) * uv;
      double noise = noise_amp * p->randn();   /* SineGen's excitation */
      gate[g++] = (float)tanh(l_w * (sine + noise) + l_b);
    }
  }
}

/* ------------------------------------------------------------ QNN runner -- */

int run_graph(rvc_pipeline_t *p, const char *tag,
              const std::vector<const char *> &names,
              const std::vector<int> &dtypes,
              const std::vector<const void *> &data,
              const std::vector<size_t> &elems,
              std::vector<float *> *outs, double *ms_acc) {
  float *od[16];
  size_t oe[16], n_out = 0;
  double t0 = now_ms();
  int rc = qnn_exec(tag, names.data(), dtypes.data(), data.data(), elems.data(),
                    elems.size(), od, oe, 16, &n_out);
  *ms_acc += now_ms() - t0;
  p->graph_calls++;
  if (rc != 0) {
    snprintf(p->err, sizeof(p->err), "%s: %s", tag, qnn_last_error());
    return -1;
  }
  outs->assign(od, od + n_out);
  return 0;
}

void free_outputs(std::vector<float *> &outs) {
  for (float *o : outs) free(o);
  outs.clear();
}

/* ------------------------------------------------------------- f0 track --- */

/* compute_f0_track: utterance-level track @100 Hz over the padded signal.
 * RMVPE runs on every 100 ms window; each global frame takes the observation
 * from the window covering it closest to that window's centre. */
int compute_f0_track(rvc_pipeline_t *p, const float *wav_pad, size_t n_pad,
                     std::vector<double> *track, double *ms_acc) {
  const int frames_per_hop = F0_CTX_HOP / (SR16 / 100);   /* 10 */
  size_t n_total = (n_pad + (size_t)(SR16 / 100) - 1) / (size_t)(SR16 / 100);
  size_t n_windows = 1;
  if (n_total > (size_t)EMIT_FRAMES)
    n_windows = (n_total - (size_t)EMIT_FRAMES + (size_t)frames_per_hop - 1) /
                (size_t)frames_per_hop + 1;
  size_t n_frames = (n_windows - 1) * (size_t)frames_per_hop + (size_t)EMIT_FRAMES;

  std::vector<double> obs(n_frames * EMIT_FRAMES, 0.0);
  std::vector<char> known(n_frames * EMIT_FRAMES, 0);
  float mel[32 * 128];
  float mel_graph[1 * 128 * 32];
  float win[F0_WIN];

  for (size_t i = 0; i < n_windows; i++) {
    size_t s = i * (size_t)F0_CTX_HOP;
    size_t take = n_pad - s;
    if (take > (size_t)F0_WIN) take = (size_t)F0_WIN;
    memcpy(win, wav_pad + s, take * sizeof(float));
    if (take < (size_t)F0_WIN)
      memset(win + take, 0, ((size_t)F0_WIN - take) * sizeof(float));

    rvc_mel_128x32(&p->dsp, win, F0_WIN, mel);
    mel_to_graph(mel, mel_graph);

    std::vector<float *> outs;
    if (run_graph(p, G_RMVPE, {"mel_128x32"}, {RVC_QNN_FLOAT32},
                  {mel_graph}, {(size_t)(128 * 32)}, &outs, ms_acc) != 0)
      return -1;
    if (outs.size() != 1) {
      snprintf(p->err, sizeof(p->err), "rmvpe: %zu outputs", outs.size());
      free_outputs(outs);
      return -1;
    }
    double f0[EMIT_FRAMES];
    rmvpe_logits_to_f0(outs[0], f0);
    free_outputs(outs);
    for (int j = 0; j < EMIT_FRAMES; j++) {
      size_t g = i * (size_t)frames_per_hop + (size_t)j;
      obs[g * EMIT_FRAMES + j] = f0[j];
      known[g * EMIT_FRAMES + j] = 1;
    }
  }

  track->assign(n_frames, 0.0);
  const double centre = (double)(EMIT_FRAMES - 1) / 2.0;
  for (size_t g = 0; g < n_frames; g++) {
    double best_cost = 1e300, best_val = 0.0;
    for (int j = 0; j < EMIT_FRAMES; j++) {
      if (!known[g * EMIT_FRAMES + j]) continue;
      double cost = fabs((double)j - centre);
      if (cost < best_cost) { best_cost = cost; best_val = obs[g * EMIT_FRAMES + j]; }
    }
    (*track)[g] = best_val;
  }
  interp_unvoiced(track->data(), n_frames);
  return 0;
}

/* ------------------------------------------------------- HuBERT context --- */

/* conv0 over the whole padded utterance (chunked, bit-exact), then per block:
 * GroupNorm+GELU on the window's own frames, R in 32-frame slices with the
 * 79-frame halo, then the encoder.  Output (n_blocks, t_ph, 768) @100 Hz. */
int compute_hubert_ctx(rvc_pipeline_t *p, std::vector<float> *wav, int n_blocks,
                       std::vector<float> *feat, double *ms_c0, double *ms_gn,
                       double *ms_r, double *ms_enc) {
  /* pad so the last block's window fits, then the conv0 tail */
  size_t need = (size_t)(n_blocks - 1) * 4000 + (size_t)HUBERT_WIN_LEN;
  if (wav->size() < need) wav->resize(need, 0.0f);
  size_t t1_total = wav->size() / 5 - 1;
  size_t n_c0 = (t1_total + (size_t)HUBERT_C0_T1_STRIDE - 1) /
                (size_t)HUBERT_C0_T1_STRIDE;
  if (n_c0 == 0) n_c0 = 1;
  size_t pad_c0 = (size_t)(n_c0 - 1) * HUBERT_C0_STRIDE + (size_t)HUBERT_C0_LEN -
                  wav->size();
  if (pad_c0 > 0) wav->resize(wav->size() + pad_c0, 0.0f);
  /* t1_total is deliberately NOT recomputed after the conv0 tail pad: the
   * extra frames the pad adds are never stored (run_rvc_npu.py:1377). */

  std::vector<float> c0((size_t)HUBERT_C0_CH * t1_total);
  std::vector<float> chunk(HUBERT_C0_LEN);
  for (size_t c = 0; c < n_c0; c++) {
    memcpy(chunk.data(), wav->data() + c * (size_t)HUBERT_C0_STRIDE,
           (size_t)HUBERT_C0_LEN * sizeof(float));
    std::vector<float *> outs;
    if (run_graph(p, G_C0, {"audio_16k"}, {RVC_QNN_FLOAT32}, {chunk.data()},
                  {(size_t)HUBERT_C0_LEN}, &outs, ms_c0) != 0)
      return -1;
    if (outs.size() != 1) {
      snprintf(p->err, sizeof(p->err), "c0: %zu outputs", outs.size());
      free_outputs(outs);
      return -1;
    }
    const float *r = outs[0];
    size_t a = c * (size_t)HUBERT_C0_T1_STRIDE;
    size_t n = (size_t)HUBERT_C0_T1;
    if (a + n > t1_total) n = t1_total > a ? t1_total - a : 0;
    for (size_t t = 0; t < n; t++)
      for (int ch = 0; ch < HUBERT_C0_CH; ch++)
        c0[(size_t)ch * t1_total + a + t] = r[(size_t)ch * HUBERT_C0_T1 + t];
    free_outputs(outs);
  }

  /* R chunk starts: 0,32,64,96,108 — the last one is shifted back so the
   * whole 140-frame window is covered without zero-padding. */
  std::vector<int> r_starts;
  for (int o = 0; o < HUBERT_T2_LEN; o += HUBERT_R_N) r_starts.push_back(o);
  if (r_starts.size() > 1 && r_starts.back() + HUBERT_R_N > HUBERT_T2_LEN)
    r_starts.back() = HUBERT_T2_LEN - HUBERT_R_N;

  std::vector<float> a_buf((size_t)HUBERT_C0_CH * HUBERT_T1_LEN);
  std::vector<float> seg((size_t)HUBERT_C0_CH * HUBERT_R_LEN);
  std::vector<float> cnn((size_t)n_blocks * HUBERT_T2_LEN * HUBERT_C0_CH);

  for (int i = 0; i < n_blocks; i++) {
    double t0 = now_ms();
    groupnorm_gelu(p, c0.data(), t1_total, (size_t)i * HUBERT_CTX_T1_HOP,
                   a_buf.data(), HUBERT_T1_LEN);
    *ms_gn += now_ms() - t0;

    for (int o0 : r_starts) {
      int s = HUBERT_R_STRIDE * o0;
      int n = HUBERT_R_LEN;
      if (n > HUBERT_T1_LEN - s) n = HUBERT_T1_LEN - s;
      memset(seg.data(), 0, seg.size() * sizeof(float));
      for (int ch = 0; ch < HUBERT_C0_CH; ch++)
        memcpy(seg.data() + (size_t)ch * HUBERT_R_LEN,
               a_buf.data() + (size_t)ch * HUBERT_T1_LEN + s,
               (size_t)n * sizeof(float));
      std::vector<float *> outs;
      if (run_graph(p, G_R, {"t1"}, {RVC_QNN_FLOAT32}, {seg.data()},
                    {(size_t)HUBERT_C0_CH * HUBERT_R_LEN}, &outs, ms_r) != 0)
        return -1;
      if (outs.size() != 1) {
        snprintf(p->err, sizeof(p->err), "r: %zu outputs", outs.size());
        free_outputs(outs);
        return -1;
      }
      float *dst = cnn.data() +
                   ((size_t)i * HUBERT_T2_LEN + (size_t)o0) * HUBERT_C0_CH;
      memcpy(dst, outs[0], (size_t)HUBERT_R_N * HUBERT_C0_CH * sizeof(float));
      free_outputs(outs);
    }
  }

  /* encoder: (1,140,512) -> (1,140,768), one graph call per block. */
  feat->assign((size_t)n_blocks * CTX_FRAMES * HUBERT_DIM, 0.0f);
  for (int i = 0; i < n_blocks; i++) {
    const float *src = cnn.data() + (size_t)i * HUBERT_T2_LEN * HUBERT_C0_CH;
    std::vector<float *> outs;
    if (run_graph(p, G_ENC, {"feat"}, {RVC_QNN_FLOAT32}, {src},
                  {(size_t)HUBERT_T2_LEN * HUBERT_C0_CH}, &outs, ms_enc) != 0)
      return -1;
    if (outs.size() != 1) {
      snprintf(p->err, sizeof(p->err), "enc: %zu outputs", outs.size());
      free_outputs(outs);
      return -1;
    }
    const float *r = outs[0];
    /* 140 HuBERT frames -> 280 latent frames by nearest duplication
     * (F.interpolate(scale_factor=2)), written straight into the output. */
    float *dst = feat->data() + (size_t)i * CTX_FRAMES * HUBERT_DIM;
    for (int f = 0; f < HUBERT_T2_LEN; f++)
      for (int d = 0; d < HUBERT_DIM; d++) {
        float v = r[(size_t)f * HUBERT_DIM + d];
        dst[(2 * f) * HUBERT_DIM + d] = v;
        dst[(2 * f + 1) * HUBERT_DIM + d] = v;
      }
    free_outputs(outs);
  }
  return 0;
}

/* ----------------------------------------------------------- synth block -- */

/* run_ctx_block: the three synthesizer graphs on one 280-frame context block.
 * phone (t_ph,768), f0 (t_ph,) -> audio (t_out*upp). */
int run_ctx_block(rvc_pipeline_t *p, const float *phone_ctx, const double *f0_ctx,
                  double phase_offset, float *audio_out, double *ms_acc) {
  const int t_ph = p->voice.t_ph, t_out = p->voice.t_out;
  const int t_flow = p->voice.t_flow, dec_head = p->voice.dec_head;
  const int upp = p->voice.upp, sr = p->voice.sr;

  std::vector<double> f0(f0_ctx, f0_ctx + t_ph);
  interp_unvoiced(f0.data(), (size_t)t_ph);

  std::vector<int32_t> pitch(t_ph);
  f0_to_pitch(f0.data(), pitch.data(), (size_t)t_ph);
  int32_t sid = p->voice.sid;

  std::vector<float> m_p((size_t)ENC_DIM * t_flow);
  std::vector<float> logs_p((size_t)ENC_DIM * t_flow);
  std::vector<float> g((size_t)G_DIM);
  {
    std::vector<float *> outs;
    if (run_graph(p, G_ENCFLOW,
                  {"phone", "pitch", "sid"},
                  {RVC_QNN_FLOAT32, RVC_QNN_INT32, RVC_QNN_INT32},
                  {phone_ctx, pitch.data(), &sid},
                  {(size_t)t_ph * HUBERT_DIM, (size_t)t_ph, 1},
                  &outs, ms_acc) != 0)
      return -1;
    if (outs.size() != 3) {
      snprintf(p->err, sizeof(p->err), "encflow: %zu outputs", outs.size());
      free_outputs(outs);
      return -1;
    }
    memcpy(m_p.data(), outs[0], m_p.size() * sizeof(float));
    memcpy(logs_p.data(), outs[1], logs_p.size() * sizeof(float));
    memcpy(g.data(), outs[2], g.size() * sizeof(float));
    free_outputs(outs);
  }

  /* z_p = m_p + eps * exp(logs_p) * NOISE_SCALE (RVC's posterior sample) */
  std::vector<float> z_p((size_t)ENC_DIM * t_flow);
  for (size_t i = 0; i < z_p.size(); i++)
    z_p[i] = m_p[i] + (float)(p->randn() * exp((double)logs_p[i]) *
                              (double)NOISE_SCALE);

  std::vector<float> mask((size_t)t_flow, 1.0f);
  std::vector<float> z((size_t)ENC_DIM * t_out);
  {
    std::vector<float *> outs;
    if (run_graph(p, G_FLOW,
                  {"z_p", "mask", "g"},
                  {RVC_QNN_FLOAT32, RVC_QNN_FLOAT32, RVC_QNN_FLOAT32},
                  {z_p.data(), mask.data(), g.data()},
                  {(size_t)ENC_DIM * t_flow, (size_t)t_flow, (size_t)G_DIM},
                  &outs, ms_acc) != 0)
      return -1;
    if (outs.size() != 1) {
      snprintf(p->err, sizeof(p->err), "flow: %zu outputs", outs.size());
      free_outputs(outs);
      return -1;
    }
    const float *r = outs[0];
    for (int ch = 0; ch < ENC_DIM; ch++)
      memcpy(z.data() + (size_t)ch * t_out,
             r + (size_t)ch * t_flow + dec_head, (size_t)t_out * sizeof(float));
    free_outputs(outs);
  }

  /* the NSF source is built from the f0's last t_out frames */
  std::vector<float> gate((size_t)t_out * upp);
  precompute_sine_gate(p, f0.data() + (t_ph - t_out), t_out, upp, sr,
                       (double)p->voice.l_w, (double)p->voice.l_b,
                       (double)p->voice.sine_amp, phase_offset, gate.data());
  {
    std::vector<float *> outs;
    if (run_graph(p, G_DEC,
                  {"z", "sid", "sine_gate_in"},
                  {RVC_QNN_FLOAT32, RVC_QNN_INT32, RVC_QNN_FLOAT32},
                  {z.data(), &sid, gate.data()},
                  {(size_t)ENC_DIM * t_out, 1, (size_t)t_out * upp},
                  &outs, ms_acc) != 0)
      return -1;
    if (outs.size() != 1) {
      snprintf(p->err, sizeof(p->err), "dec: %zu outputs", outs.size());
      free_outputs(outs);
      return -1;
    }
    memcpy(audio_out, outs[0], (size_t)t_out * upp * sizeof(float));
    free_outputs(outs);
  }
  return 0;
}

/* ------------------------------------------------------------- streaming --- */

/* Left margin of the highpass slice: the zero-phase filtfilt's edge transient
 * must not reach the samples block i actually reads. */
constexpr int HP_MARGIN = 4096;

/* How far prep runs ahead of the emitting block.  Two is what it takes to keep
 * the NPU fed: block i's GroupNorm must overlap block i+1's rmvpe, and that
 * rmvpe must already be queued when block i+2's mel starts. */
constexpr int PREP_AHEAD = 2;

/* ------------------------------------------------------- async NPU queue --- */

static std::shared_ptr<npu_job> npu_make(
    const char *tag, std::initializer_list<const char *> ns,
    std::initializer_list<int> dts, std::initializer_list<const void *> ds,
    std::initializer_list<size_t> es, double *ms_acc) {
  auto j = std::make_shared<npu_job>();
  j->tag = tag;
  j->ms_acc = ms_acc;
  j->name_str.assign(ns.begin(), ns.end());
  j->dtypes.assign(dts);
  j->data.assign(ds);
  j->elems.assign(es);
  j->names.resize(j->name_str.size());
  for (size_t i = 0; i < j->name_str.size(); i++)
    j->names[i] = j->name_str[i].c_str();
  return j;
}

static void npu_worker(rvc_pipeline *p) {
  p->q_workers.fetch_add(1);
  for (;;) {
    std::shared_ptr<npu_job> j;
    {
      std::unique_lock<std::mutex> lk(p->qmu);
      p->qcv_work.wait(lk, [p] { return p->q_stop || !p->pending.empty(); });
      if (p->pending.empty()) {
        p->q_workers.fetch_sub(1);
        return;   /* asked to stop and fully drained */
      }
      j = p->pending.front();
      p->pending.pop_front();   /* take ownership: a run job is never re-run */
    }
    float *od[16];
    size_t oe[16], n_out = 0;
    snprintf(p->qcur, sizeof(p->qcur), "%s:%llu", j->tag.c_str(),
             (unsigned long long)j->ticket);
    if (p->qlog) {
      fprintf(p->qlog, "> %s:%llu in=%zu\n", j->tag.c_str(),
              (unsigned long long)j->ticket, j->elems.size() ? j->elems[0] : 0);
      fflush(p->qlog);
    }
    double t0 = now_ms();
    int rc = qnn_exec(j->tag.c_str(), j->names.data(), j->dtypes.data(),
                      j->data.data(), j->elems.data(), j->elems.size(), od, oe,
                      16, &n_out);
    *j->ms_acc += now_ms() - t0;
    p->qcur[0] = '\0';
    if (p->qlog) {
      fprintf(p->qlog, "< %s:%llu rc=%d ms=%.1f\n", j->tag.c_str(),
              (unsigned long long)j->ticket, rc, now_ms() - t0);
      fflush(p->qlog);
    }
    {
      std::lock_guard<std::mutex> lk(p->qmu);
      p->graph_calls++;
      for (size_t i = 0; i < n_out; i++) {
        j->out[i] = od[i];
        j->out_elems[i] = oe[i];
      }
      j->n_out = n_out;
      j->rc = rc;
      if (rc != 0 && !p->q_fail) {
        p->q_fail = true;
        p->q_err = std::string(j->tag) + ": " + qnn_last_error();
      }
      p->q_done++;
      p->done.push_back(j);   /* collect retires it from here, in ticket order */
      p->qtrace_push('W', j->ticket, p->q_done);
    }
    p->qcv_done.notify_all();
  }
}

static uint64_t npu_submit(rvc_pipeline *p, const std::shared_ptr<npu_job> &j) {
  std::lock_guard<std::mutex> lk(p->qmu);
  j->ticket = ++p->q_next;
  p->pending.push_back(j);
  p->qtrace_push('S', j->ticket, p->pending.size());
  p->qcv_work.notify_one();
  return j->ticket;
}

/* Block until the job with [ticket] has run.  Does not retire it: the outputs
 * may still be consumed later. */
static int npu_wait(rvc_pipeline *p, uint64_t ticket,
                    std::shared_ptr<npu_job> *out) {
  std::unique_lock<std::mutex> lk(p->qmu);
  p->qcv_done.wait(lk, [p, ticket] { return p->q_done >= ticket; });
  for (auto &j : p->done)
    if (j->ticket == ticket) { *out = j; return j->rc; }
  return -1;   /* already collected by an earlier call */
}

/* Same, and retire the job: the caller has consumed its outputs.  Only the
 * requested ticket is retired — consumers collect out of ticket order (the
 * fold waits on f0 tickets while the feature pass collects the later R
 * slices), so dropping everything in front of it would discard jobs their
 * owners have not collected yet. */
static int npu_collect(rvc_pipeline *p, uint64_t ticket,
                       std::shared_ptr<npu_job> *out) {
  std::unique_lock<std::mutex> lk(p->qmu);
  p->qcv_done.wait(lk, [p, ticket] { return p->q_done >= ticket; });
  p->qtrace_push('C', ticket, p->done.empty() ? 0 : p->done.front()->ticket);
  for (auto it = p->done.begin(); it != p->done.end(); ++it) {
    if ((*it)->ticket == ticket) {
      std::shared_ptr<npu_job> j = *it;
      p->done.erase(it);
      p->qtrace_push('R', ticket, (uint64_t)j->rc);
      *out = j;
      return j->rc;
    }
  }
  p->qtrace_push('R', ticket, (uint64_t)-1);
  return -1;   /* already collected by an earlier call */
}

template <size_t N>
float *slot_pool<N>::acquire(rvc_pipeline *p, size_t want) {
  if (ticket[next]) {
    std::shared_ptr<npu_job> j;
    npu_wait(p, ticket[next], &j);
    ticket[next] = 0;
  }
  std::vector<float> &v = buf[next];
  if (v.size() < want) v.resize(want);
  size_t slot = next;
  next = (next + 1) % N;
  return v.data();
}

template <size_t N>
void slot_pool<N>::submitted(uint64_t t) {
  ticket[(next + N - 1) % N] = t;
}

/* Real input block i needs before it is computable: the RMVPE window of its
 * last f0 frame (25i + t_ph - 1) must have fully arrived.  compute_f0_track
 * selects, for frame g, the observation from window floor(g/10) - 1. */
static inline size_t stream_need_raw(const rvc_pipeline_t *p, int i) {
  size_t g = (size_t)i * (size_t)CTX_STRIDE + (size_t)p->voice.t_ph - 1;
  size_t w = (g >= 10) ? g / 10 - 1 : 0;
  size_t need_sig = (size_t)F0_CTX_HOP * w + (size_t)F0_WIN;
  return (need_sig > (size_t)PAD_SAMPLES) ? need_sig - (size_t)PAD_SAMPLES : 0;
}

static inline float hp_at(const rvc_pipeline_t *p, size_t off) {
  return (off >= p->hp_lo && off < p->sig.size()) ? p->hpv[off - p->hp_lo]
                                                  : 0.0f;
}

/* filtfilt the slice of sig that block i reads; hpv holds the result. */
static void stream_highpass(rvc_pipeline_t *p, int i) {
  int w_lo = i * CTX_STRIDE;
  w_lo = (w_lo >= 31) ? (w_lo - 31 + 9) / 10 : 0;   /* ceil, earliest window */
  size_t lo = (size_t)F0_CTX_HOP * (size_t)w_lo;
  if (lo > (size_t)HP_MARGIN) lo -= (size_t)HP_MARGIN;
  else lo = 0;
  if (lo < (size_t)PAD_SAMPLES) lo = (size_t)PAD_SAMPLES;
  if (lo == p->hp_lo && p->hpv.size() == p->sig.size() - lo) return;
  double t0 = now_ms();
  p->hp_lo = lo;
  p->hpv.resize(p->sig.size() - lo);
  rvc_highpass_48(p->sig.data() + lo, p->hpv.size(), p->hpv.data());
  p->st_hp += now_ms() - t0;
}

static c0_chunk_t *stream_find_chunk(rvc_pipeline_t *p, size_t c) {
  for (auto &e : p->chunks) if (e.c == c) return &e;
  return nullptr;
}

/* PREP (runs ahead of the emitting block, on the feeding thread): submit the
 * rmvpe windows block j needs and refresh any conv0 chunk whose input has
 * grown past what it currently reflects.  Nothing is collected here — the
 * output pass does that, so this stays pure host work plus submission. */
static int stream_prep_block(rvc_pipeline_t *p, int j) {
  size_t g = (size_t)j * (size_t)CTX_STRIDE + (size_t)p->voice.t_ph - 1;
  size_t need_w = (g >= 10) ? g / 10 - 1 : 0;
  stream_highpass(p, j);

  while (p->f0_windows <= need_w) {
    size_t w = p->f0_windows;
    size_t s = w * (size_t)F0_CTX_HOP;
    float win[F0_WIN];
    for (size_t k = 0; k < (size_t)F0_WIN; k++) win[k] = hp_at(p, s + k);
    float mel[32 * 128];
    rvc_mel_128x32(&p->dsp, win, F0_WIN, mel);
    float *g_in = p->mel_pool.acquire(p, (size_t)(128 * 32));
    mel_to_graph(mel, g_in);
    auto job = npu_make(G_RMVPE, {"mel_128x32"}, {RVC_QNN_FLOAT32}, {g_in},
                        {(size_t)(128 * 32)}, &p->st_f0);
    p->f0_job_tickets.push_back(npu_submit(p, job));
    p->mel_pool.submitted(p->f0_job_tickets.back());
    p->f0_windows++;
  }

  size_t col0 = (size_t)j * (size_t)HUBERT_CTX_T1_HOP;
  size_t col1 = col0 + (size_t)HUBERT_T1_LEN;
  size_t c_lo = col0 / (size_t)HUBERT_C0_T1_STRIDE;
  size_t c_hi = (col1 - 1) / (size_t)HUBERT_C0_T1_STRIDE;
  uint64_t last_c0 = ((size_t)j < p->blk_c0_last.size()) ? p->blk_c0_last[j] : 0;
  for (size_t c = c_lo; c <= c_hi; c++) {
    size_t last_col = (size_t)HUBERT_C0_T1_STRIDE * c + (size_t)HUBERT_C0_T1;
    if (last_col > col1) last_col = col1;
    size_t need_sig = 5 * last_col + 10;
    c0_chunk_t *ch = stream_find_chunk(p, c);
    if (!ch) {
      p->chunks.push_back(c0_chunk_t{});
      ch = &p->chunks.back();
      ch->c = c;
      ch->buf.assign((size_t)HUBERT_C0_CH * (size_t)HUBERT_C0_T1, 0.0f);
    }
    if (ch->last_sig >= need_sig || ch->pending >= need_sig) continue;
    ch->pending = need_sig;
    float *in = p->c0_pool.acquire(p, (size_t)HUBERT_C0_LEN);
    size_t off = (size_t)HUBERT_C0_STRIDE * c;
    for (size_t k = 0; k < (size_t)HUBERT_C0_LEN; k++) in[k] = hp_at(p, off + k);
    auto job = npu_make(G_C0, {"audio_16k"}, {RVC_QNN_FLOAT32}, {in},
                        {(size_t)HUBERT_C0_LEN}, &p->st_c0);
    last_c0 = npu_submit(p, job);
    p->c0_pool.submitted(last_c0);
    p->c0_pending.push_back({c, last_c0});
  }
  if ((size_t)(j + 1) > p->blk_c0_last.size()) p->blk_c0_last.resize(j + 1, 0);
  p->blk_c0_last[j] = last_c0;
  return 0;
}

/* Fold the rmvpe results of every window [0, need_w] into the per-frame
 * selection.  Windows are folded in order and a frame's selected value only
 * ever improves, so the incremental result matches computing the whole track
 * at once. */
static int stream_f0_fold(rvc_pipeline_t *p, size_t need_w) {
  const double centre = (double)(EMIT_FRAMES - 1) / 2.0;
  while (!p->f0_job_tickets.empty() && p->f0_folded <= need_w) {
    uint64_t ticket = p->f0_job_tickets.front();
    p->f0_job_tickets.pop_front();
    std::shared_ptr<npu_job> j;
    if (npu_collect(p, ticket, &j) != 0) {
      char pq[96];
      int pn = 0;
      std::lock_guard<std::mutex> lk(p->qmu);
      pn += snprintf(pq + pn, sizeof(pq) - pn, "done=[");
      int shown = 0;
      for (auto &jj : p->done) {
        if (shown++ >= 8) { pn += snprintf(pq + pn, sizeof(pq) - pn, ".."); break; }
        pn += snprintf(pq + pn, sizeof(pq) - pn, "%s%llu", shown > 1 ? "," : "",
                       (unsigned long long)jj->ticket);
      }
      pn += snprintf(pq + pn, sizeof(pq) - pn, "] pend=%zu tickets=%zu",
                     p->pending.size(), p->f0_job_tickets.size());
      snprintf(p->stall, sizeof(p->stall),
               "f0_fold: collect(%llu) failed, q_done=%llu q_next=%llu %s",
               (unsigned long long)ticket, (unsigned long long)p->q_done,
               (unsigned long long)p->q_next, pq);
      return -1;
    }
    double f0[EMIT_FRAMES];
    if (!j->out[0] || !j->n_out) {
      snprintf(p->err, sizeof(p->err), "f0_fold: rmvpe returned no output");
      return -1;
    }
    rmvpe_logits_to_f0(j->out[0], f0);

    size_t w = p->f0_folded;
    size_t base = w * (size_t)EMIT_FRAMES;
    if (p->f0_obs.size() < base + (size_t)EMIT_FRAMES)
      p->f0_obs.resize(base + (size_t)EMIT_FRAMES, 0.0);
    for (int k = 0; k < EMIT_FRAMES; k++) p->f0_obs[base + k] = f0[k];

    size_t g0 = w * 10;      /* window w covers frames [10w, 10w+32) */
    if (p->f0_track.size() < g0 + (size_t)EMIT_FRAMES) {
      p->f0_track.resize(g0 + (size_t)EMIT_FRAMES, 0.0);
      p->f0_best.resize(g0 + (size_t)EMIT_FRAMES, 1e300);
      p->f0_frames = p->f0_track.size();
    }
    for (int k = 0; k < EMIT_FRAMES; k++) {
      size_t g = g0 + (size_t)k;
      double cost = fabs((double)k - centre);
      if (cost < p->f0_best[g]) {
        p->f0_best[g] = cost;
        p->f0_track[g] = f0[k];
      }
    }
    p->f0_folded++;
  }
  if (p->f0_folded <= need_w)
    snprintf(p->stall, sizeof(p->stall),
             "f0_fold: tickets empty, folded=%zu need_w=%zu q_next=%llu",
             (size_t)p->f0_folded, need_w, (unsigned long long)p->q_next);
  return 0;
}

/* interp the unvoiced frames of the current prefix and rebuild the cumsum
 * the sine-gate phase is anchored to. */
static void stream_f0_finalize(rvc_pipeline_t *p) {
  interp_unvoiced(p->f0_track.data(), p->f0_frames);
  p->f0_cum.assign(p->f0_frames + 1, 0.0);
  for (size_t g = 0; g < p->f0_frames; g++)
    p->f0_cum[g + 1] = p->f0_cum[g] + p->f0_track[g];
}

/* Collect the conv0 jobs prep submitted for block i. */
static int stream_c0_collect(rvc_pipeline_t *p, int i) {
  uint64_t last = ((size_t)i < p->blk_c0_last.size()) ? p->blk_c0_last[i] : 0;
  while (!p->c0_pending.empty() && p->c0_pending.front().second <= last) {
    size_t c = p->c0_pending.front().first;
    uint64_t ticket = p->c0_pending.front().second;
    p->c0_pending.pop_front();
    std::shared_ptr<npu_job> j;
    if (npu_collect(p, ticket, &j) != 0) return -1;
    c0_chunk_t *ch = stream_find_chunk(p, c);
    if (!ch || !j->out[0] || !j->n_out) {
      snprintf(p->err, sizeof(p->err),
               "c0_collect: chunk %zu missing or graph returned no output", c);
      return -1;
    }
    size_t want = j->out_elems[0];
    if (ch->buf.size() < want) ch->buf.resize(want);
    memcpy(ch->buf.data(), j->out[0], want * sizeof(float));
    ch->last_sig = p->sig.size();
    ch->pending = 0;
  }
  return 0;
}

/* Gather the cached conv0 columns [800i, 800i+8991) into c0_buf.  A chunk is
 * re-run by prep only when input the window actually reads has arrived since
 * it was last computed: conv0 is a 10-tap stride-5 conv, so column t depends
 * on samples [5t, 5t+10).  The unfinal columns past col1 are never read by
 * this block, so a chunk that is stale only beyond col1 is used as-is. */
static void stream_c0_gather(rvc_pipeline_t *p, int i) {
  size_t col0 = (size_t)i * (size_t)HUBERT_CTX_T1_HOP;
  size_t col1 = col0 + (size_t)HUBERT_T1_LEN;
  size_t c_lo = col0 / (size_t)HUBERT_C0_T1_STRIDE;
  size_t c_hi = (col1 - 1) / (size_t)HUBERT_C0_T1_STRIDE;

  for (size_t k = 0; k < p->chunks.size();) {
    if (p->chunks[k].c < c_lo) p->chunks.erase(p->chunks.begin() + (long)k);
    else k++;
  }

  p->c0_buf.assign((size_t)HUBERT_C0_CH * (size_t)HUBERT_T1_LEN, 0.0f);
  for (size_t c = c_lo; c <= c_hi; c++) {
    const c0_chunk_t *ch = stream_find_chunk(p, c);
    size_t a = ((size_t)HUBERT_C0_T1_STRIDE * c > col0)
                   ? (size_t)HUBERT_C0_T1_STRIDE * c
                   : col0;
    size_t e = ((size_t)HUBERT_C0_T1_STRIDE * c + (size_t)HUBERT_C0_T1 < col1)
                   ? (size_t)HUBERT_C0_T1_STRIDE * c + (size_t)HUBERT_C0_T1
                   : col1;
    for (int chn = 0; chn < HUBERT_C0_CH; chn++) {
      const float *src = ch->buf.data() + (size_t)chn * (size_t)HUBERT_C0_T1 +
                         (a - (size_t)HUBERT_C0_T1_STRIDE * c);
      float *dst = p->c0_buf.data() + (size_t)chn * (size_t)HUBERT_T1_LEN +
                   (a - col0);
      memcpy(dst, src, (e - a) * sizeof(float));
    }
  }
}

/* OUTPUT PASS (block i, parity [par]): GroupNorm, the five R slices and the
 * encoder.  All five R slices are submitted in a burst so the NPU runs them
 * back to back; their results are collected afterwards. */
static int stream_out_features(rvc_pipeline_t *p, int i, int par) {
  (void)i;
  double t0 = now_ms();
  groupnorm_gelu(p, p->c0_buf.data(), (size_t)HUBERT_T1_LEN, 0,
                 p->gn_out[par].data(), HUBERT_T1_LEN);
  p->st_gn += now_ms() - t0;

  /* R chunk starts: 0,32,64,96,108 — the last shifted back like the batch. */
  static const int r_starts[] = {0, 32, 64, 96, 108};
  uint64_t r_tickets[5];
  int n_r = 0;
  for (int o0 : r_starts) {
    int s = HUBERT_R_STRIDE * o0;
    int n = HUBERT_R_LEN;
    if (n > HUBERT_T1_LEN - s) n = HUBERT_T1_LEN - s;
    float *seg = p->rseg_pool.acquire(p,
                                      (size_t)HUBERT_C0_CH * (size_t)HUBERT_R_LEN);
    if (n < HUBERT_R_LEN) {
      for (int ch = 0; ch < HUBERT_C0_CH; ch++)
        memset(seg + (size_t)ch * (size_t)HUBERT_R_LEN + (size_t)n, 0,
               (size_t)(HUBERT_R_LEN - n) * sizeof(float));
    }
    for (int ch = 0; ch < HUBERT_C0_CH; ch++)
      memcpy(seg + (size_t)ch * (size_t)HUBERT_R_LEN,
             p->gn_out[par].data() + (size_t)ch * (size_t)HUBERT_T1_LEN + s,
             (size_t)n * sizeof(float));
    auto job = npu_make(G_R, {"t1"}, {RVC_QNN_FLOAT32}, {seg},
                        {(size_t)HUBERT_C0_CH * (size_t)HUBERT_R_LEN},
                        &p->st_r);
    r_tickets[n_r] = npu_submit(p, job);
    p->rseg_pool.submitted(r_tickets[n_r]);
    n_r++;
  }
  for (int k = 0; k < n_r; k++) {
    std::shared_ptr<npu_job> j;
    if (npu_collect(p, r_tickets[k], &j) != 0) return -1;
    int o0 = r_starts[k];
    float *dst = p->cnn[par].data() + (size_t)o0 * (size_t)HUBERT_C0_CH;
    if (!j->out[0] || !j->n_out) {
      snprintf(p->err, sizeof(p->err), "r slice %d returned no output", o0);
      return -1;
    }
    memcpy(dst, j->out[0],
           (size_t)HUBERT_R_N * (size_t)HUBERT_C0_CH * sizeof(float));
  }

  auto job = npu_make(G_ENC, {"feat"}, {RVC_QNN_FLOAT32},
                      {p->cnn[par].data()},
                      {(size_t)HUBERT_T2_LEN * (size_t)HUBERT_C0_CH},
                      &p->st_enc);
  uint64_t t_enc = npu_submit(p, job);
  std::shared_ptr<npu_job> je;
  if (npu_collect(p, t_enc, &je) != 0) return -1;
  const float *r = je->out[0];
  if (!r || !je->n_out) {
    snprintf(p->err, sizeof(p->err), "enc returned no output");
    return -1;
  }
  const int n = HUBERT_T2_LEN;
  p->feat[par].resize((size_t)CTX_FRAMES * (size_t)HUBERT_DIM);
  if (p->index) {
    /* retrieval is on the raw (n, d) rows — BEFORE the x2 repeat, per the
     * official pipeline (run_rvc_npu.apply_index_retrieval). */
    p->raw_feat[par].resize((size_t)n * (size_t)HUBERT_DIM);
    memcpy(p->raw_feat[par].data(), r, (size_t)n * (size_t)HUBERT_DIM * sizeof(float));
    double tix = now_ms();
    rvc_index_apply(p->index, p->raw_feat[par].data(), (size_t)n,
                    p->voice.index_rate, p->voice.nprobe);
    p->st_index += now_ms() - tix;
    r = p->raw_feat[par].data();
  }
  for (int f = 0; f < n; f++)
    for (int d = 0; d < HUBERT_DIM; d++) {
      float v = r[(size_t)f * HUBERT_DIM + d];
      p->feat[par][(2 * f) * HUBERT_DIM + d] = v;
      p->feat[par][(2 * f + 1) * HUBERT_DIM + d] = v;
    }
  return 0;
}

/* encflow → posterior sample → flow → NSF gate → dec, then SOLA.  Host work
 * is ordered so the NPU never waits on it: the sine gate is built while
 * encflow runs, and z_p while the previous job is still in flight. */
static int stream_synth(rvc_pipeline_t *p, int i, int par, float *out) {
  const int t_ph = p->voice.t_ph, t_out = p->voice.t_out;
  const int t_flow = p->voice.t_flow, dec_head = p->voice.dec_head;
  const int upp = p->voice.upp, sr = p->voice.sr;
  const int block_frame = CTX_STRIDE * upp;
  const int sola_buffer = 4 * upp;
  const int sola_search = upp;
  ctx_bufs *c = &p->ctx[par];

  size_t c0i = (size_t)i * (size_t)CTX_STRIDE;
  double phase_off = (p->f0_cum[c0i + (size_t)PAD_FRAMES] -
                      p->f0_cum[(size_t)PAD_FRAMES]) * (double)upp;

  c->f0.assign(p->f0_track.data() + c0i, p->f0_track.data() + c0i + t_ph);
  interp_unvoiced(c->f0.data(), (size_t)t_ph);
  c->pitch.resize((size_t)t_ph);
  f0_to_pitch(c->f0.data(), c->pitch.data(), (size_t)t_ph);
  c->sid = p->voice.sid;

  auto job = npu_make(G_ENCFLOW, {"phone", "pitch", "sid"},
                      {RVC_QNN_FLOAT32, RVC_QNN_INT32, RVC_QNN_INT32},
                      {p->feat[par].data(), c->pitch.data(), &c->sid},
                      {(size_t)t_ph * HUBERT_DIM, (size_t)t_ph, 1},
                      &p->st_synth);
  uint64_t t_ef = npu_submit(p, job);

  /* the NSF source only depends on f0, so build it while encflow runs */
  c->gate.resize((size_t)t_out * upp);
  precompute_sine_gate(p, c->f0.data() + (t_ph - t_out), t_out, upp, sr,
                       (double)p->voice.l_w, (double)p->voice.l_b,
                       (double)p->voice.sine_amp, phase_off, c->gate.data());

  std::shared_ptr<npu_job> jef;
  if (npu_collect(p, t_ef, &jef) != 0) return -1;
  if (jef->n_out != 3) {
    snprintf(p->err, sizeof(p->err), "encflow: %zu outputs", jef->n_out);
    return -1;
  }
  c->m_p.resize((size_t)ENC_DIM * t_flow);
  c->logs_p.resize((size_t)ENC_DIM * t_flow);
  c->g.resize((size_t)G_DIM);
  memcpy(c->m_p.data(), jef->out[0], c->m_p.size() * sizeof(float));
  memcpy(c->logs_p.data(), jef->out[1], c->logs_p.size() * sizeof(float));
  memcpy(c->g.data(), jef->out[2], c->g.size() * sizeof(float));

  /* z_p = m_p + eps * exp(logs_p) * NOISE_SCALE (RVC's posterior sample) */
  c->z_p.resize((size_t)ENC_DIM * t_flow);
  for (size_t k = 0; k < c->z_p.size(); k++)
    c->z_p[k] = c->m_p[k] + (float)(p->randn() * exp((double)c->logs_p[k]) *
                                    (double)NOISE_SCALE);
  c->mask.assign((size_t)t_flow, 1.0f);

  job = npu_make(G_FLOW, {"z_p", "mask", "g"},
                 {RVC_QNN_FLOAT32, RVC_QNN_FLOAT32, RVC_QNN_FLOAT32},
                 {c->z_p.data(), c->mask.data(), c->g.data()},
                 {(size_t)ENC_DIM * t_flow, (size_t)t_flow, (size_t)G_DIM},
                 &p->st_synth);
  uint64_t t_fl = npu_submit(p, job);
  std::shared_ptr<npu_job> jfl;
  if (npu_collect(p, t_fl, &jfl) != 0) return -1;
  if (jfl->n_out != 1) {
    snprintf(p->err, sizeof(p->err), "flow: %zu outputs", jfl->n_out);
    return -1;
  }
  c->z.resize((size_t)ENC_DIM * t_out);
  const float *r = jfl->out[0];
  for (int ch = 0; ch < ENC_DIM; ch++)
    memcpy(c->z.data() + (size_t)ch * t_out, r + (size_t)ch * t_flow + dec_head,
           (size_t)t_out * sizeof(float));

  job = npu_make(G_DEC, {"z", "sid", "sine_gate_in"},
                 {RVC_QNN_FLOAT32, RVC_QNN_INT32, RVC_QNN_FLOAT32},
                 {c->z.data(), &c->sid, c->gate.data()},
                 {(size_t)ENC_DIM * t_out, 1, (size_t)t_out * upp},
                 &p->st_synth);
  uint64_t t_dc = npu_submit(p, job);
  std::shared_ptr<npu_job> jdc;
  if (npu_collect(p, t_dc, &jdc) != 0) return -1;
  if (jdc->n_out != 1) {
    snprintf(p->err, sizeof(p->err), "dec: %zu outputs", jdc->n_out);
    return -1;
  }
  memcpy(p->audio.data(), jdc->out[0], (size_t)t_out * upp * sizeof(float));

  double t0 = now_ms();
  int conv_len = sola_buffer + sola_search;
  p->sola_cv.resize(conv_len);
  p->sola_cumsq.resize(conv_len + 1);
  double *cv = p->sola_cv.data();
  double *cumsq = p->sola_cumsq.data();
  for (int k = 0; k < conv_len; k++) cv[k] = (double)p->audio[k];
  for (int k = 0; k < conv_len; k++) cumsq[k + 1] = cumsq[k] + cv[k] * cv[k];
  int best_off = 0;
  double best_ratio = -1e300;
  for (int off = 0; off <= sola_search; off++) {
    double s = 0.0;
    for (int j = 0; j < sola_buffer; j++) s += cv[off + j] * (double)p->sola_buf[j];
    double den = sqrt(cumsq[off + sola_buffer] - cumsq[off] + 1e-8);
    double r2 = den > 0.0 ? s / den : s;
    if (r2 > best_ratio) { best_ratio = r2; best_off = off; }
  }
  int seg_len = (t_out * upp) - best_off;
  memcpy(p->seg.data(), p->audio.data() + best_off, (size_t)seg_len * sizeof(float));
  for (int j = 0; j < sola_buffer; j++)
    p->seg[j] = p->seg[j] * p->fade_in[j] + p->sola_buf[j] * (1.0f - p->fade_in[j]);
  memcpy(p->sola_buf.data(), p->seg.data() + block_frame,
         (size_t)sola_buffer * sizeof(float));
  memcpy(out, p->seg.data(), (size_t)block_frame * sizeof(float));

  /* official scales down only, never up (pipeline.py:402-405) */
  float peak = 0.0f;
  for (int k = 0; k < block_frame; k++) {
    float a = fabsf(out[k]);
    if (a > peak) peak = a;
  }
  if (peak > 0.99f) {
    float s = 0.99f / peak;
    for (int k = 0; k < block_frame; k++) out[k] *= s;
  }
  p->st_sola += now_ms() - t0;
  return 0;
}

}  // namespace

int rvc_pipeline_init(rvc_pipeline_t *p, const char *models_dir,
                      const char *lib_dir, const rvc_voice_t *voice) {
  if (p->inited) return 0;
  if (rvc_dsp_init(&p->dsp) != 0) {
    snprintf(p->err, sizeof(p->err), "dsp init failed");
    return -1;
  }
  p->voice = *voice;

  std::string dir(models_dir);
  auto join = [&](const char *f) { return dir + "/" + f; };

  /* per-job trail survives a hung feed or a process kill (models_dir is
   * filesDir/models, so the parent is the app's writable files dir).  Each
   * entry is an fflush, which costs the NPU worker ~0.5 ms — it stays off by
   * default and turns on only when filesDir/models/qlog_on exists. */
  std::string crash_path = dir + "/../crash.txt";
  rvc_install_crash_handler(crash_path.c_str());
  if (fopen(join("qlog_on").c_str(), "r") != nullptr)
    p->qlog = fopen((dir + "/../qnn_jobs.log").c_str(), "w");

  /* qnn_init sets ADSP_LIBRARY_PATH to skel_dir — it must be the native
   * library dir (where libQnnHtpV79Skel.so lives), NOT the models dir, or
   * the DSP-side search can't find the skel and deviceCreate fails 0x36b1. */
  if (qnn_init(lib_dir, lib_dir) != 0) {
    snprintf(p->err, sizeof(p->err), "qnn_init: %s", qnn_last_error());
    return -1;
  }
  struct { const char *tag, *file; } graphs[] = {
    {G_C0, "hubert_c0_s23.bin"},
    {G_R, "hubert_r_2063_s23.bin"},
    {G_ENC, "hubert_enc_140_s23.bin"},
    {G_RMVPE, "rmvpe_e2e_s23.bin"},
    {G_ENCFLOW, "encflow.bin"},
    {G_FLOW, "flowreverse.bin"},
    {G_DEC, "decnsf_ext.bin"},
  };
  for (const auto &gr : graphs) {
    std::string path = join(gr.file);
    if (qnn_load(gr.tag, path.c_str()) != 0) {
      snprintf(p->err, sizeof(p->err), "load %s: %s", gr.file, qnn_last_error());
      return -1;
    }
  }
  /* conv0's GroupNorm affine, staged as raw float32 (512 weights, 512 biases). */
  FILE *f = fopen(join("conv0_gn.f32").c_str(), "rb");
  if (!f) {
    snprintf(p->err, sizeof(p->err), "open %s", join("conv0_gn.f32").c_str());
    return -1;
  }
  if (fread(p->gn_w, sizeof(float), HUBERT_C0_CH, f) != (size_t)HUBERT_C0_CH ||
      fread(p->gn_b, sizeof(float), HUBERT_C0_CH, f) != (size_t)HUBERT_C0_CH) {
    snprintf(p->err, sizeof(p->err), "conv0_gn.f32 too short");
    fclose(f);
    return -1;
  }
  fclose(f);
  p->inited = true;
  return 0;
}

long rvc_pipeline_convert(rvc_pipeline_t *p, const float *x16k, size_t n,
                          float *out, size_t out_cap, rvc_stats_t *st) {
  double t_total = now_ms();
  memset(st, 0, sizeof(*st));
  st->n_in = (long)n;

  /* official highpass, whole-utterance filtfilt (pipeline.py:20) */
  std::vector<float> hp(n);
  double t0 = now_ms();
  rvc_highpass_48(x16k, n, hp.data());
  st->ms_highpass = now_ms() - t0;

  const int upp = p->voice.upp, sr = p->voice.sr;
  const int t_ph = p->voice.t_ph, t_out = p->voice.t_out;
  const int block_frame = CTX_STRIDE * upp;   /* 250 ms of output */
  const int sola_buffer = 4 * upp;            /* 40 ms */
  const int sola_search = upp;                /* 10 ms */

  /* 2.5 s of leading zero history: block i's window is padded samples
   * [4000i, 4000i+44960) and its emitted content is real samples
   * [4000i, 4000i+4800) — the pad makes the padded index of real sample k
   * exactly k/160, so advancing a block is a plain frame step. */
  std::vector<float> wav((size_t)PAD_SAMPLES + n);
  memcpy(wav.data() + PAD_SAMPLES, hp.data(), n * sizeof(float));

  long n_latent = ((long)n + (SR16 / 100) - 1) / (SR16 / 100);
  int n_blocks = (int)((n_latent + CTX_STRIDE - 1) / CTX_STRIDE);
  if (n_blocks < 1) n_blocks = 1;
  st->n_blocks = n_blocks;

  t0 = now_ms();
  std::vector<double> track;
  if (compute_f0_track(p, wav.data(), wav.size(), &track, &st->ms_f0) != 0)
    return -1;
  st->ms_f0 = now_ms() - t0;

  std::vector<float> feat;
  if (compute_hubert_ctx(p, &wav, n_blocks, &feat, &st->ms_c0, &st->ms_gn,
                         &st->ms_r, &st->ms_enc) != 0)
    return -1;

  /* extend the f0 track to the last block's window */
  long need = (long)(n_blocks - 1) * CTX_STRIDE + t_ph;
  if ((long)track.size() < need) {
    double last = track.empty() ? 0.0 : track.back();
    track.resize((size_t)need, last);
  }
  std::vector<double> cum((size_t)need + 1, 0.0);
  for (long i = 0; i < need; i++) cum[i + 1] = cum[i] + track[i];
  double pad_phase = cum[PAD_FRAMES];   /* the leading zeros' contribution */

  if (out_cap < (size_t)n_blocks * (size_t)block_frame) {
    snprintf(p->err, sizeof(p->err), "output buffer too small");
    return -1;
  }

  /* realtime_gui.py::fade_in_window — a SQUARED sine, not the plain one. */
  std::vector<float> fade_in(sola_buffer), buf(sola_buffer, 0.0f);
  for (int i = 0; i < sola_buffer; i++)
    fade_in[i] = (float)pow(sin(0.5 * M_PI * (double)i / (double)(sola_buffer - 1)), 2);

  std::vector<float> audio(t_out * upp);
  std::vector<float> seg(t_out * upp);
  size_t out_n = 0;
  for (int i = 0; i < n_blocks; i++) {
    int c0i = i * CTX_STRIDE;
    const float *phone = feat.data() + (size_t)i * t_ph * HUBERT_DIM;
    const double *f0b = track.data() + c0i;

    /* SineGen's phase runs across callbacks: anchor this block's harmonic
     * phase to the utterance start. */
    double phase_off = (cum[c0i + PAD_FRAMES] - pad_phase) * (double)upp;
    if (run_ctx_block(p, phone, f0b, phase_off, audio.data(), &st->ms_synth) != 0)
      return -1;

    /* SOLA: cross-correlate the incoming head against the stored tail, pick
     * the best offset in [0, sola_search], crossfade. */
    t0 = now_ms();
    int conv_len = sola_buffer + sola_search;
    std::vector<double> cv(conv_len);
    for (int k = 0; k < conv_len; k++) cv[k] = (double)audio[k];
    std::vector<double> cumsq(conv_len + 1, 0.0);
    for (int k = 0; k < conv_len; k++) cumsq[k + 1] = cumsq[k] + cv[k] * cv[k];
    int best_off = 0;
    double best_ratio = -1e300;
    for (int off = 0; off <= sola_search; off++) {
      double s = 0.0;
      for (int j = 0; j < sola_buffer; j++) s += cv[off + j] * (double)buf[j];
      double den = sqrt(cumsq[off + sola_buffer] - cumsq[off] + 1e-8);
      double r = den > 0.0 ? s / den : s;
      if (r > best_ratio) { best_ratio = r; best_off = off; }
    }
    int seg_len = (t_out * upp) - best_off;
    memcpy(seg.data(), audio.data() + best_off, (size_t)seg_len * sizeof(float));
    for (int j = 0; j < sola_buffer; j++)
      seg[j] = seg[j] * fade_in[j] + buf[j] * (1.0f - fade_in[j]);
    memcpy(buf.data(), seg.data() + block_frame,
           (size_t)sola_buffer * sizeof(float));
    memcpy(out + out_n, seg.data(), (size_t)block_frame * sizeof(float));
    out_n += (size_t)block_frame;
    st->ms_sola += now_ms() - t0;
  }

  /* the emitted length is the input duration at the model's rate */
  size_t target = (size_t)llround((double)n * (double)sr / (double)SR16);
  if (target > out_n) target = out_n;

  /* RVC loudness match: (rms_in / rms_out) ** (1 - rate), rate 0.25 */
  t0 = now_ms();
  std::vector<float> rms_buf(target);
  rvc_change_rms(hp.data(), n, out, target, SR16, sr, 0.25, rms_buf.data());
  st->ms_rms = now_ms() - t0;
  memcpy(out, rms_buf.data(), target * sizeof(float));

  /* official scales down only, never up (pipeline.py:402-405) */
  float peak = 0.0f;
  for (size_t i = 0; i < target; i++) {
    float a = fabsf(out[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0.99f) {
    float s = 0.99f / peak;
    for (size_t i = 0; i < target; i++) out[i] *= s;
  }

  st->n_out = (long)target;
  st->graph_calls = p->graph_calls;
  st->ms_total = now_ms() - t_total;
  st->rtf = st->ms_total / ((double)target / (double)sr * 1000.0);
  p->last_stats = *st;
  return (long)target;
}

/* ------------------------------------------------------------ streaming --- */

static void stream_publish_stats(rvc_pipeline_t *p, rvc_stats_t *st) {
  const int sr = p->voice.sr;
  memset(st, 0, sizeof(*st));
  st->ms_highpass = p->st_hp;
  st->ms_f0 = p->st_f0;
  st->ms_c0 = p->st_c0;
  st->ms_gn = p->st_gn;
  st->ms_r = p->st_r;
  st->ms_enc = p->st_enc;
  st->ms_index = p->st_index;
  st->ms_synth = p->st_synth;
  st->ms_sola = p->st_sola;
  st->n_blocks = p->st_blocks;
  st->n_in = p->st_in;
  st->n_out = p->st_out;
  st->graph_calls = p->graph_calls;
  st->ms_total = p->st_total;
  st->rtf = (st->n_out > 0)
                ? st->ms_total / ((double)st->n_out / (double)sr * 1000.0)
                : 0.0;
  st->next_block = p->next_block;
  st->next_prep = p->next_prep;
  st->n_real = (long)p->n_real;
  st->f0_windows = (long)p->f0_windows;
  st->f0_folded = (long)p->f0_folded;
  st->q_next = (unsigned long long)p->q_next;
  st->q_done = (unsigned long long)p->q_done;
  st->q_fail = p->q_fail ? 1 : 0;
  st->q_workers = p->q_workers.load();
  snprintf(st->dbg, sizeof(st->dbg), "%s%s%s | qcur=%s | ",
           p->q_err.c_str(), p->stall[0] ? " | " : "", p->stall, p->qcur);
  /* the last ~24 queue events: S=submit W=worker-done C=collect-enter
   * (front ticket) R=collect-result (rc).  0 as the front means empty. */
  size_t pos = strlen(st->dbg), total = p->qtrace_n < 64 ? p->qtrace_n : 64;
  size_t start = p->qtrace_n > 64 ? p->qtrace_n % 64 : 0;
  for (size_t k = 0; k < total && pos + 12 < sizeof(st->dbg); k++) {
    const auto &e = p->qtrace[(start + k) % 64];
    int m = snprintf(st->dbg + pos, sizeof(st->dbg) - pos, "%c%llu:%llu ",
                     e.op, (unsigned long long)e.a, (unsigned long long)e.b);
    if (m <= 0) break;
    pos += (size_t)m;
  }
  p->last_stats = *st;
}

int rvc_pipeline_start_stream(rvc_pipeline_t *p) {
  if (!p->inited) {
    snprintf(p->err, sizeof(p->err), "pipeline not initialised");
    return -1;
  }
  const int upp = p->voice.upp;
  const int sola_buffer = 4 * upp;

  p->stream_on = true;
  p->sig.assign((size_t)PAD_SAMPLES, 0.0f);
  p->n_real = 0;
  p->hpv.clear();
  p->hp_lo = 0;
  p->f0_obs.clear();
  p->f0_best.clear();
  p->f0_track.clear();
  p->f0_cum.clear();
  p->f0_frames = 0;
  p->f0_windows = 0;
  p->f0_folded = 0;
  p->chunks.clear();
  p->next_block = 0;
  p->next_prep = 0;

  p->c0_buf.assign((size_t)HUBERT_C0_CH * (size_t)HUBERT_T1_LEN, 0.0f);
  for (int k = 0; k < 2; k++) {
    p->gn_out[k].assign((size_t)HUBERT_C0_CH * (size_t)HUBERT_T1_LEN, 0.0f);
    p->cnn[k].assign((size_t)HUBERT_T2_LEN * (size_t)HUBERT_C0_CH, 0.0f);
    p->feat[k].assign((size_t)CTX_FRAMES * (size_t)HUBERT_DIM, 0.0f);
  }
  p->audio.assign((size_t)p->voice.t_out * upp, 0.0f);
  p->seg.assign((size_t)p->voice.t_out * upp, 0.0f);
  p->sola_buf.assign((size_t)sola_buffer, 0.0f);
  p->fade_in.assign((size_t)sola_buffer, 0.0f);
  for (int i = 0; i < sola_buffer; i++)
    p->fade_in[i] =
        (float)pow(sin(0.5 * M_PI * (double)i / (double)(sola_buffer - 1)), 2);

  p->st_total = p->st_hp = p->st_f0 = p->st_c0 = p->st_gn = p->st_r = 0;
  p->st_enc = p->st_synth = p->st_sola = p->st_index = 0;
  p->st_blocks = p->st_in = p->st_out = 0;

  p->q_next = p->q_done = 0;
  p->q_stop = p->q_fail = false;
  p->q_err.clear();
  p->pending.clear();
  p->done.clear();
  p->f0_job_tickets.clear();
  p->c0_pending.clear();
  p->blk_f0_last.clear();
  p->blk_c0_last.clear();
  p->npu_thr = std::thread(npu_worker, p);
  return 0;
}

long rvc_pipeline_feed_stream(rvc_pipeline_t *p, const float *x, size_t n,
                              float *out, size_t out_cap, rvc_stats_t *st) {
  const int block_frame = CTX_STRIDE * p->voice.upp;
  size_t out_n = 0;
  if (!p->stream_on) {
    snprintf(p->err, sizeof(p->err), "stream not started");
    goto done;
  }
  if (p->q_fail) {
    snprintf(p->err, sizeof(p->err), "%s", p->q_err.c_str());
    goto done;
  }
  if (n > 0) {
    p->sig.resize(p->sig.size() + n);
    memcpy(p->sig.data() + (size_t)PAD_SAMPLES + p->n_real, x, n * sizeof(float));
    p->n_real += n;
    p->st_in += (long)n;
  }

  for (;;) {
    /* PREP: submit the NPU work for every block whose input has arrived, up
     * to PREP_AHEAD past the one being emitted.  Topping up inside the output
     * loop (not once per feed) is what keeps the worker's queue from draining
     * mid-feed: without it, blocks 2 and 3 of a batch run fully serialized. */
    while (p->next_prep <= p->next_block + PREP_AHEAD) {
      int j = p->next_prep;
      if (p->n_real < stream_need_raw(p, j)) break;
      if (stream_prep_block(p, j) != 0) {
        if (p->q_err.size()) snprintf(p->err, sizeof(p->err), "%s", p->q_err.c_str());
        goto done;
      }
      p->next_prep++;
    }

    int i = p->next_block;
    if (p->n_real < stream_need_raw(p, i)) {
      snprintf(p->stall, sizeof(p->stall),
               "out: need_raw(i=%d)=%zu > n_real=%zu", i,
               stream_need_raw(p, i), p->n_real);
      break;
    }
    if (out_n + (size_t)block_frame > out_cap) {
      snprintf(p->stall, sizeof(p->stall), "out: cap %zu full", out_cap);
      break;                                            /* caller drains */
    }
    if (p->next_prep <= i) {
      snprintf(p->stall, sizeof(p->stall),
               "out: prep capped (next_prep=%d <= i=%d)", p->next_prep, i);
      break;
    }

    double t_blk = now_ms();
    size_t g = (size_t)i * (size_t)CTX_STRIDE + (size_t)p->voice.t_ph - 1;
    size_t need_w = (g >= 10) ? g / 10 - 1 : 0;
    int par = i & 1;
    if (stream_f0_fold(p, need_w) != 0) goto fail;
    stream_f0_finalize(p);
    if (stream_c0_collect(p, i) != 0) goto fail;
    stream_c0_gather(p, i);
    if (stream_out_features(p, i, par) != 0) goto fail;
    if (stream_synth(p, i, par, out + out_n) != 0) goto fail;
    out_n += (size_t)block_frame;
    p->next_block++;
    p->st_blocks++;
    p->st_out += (long)block_frame;
    p->st_total += now_ms() - t_blk;
    p->stall[0] = '\0';     /* emitted: the previous stall no longer applies */
  }
done:
  stream_publish_stats(p, st);
  return (long)out_n;
fail:
  if (p->q_err.size()) snprintf(p->err, sizeof(p->err), "%s", p->q_err.c_str());
  stream_publish_stats(p, st);
  return -1;
}

long rvc_pipeline_end_stream(rvc_pipeline_t *p, float *out, size_t out_cap,
                             rvc_stats_t *st) {
  if (!p->stream_on) return 0;
  /* pad zeros so the pending block's look-ahead window fills and it emits;
   * the trailing output beyond the real input duration is synthetic. */
  size_t need = stream_need_raw(p, p->next_block);
  if (p->n_real < need) {
    p->sig.resize((size_t)PAD_SAMPLES + need, 0.0f);
    p->n_real = need;
  }
  long n = rvc_pipeline_feed_stream(p, nullptr, 0, out, out_cap, st);
  p->stream_on = false;
  p->stop_npu();
  return n;
}
