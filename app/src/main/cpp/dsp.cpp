/* dsp.cpp — host-side DSP, ported function-for-function from run_rvc_npu.py.
 */
#include "dsp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ FFT --- */

/* Iterative radix-2 Cooley-Tukey, in place, interleaved complex.
 * np.fft.rfft(x) == the first n/2+1 bins of the complex FFT of (x, 0), so a
 * real input is handled by zeroing the imaginary part. */
static void fft_radix2(double *re, double *im, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double tr = re[i]; re[i] = re[j]; re[j] = tr;
      double ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = -2.0 * M_PI / (double)len;
    double wr = cos(ang), wi = sin(ang);
    for (int i = 0; i < n; i += len) {
      double cr = 1.0, ci = 0.0;
      for (int k = 0; k < (len >> 1); k++) {
        int a = i + k, b = i + k + (len >> 1);
        double tr = cr * re[b] - ci * im[b];
        double ti = cr * im[b] + ci * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;         im[a] += ti;
        double ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/* ------------------------------------------------------------- mel bank --- */

static double hz_to_mel_htk(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz_htk(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

/* _mel_filter_htk(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000). */
int rvc_dsp_init(rvc_dsp_t *d) {
  const int n_freqs = RVC_N_FFT / 2 + 1;
  double *w = (double *)calloc((size_t)RVC_N_MELS * n_freqs, sizeof(double));
  float *fw = (float *)malloc((size_t)RVC_N_MELS * n_freqs * sizeof(float));
  float *hann = (float *)malloc((size_t)RVC_N_FFT * sizeof(float));
  if (!w || !fw || !hann) {
    free(w); free(fw); free(hann);
    return -1;
  }
  double fft_freqs[513];
  for (int i = 0; i < n_freqs; i++)
    fft_freqs[i] = (double)RVC_SR_16K / 2.0 * (double)i / (double)(n_freqs - 1);

  double mel_min = hz_to_mel_htk(RVC_F_MIN), mel_max = hz_to_mel_htk(RVC_F_MAX);
  double hz_pts[130];
  for (int i = 0; i < RVC_N_MELS + 2; i++)
    hz_pts[i] = mel_to_hz_htk(mel_min + (mel_max - mel_min) * (double)i /
                              (double)(RVC_N_MELS + 1));
  long bin_pts[130];
  for (int i = 0; i < RVC_N_MELS + 2; i++)
    bin_pts[i] = (long)floor((double)(RVC_N_FFT + 1) * hz_pts[i] / (double)RVC_SR_16K);

  for (int m = 1; m <= RVC_N_MELS; m++) {
    long lo = bin_pts[m - 1], mid = bin_pts[m], hi = bin_pts[m + 1];
    if (mid == lo || mid == hi) continue;
    double *row = w + (size_t)(m - 1) * n_freqs;
    if (mid > lo)
      for (long k = lo; k < mid; k++)
        row[k] = ((double)k - (double)bin_pts[m - 1]) /
                 ((double)bin_pts[m] - (double)bin_pts[m - 1]);
    if (hi > mid)
      for (long k = mid; k < hi; k++)
        row[k] = ((double)bin_pts[m + 1] - (double)k) /
                 ((double)bin_pts[m + 1] - (double)bin_pts[m]);
  }
  for (int i = 0; i < RVC_N_MELS * n_freqs; i++) fw[i] = (float)w[i];
  free(w);

  for (int i = 0; i < RVC_N_FFT; i++)
    hann[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (double)(RVC_N_FFT - 1))));

  d->mel_w = fw;
  d->hann = hann;
  return 0;
}

void rvc_dsp_free(rvc_dsp_t *d) {
  free(d->mel_w); free(d->hann);
  d->mel_w = NULL; d->hann = NULL;
}

/* compute_mel_128x32. Returns the number of frames written (<= 32). */
int rvc_mel_128x32(const rvc_dsp_t *d, const float *x, size_t n, float *out) {
  const int pad = RVC_N_FFT / 2;
  /* reflect-pad by N_FFT//2 on each side (librosa center=True). np.pad
   * 'reflect' mirrors without duplicating the edge sample. */
  size_t m = n;
  if (m < (size_t)RVC_N_FFT) m = (size_t)RVC_N_FFT;   /* zero-pad a short input */
  size_t n_pad = m + 2 * (size_t)pad;
  float *xp = (float *)malloc(n_pad * sizeof(float));
  if (!xp) return 0;
  memset(xp, 0, n_pad * sizeof(float));
  memcpy(xp + pad, x, n * sizeof(float));            /* x may be shorter than m */
  for (int i = 0; i < pad; i++) {
    xp[pad - 1 - i] = x[i + 1];                       /* left: x[1..pad] reversed */
    if (n > (size_t)(i + 2))
      xp[pad + n + i] = x[n - 2 - i];                 /* right: x[n-2..n-pad-1] reversed */
  }

  int n_frames = 1 + (int)((n_pad - (size_t)RVC_N_FFT) / (size_t)RVC_N_FFT_HOP);
  if (n_frames > RVC_RMVPE_FRAMES) n_frames = RVC_RMVPE_FRAMES;

  double *re = (double *)malloc((size_t)RVC_N_FFT * sizeof(double));
  double *im = (double *)malloc((size_t)RVC_N_FFT * sizeof(double));
  double *mel = (double *)malloc((size_t)n_frames * RVC_N_MELS * sizeof(double));
  if (!re || !im || !mel) {
    free(xp); free(re); free(im); free(mel);
    return 0;
  }
  const int n_freqs = RVC_N_FFT / 2 + 1;
  double *prow = (double *)malloc((size_t)n_freqs * sizeof(double));
  if (!prow) { free(xp); free(re); free(im); free(mel); return 0; }

  for (int f = 0; f < n_frames; f++) {
    const float *fr = xp + (size_t)f * RVC_N_FFT_HOP;
    for (int i = 0; i < RVC_N_FFT; i++) {
      re[i] = (double)(fr[i] * d->hann[i]);
      im[i] = 0.0;
    }
    fft_radix2(re, im, RVC_N_FFT);
    for (int i = 0; i < n_freqs; i++)
      prow[i] = re[i] * re[i] + im[i] * im[i];
    double *mrow = mel + (size_t)f * RVC_N_MELS;
    for (int m_i = 0; m_i < RVC_N_MELS; m_i++) {
      const float *wrow = d->mel_w + (size_t)m_i * n_freqs;
      double s = 0.0;
      for (int i = 0; i < n_freqs; i++) s += prow[i] * (double)wrow[i];
      double v = s;
      if (v < 1e-10) v = 1e-10;
      mrow[m_i] = log(v);
    }
  }
  for (int i = 0; i < n_frames * RVC_N_MELS; i++) out[i] = (float)mel[i];

  free(xp); free(re); free(im); free(prow); free(mel);
  return n_frames;
}

/* ------------------------------------------------------------ highpass --- */

/* scipy.signal.lfilter, direct form II transposed, double. */
static void lfilter_d(const double *b, int nb, const double *a, int na,
                      const double *x, double *y, size_t n) {
  double *z = (double *)calloc((size_t)(nb > na ? nb : na), sizeof(double));
  for (size_t i = 0; i < n; i++) {
    double xn = x[i];
    double yn = b[0] * xn + z[0];
    for (int k = 1; k < nb; k++) z[k - 1] = b[k] * xn + z[k] - a[k] * yn;
    for (int k = nb; k < na; k++) z[k - 1] = z[k] - a[k] * yn;
    y[i] = yn;
  }
  free(z);
}

/* scipy.signal.filtfilt with padtype='odd', padlen = 3*max(len(a), len(b)). */
void rvc_highpass_48(const float *x, size_t n, float *out) {
  /* butter(5, 48, 'high', fs=16000), computed once offline. */
  static const double b[6] = {0.9699606451838447, -4.849803225919223,
                              9.699606451838447, -9.699606451838447,
                              4.849803225919223, -0.9699606451838447};
  static const double a[6] = {1.0, -4.939001819168364, 9.757863526739541,
                              -9.63954484941346, 4.76150679735621,
                              -0.9408236532054606};
  const int padlen = 18;   /* 3 * max(len(a), len(b)) */
  if (n <= (size_t)padlen) {   /* scipy refuses; mirror the caller's contract */
    memcpy(out, x, n * sizeof(float));
    return;
  }
  size_t ext_n = n + 2 * (size_t)padlen;
  double *ext = (double *)malloc(ext_n * sizeof(double));
  double *tmp = (double *)malloc(ext_n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  if (!ext || !tmp || !y) { free(ext); free(tmp); free(y); return; }

  double x0 = (double)x[0], xL = (double)x[n - 1];
  for (int i = 0; i < padlen; i++)
    ext[i] = 2.0 * x0 - (double)x[padlen - i];        /* x[18..1] mirrored */
  for (size_t i = 0; i < n; i++) ext[padlen + i] = (double)x[i];
  for (int i = 0; i < padlen; i++)
    ext[padlen + n + i] = 2.0 * xL - (double)x[n - 2 - i];

  lfilter_d(b, 6, a, 6, ext, tmp, ext_n);
  /* reverse, filter, reverse */
  for (size_t i = 0; i < ext_n; i++) ext[ext_n - 1 - i] = tmp[i];
  lfilter_d(b, 6, a, 6, ext, tmp, ext_n);
  for (size_t i = 0; i < ext_n; i++) ext[ext_n - 1 - i] = tmp[i];

  for (size_t i = 0; i < n; i++) y[i] = ext[padlen + i];
  for (size_t i = 0; i < n; i++) out[i] = (float)y[i];

  free(ext); free(tmp); free(y);
}

/* --------------------------------------------------------------- rms ----- */

void rvc_rms_envelope(const double *y, size_t n, int frame_length, int hop,
                      double *out, size_t *out_n) {
  size_t pad = (size_t)frame_length / 2;
  size_t np = n + 2 * pad;
  double *yp = (double *)calloc(np, sizeof(double));
  if (!yp) { *out_n = 0; return; }
  memcpy(yp + pad, y, n * sizeof(double));
  size_t n_frames = 1 + (np - (size_t)frame_length) / (size_t)hop;
  double *cs = (double *)malloc((np + 1) * sizeof(double));
  if (!cs) { free(yp); *out_n = 0; return; }
  cs[0] = 0.0;
  for (size_t i = 0; i < np; i++) cs[i + 1] = cs[i] + yp[i] * yp[i];
  for (size_t f = 0; f < n_frames; f++) {
    size_t s = f * (size_t)hop;
    double v = (cs[s + frame_length] - cs[s]) / (double)frame_length;
    out[f] = sqrt(v > 0.0 ? v : 0.0);
  }
  *out_n = n_frames;
  free(yp); free(cs);
}

void rvc_interp_linear(const double *src, size_t n_src, double *out,
                       size_t n_out) {
  if (n_src == n_out) { memcpy(out, src, n_out * sizeof(double)); return; }
  for (size_t i = 0; i < n_out; i++) {
    double pos = ((double)i + 0.5) * ((double)n_src / (double)n_out) - 0.5;
    if (pos < 0.0) pos = 0.0;
    if (pos > (double)(n_src - 1)) pos = (double)(n_src - 1);
    size_t j = (size_t)pos;
    double frac = pos - (double)j;
    if (j + 1 >= n_src) { out[i] = src[n_src - 1]; continue; }
    out[i] = src[j] * (1.0 - frac) + src[j + 1] * frac;
  }
}

double rvc_np_interp(double x, const double *xp, const double *fp, size_t n) {
  if (n == 0) return 0.0;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  size_t lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    size_t mid = lo + (hi - lo) / 2;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  double t = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] * (1.0 - t) + fp[hi] * t;
}

void rvc_change_rms(const float *in_audio, size_t n_in,
                    const float *out_audio, size_t n_out,
                    int sr_in, int sr_out, double rate, float *result) {
  /* frame_length = sr (1 s window), hop = sr/2 (0.5 s stride) on each side. */
  int fl_in = sr_in, hop_in = sr_in / 2;
  int fl_out = sr_out, hop_out = sr_out / 2;
  double *in_d = (double *)malloc(n_in * sizeof(double));
  double *out_d = (double *)malloc(n_out * sizeof(double));
  size_t cap = n_in / (size_t)hop_in + 4;
  double *env1 = (double *)malloc(cap * sizeof(double));
  double *env2 = (double *)malloc((n_out / (size_t)hop_out + 4) * sizeof(double));
  double *e1 = (double *)malloc(n_out * sizeof(double));
  double *e2 = (double *)malloc(n_out * sizeof(double));
  if (!in_d || !out_d || !env1 || !env2 || !e1 || !e2) {
    free(in_d); free(out_d); free(env1); free(env2); free(e1); free(e2);
    memcpy(result, out_audio, n_out * sizeof(float));
    return;
  }
  for (size_t i = 0; i < n_in; i++) in_d[i] = (double)in_audio[i];
  for (size_t i = 0; i < n_out; i++) out_d[i] = (double)out_audio[i];

  size_t n1, n2;
  rvc_rms_envelope(in_d, n_in, fl_in, hop_in, env1, &n1);
  rvc_rms_envelope(out_d, n_out, fl_out, hop_out, env2, &n2);
  rvc_interp_linear(env1, n1, e1, n_out);
  rvc_interp_linear(env2, n2, e2, n_out);

  for (size_t i = 0; i < n_out; i++) {
    double r2 = e2[i] < 1e-6 ? 1e-6 : e2[i];
    double g = pow(e1[i], 1.0 - rate) * pow(r2, rate - 1.0);
    result[i] = (float)(out_d[i] * g);
  }
  free(in_d); free(out_d); free(env1); free(env2); free(e1); free(e2);
}
