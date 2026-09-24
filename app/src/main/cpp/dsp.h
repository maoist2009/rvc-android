/* dsp.h — the host-side signal processing the RVC pipeline needs.
 *
 * Every routine here is a direct port of a numpy/scipy function in
 * run_rvc_npu.py, kept numerically faithful because the graphs were compiled
 * against the values those functions produce.
 */
#ifndef RVC_DSP_H
#define RVC_DSP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* N_FFT / hop / mels of the RMVPE mel front end, and the RMVPE frame count. */
#define RVC_N_FFT        1024
#define RVC_N_FFT_HOP    160
#define RVC_N_MELS       128
#define RVC_F_MIN        30.0f
#define RVC_F_MAX        8000.0f
#define RVC_SR_16K       16000
#define RVC_RMVPE_FRAMES 32
#define RVC_T_LATENT     30

typedef struct {
  float *mel_w;      /* (RVC_N_MELS, RVC_N_FFT/2+1), row-major */
  float *hann;       /* RVC_N_FFT */
} rvc_dsp_t;

/* Builds the HTK mel filterbank and the Hann window. Returns 0 on success. */
int rvc_dsp_init(rvc_dsp_t *d);
void rvc_dsp_free(rvc_dsp_t *d);

/* compute_mel_128x32: x is n samples @16k. Writes min(n_frames, 32)*128
 * floats to out (row-major, time-major). Returns the frame count. */
int rvc_mel_128x32(const rvc_dsp_t *d, const float *x, size_t n, float *out);

/* scipy.signal.butter(5, 48Hz, high, fs=16000) + filtfilt (odd pad, padlen 18).
 * out may alias x. */
void rvc_highpass_48(const float *x, size_t n, float *out);

/* librosa.feature.rms(y, frame_length, hop, center=True, zero pad).
 * out must hold 1 + (n + frame_length - hop) / hop entries... precisely
 * 1 + (n_padded - frame_length) / hop where n_padded = n + frame_length.
 * Writes the count to *out_n. */
void rvc_rms_envelope(const double *y, size_t n, int frame_length, int hop,
                      double *out, size_t *out_n);

/* torch.nn.functional.interpolate(mode='linear', align_corners=False). */
void rvc_interp_linear(const double *src, size_t n_src, double *out,
                       size_t n_out);

/* np.interp: linear with endpoint clamping. xp must be increasing. */
double rvc_np_interp(double x, const double *xp, const double *fp, size_t n);

/* change_rms: multiplies out_audio by (rms_in / rms_out) ** (1 - rate). */
void rvc_change_rms(const float *in_audio, size_t n_in,
                    const float *out_audio, size_t n_out,
                    int sr_in, int sr_out, double rate, float *result);

#ifdef __cplusplus
}
#endif

#endif /* RVC_DSP_H */
