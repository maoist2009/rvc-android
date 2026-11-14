/* rvc_jni.cpp — Kotlin <-> qnn_executor bridge.
 *
 * Inputs reach the graphs as FloatArray (FLOAT_32) or IntArray (INT_32); the
 * native side validates name and dtype against the .bin metadata and refuses a
 * mismatch rather than miscasting. All RVC graph outputs are FLOAT_32, so the
 * exec result is an array of FloatArray in graph declaration order.
 */
#include <jni.h>
#include <string>
#include <vector>

#include "qnn_executor.h"
#include "index_retrieval.h"
#include "rvc_pipeline.h"

#define JNI_FUNC(ret, name) \
  extern "C" JNIEXPORT ret JNICALL Java_com_dreamlite_rvc_RvcJni_##name

static std::string jstr(JNIEnv *env, jstring s) {
  if (!s) return std::string();
  const char *c = env->GetStringUTFChars(s, nullptr);
  std::string r(c ? c : "");
  if (c) env->ReleaseStringUTFChars(s, c);
  return r;
}

/* ---------- graph metadata -> JSON ---------- */

struct JsonCtx {
  std::string json;
  bool first = true;
};

static void emit_tensor(void *user, const char *name, int dtype, uint32_t rank,
                        const uint32_t *dims) {
  JsonCtx *c = static_cast<JsonCtx *>(user);
  const char *dt = (dtype == RVC_QNN_FLOAT32) ? "float32" : "int32";
  c->json += (c->first ? "" : ",");
  c->first = false;
  c->json += "{\"name\":\"";
  c->json += name;
  c->json += "\",\"dtype\":\"";
  c->json += dt;
  c->json += "\",\"dims\":[";
  for (uint32_t i = 0; i < rank; i++) {
    c->json += (i ? "," : "");
    c->json += std::to_string(dims[i]);
  }
  c->json += "]}";
}

/* ---------- lifecycle ---------- */

JNI_FUNC(jstring, nativeInit)(JNIEnv *env, jobject, jstring libDir,
                              jstring skelDir) {
  std::string lib = jstr(env, libDir);
  std::string skel = jstr(env, skelDir);
  if (qnn_init(lib.c_str(), skel.c_str()) != 0)
    return env->NewStringUTF(qnn_last_error());
  return env->NewStringUTF("");
}

JNI_FUNC(jstring, loadGraph)(JNIEnv *env, jobject, jstring tag, jstring binPath) {
  std::string t = jstr(env, tag);
  std::string p = jstr(env, binPath);
  if (qnn_load(t.c_str(), p.c_str()) != 0)
    return env->NewStringUTF(qnn_last_error());
  return env->NewStringUTF("");
}

JNI_FUNC(jstring, loadGraphDlc)(JNIEnv *env, jobject, jstring tag, jstring dlcPath) {
  std::string t = jstr(env, tag);
  std::string p = jstr(env, dlcPath);
  if (qnn_load_dlc(t.c_str(), p.c_str()) != 0)
    return env->NewStringUTF(qnn_last_error());
  return env->NewStringUTF("");
}

JNI_FUNC(jstring, freeGraph)(JNIEnv *env, jobject, jstring tag) {
  std::string t = jstr(env, tag);
  if (qnn_free(t.c_str()) != 0)
    return env->NewStringUTF(qnn_last_error());
  return env->NewStringUTF("");
}

JNI_FUNC(jstring, graphInfo)(JNIEnv *env, jobject, jstring tag) {
  std::string t = jstr(env, tag);
  JsonCtx in, out;
  /* in_cb only — passing emit_tensor for out_cb too would append the outputs
   * to the "inputs" array, and every consumer would over-feed the graph. */
  if (qnn_describe(t.c_str(), emit_tensor, nullptr, &in) != 0)
    return env->NewStringUTF(qnn_last_error());
  if (qnn_describe(t.c_str(), nullptr, emit_tensor, &out) != 0)
    return env->NewStringUTF(qnn_last_error());
  std::string json = "{\"tag\":\"" + t + "\",\"inputs\":[" + in.json +
                     "],\"outputs\":[" + out.json + "]}";
  return env->NewStringUTF(json.c_str());
}

JNI_FUNC(void, shutdown)(JNIEnv *, jobject) { qnn_shutdown(); }

JNI_FUNC(jstring, lastError)(JNIEnv *env, jobject) {
  return env->NewStringUTF(qnn_last_error());
}

/* ---------- execution ---------- */

JNI_FUNC(jobjectArray, execGraph)(JNIEnv *env, jobject, jstring tag,
                                  jobjectArray names, jobjectArray data) {
  std::string t = jstr(env, tag);
  jsize n = env->GetArrayLength(names);
  if (n != env->GetArrayLength(data)) {
    return env->NewObjectArray(0, env->FindClass("[F"), nullptr);
  }

  std::vector<std::string>  name_strs(n);
  std::vector<const char *> name_ptrs(n);
  std::vector<int>          dtypes(n);
  std::vector<const void *> ptrs(n);
  std::vector<size_t>       elems(n);
  std::vector<jfloatArray>  float_refs(n);
  std::vector<jintArray>    int_refs(n);
  jboolean is_copy = JNI_FALSE;

  for (jsize i = 0; i < n; i++) {
    jstring js = (jstring)env->GetObjectArrayElement(names, i);
    name_strs[i] = jstr(env, js);
    name_ptrs[i] = name_strs[i].c_str();
    env->DeleteLocalRef(js);

    jobject el = env->GetObjectArrayElement(data, i);
    if (env->IsInstanceOf(el, env->FindClass("[F"))) {
      jfloatArray fa = (jfloatArray)el;
      float_refs[i] = fa;
      jfloat *p = env->GetFloatArrayElements(fa, &is_copy);
      dtypes[i] = RVC_QNN_FLOAT32;
      ptrs[i] = p;
      elems[i] = (size_t)env->GetArrayLength(fa);
    } else if (env->IsInstanceOf(el, env->FindClass("[I"))) {
      jintArray ia = (jintArray)el;
      int_refs[i] = ia;
      jint *p = env->GetIntArrayElements(ia, &is_copy);
      dtypes[i] = RVC_QNN_INT32;
      ptrs[i] = p;
      elems[i] = (size_t)env->GetArrayLength(ia);
    } else {
      for (jsize k = 0; k <= i; k++) {
        if (float_refs[k]) env->ReleaseFloatArrayElements(float_refs[k], (jfloat *)ptrs[k], JNI_ABORT);
        if (int_refs[k]) env->ReleaseIntArrayElements(int_refs[k], (jint *)ptrs[k], JNI_ABORT);
      }
      return env->NewObjectArray(0, env->FindClass("[F"), nullptr);
    }
    /* el and float_refs[i]/int_refs[i] are the same local reference — it must
     * stay live until the matching Release*ArrayElements below. */
  }

  float  *out_data[16];
  size_t  out_elems[16];
  size_t  n_out = 0;
  int rc = qnn_exec(t.c_str(), name_ptrs.data(), dtypes.data(), ptrs.data(),
                    elems.data(), (size_t)n, out_data, out_elems, 16, &n_out);

  for (jsize i = 0; i < n; i++) {
    if (float_refs[i]) env->ReleaseFloatArrayElements(float_refs[i], (jfloat *)ptrs[i], JNI_ABORT);
    if (int_refs[i]) env->ReleaseIntArrayElements(int_refs[i], (jint *)ptrs[i], JNI_ABORT);
  }
  for (jsize i = 0; i < n; i++) {
    if (float_refs[i]) env->DeleteLocalRef(float_refs[i]);
    if (int_refs[i]) env->DeleteLocalRef(int_refs[i]);
  }

  if (rc != 0) {
    for (size_t i = 0; i < n_out; i++) free(out_data[i]);
    return env->NewObjectArray(0, env->FindClass("[F"), nullptr);
  }

  jclass fa_class = env->FindClass("[F");
  jobjectArray result = env->NewObjectArray((jsize)n_out, fa_class, nullptr);
  for (size_t i = 0; i < n_out; i++) {
    jfloatArray fa = env->NewFloatArray((jsize)out_elems[i]);
    env->SetFloatArrayRegion(fa, 0, (jsize)out_elems[i], out_data[i]);
    env->SetObjectArrayElement(result, (jsize)i, fa);
    env->DeleteLocalRef(fa);
    free(out_data[i]);
  }
  return result;
}

/* ---------- full pipeline ---------- */

static rvc_pipeline_t *g_pipeline = nullptr;

JNI_FUNC(jstring, pipelineCreate)(JNIEnv *env, jobject) {
  if (g_pipeline) rvc_pipeline_destroy(g_pipeline);
  g_pipeline = rvc_pipeline_create();
  return env->NewStringUTF(g_pipeline ? "" : "out of memory");
}

JNI_FUNC(jstring, pipelineInit)(JNIEnv *env, jobject, jstring modelsDir,
                                jstring libDir, jstring voiceJson) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  std::string dir = jstr(env, modelsDir);
  std::string lib = jstr(env, libDir);
  std::string json = jstr(env, voiceJson);

  /* Defaults are the mygf_zbhy ctx-280 build (models/synth_v2_ctx/
   * mygf_zbhy_gate.json) — the only voice compiled with the official realtime
   * t_ph=280 layout.  A voice manifest overrides any of these. */
  rvc_voice_t v;
  memset(&v, 0, sizeof(v));
  v.sr = 40000;
  v.upp = 400;
  v.t_ph = 280;
  v.t_out = 30;
  v.flow_head = 226;
  v.dec_head = 24;
  v.t_flow = 54;
  v.n_spk = 109;
  v.sid = 0;
  v.l_w = -0.826171875f;
  v.l_b = 0.002140045166015625f;
  v.sine_amp = 0.1f;
  v.index_rate = 0.0f;    /* no .index staged -> retrieval off */
  v.nprobe = 0;           /* 0 = the value stored in the .index file */
  v.pitch_mult = 1.0;     /* no transpose until the user sets one */

  /* the voice parameters come from the voice's manifest; absent keys keep the
   * defaults above, which match the v2 40k voice used during bring-up. */
  auto find_num = [&](const char *key) -> double {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = json.find(pat);
    if (p == std::string::npos) return HUGE_VAL;
    p = json.find(':', p);
    if (p == std::string::npos) return HUGE_VAL;
    return strtod(json.c_str() + p + 1, nullptr);
  };
  double val;
  if ((val = find_num("sr")) != HUGE_VAL) v.sr = (int)val;
  if ((val = find_num("upp")) != HUGE_VAL) v.upp = (int)val;
  if ((val = find_num("t_ph")) != HUGE_VAL) v.t_ph = (int)val;
  if ((val = find_num("t_out")) != HUGE_VAL) v.t_out = (int)val;
  if ((val = find_num("flow_head")) != HUGE_VAL) v.flow_head = (int)val;
  if ((val = find_num("dec_head")) != HUGE_VAL) v.dec_head = (int)val;
  if ((val = find_num("t_flow")) != HUGE_VAL) v.t_flow = (int)val;
  if ((val = find_num("n_spk")) != HUGE_VAL) v.n_spk = (int)val;
  if ((val = find_num("sid")) != HUGE_VAL) v.sid = (int)val;
  if ((val = find_num("l_w")) != HUGE_VAL) v.l_w = (float)val;
  if ((val = find_num("l_b")) != HUGE_VAL) v.l_b = (float)val;
  if ((val = find_num("sine_amp")) != HUGE_VAL) v.sine_amp = (float)val;
  if ((val = find_num("index_rate")) != HUGE_VAL) v.index_rate = (float)val;
  if ((val = find_num("nprobe")) != HUGE_VAL) v.nprobe = (int)val;
  v.rms_mix_rate = 0.25;           /* official default until the manifest says otherwise */
  v.protect = 0.5;                 /* 0.5 = off in the official client */
  v.filter_radius = 3;
  v.f0_margin = 0;                 /* realtime: earliest f0 window (low latency) */
  if ((val = find_num("rms_mix_rate")) != HUGE_VAL) v.rms_mix_rate = val;
  if ((val = find_num("protect")) != HUGE_VAL) v.protect = val;
  if ((val = find_num("filter_radius")) != HUGE_VAL) v.filter_radius = (int)val;

  /* index_rate > 0 turns on the retrieval step; the index file is resolved
   * relative to the models dir. */
  if (v.index_rate > 0.0f) {
    const char *e = rvc_pipeline_load_index(g_pipeline, dir.c_str(), &v);
    if (e && e[0]) return env->NewStringUTF(e);
  }

  if (rvc_pipeline_init(g_pipeline, dir.c_str(), lib.c_str(), &v) != 0)
    return env->NewStringUTF(rvc_pipeline_last_error(g_pipeline));
  return env->NewStringUTF("");
}

/* Runtime pitch transpose in semitones; "" on success. Takes effect on the
 * next emitted block — no graph reload, unlike an index_rate change. */
JNI_FUNC(jstring, pipelineSetPitch)(JNIEnv *env, jobject, jfloat semitones) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  rvc_pipeline_set_pitch(g_pipeline, (double)semitones);
  return env->NewStringUTF("");
}

/* Runtime overrides for the three official tunables; like pitch they apply
 * to the next emitted block without a graph reload. */
JNI_FUNC(jstring, pipelineSetRmsMixRate)(JNIEnv *env, jobject, jfloat rate) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  rvc_pipeline_set_rms_mix_rate(g_pipeline, (double)rate);
  return env->NewStringUTF("");
}

JNI_FUNC(jstring, pipelineSetProtect)(JNIEnv *env, jobject, jfloat protect) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  rvc_pipeline_set_protect(g_pipeline, (double)protect);
  return env->NewStringUTF("");
}

JNI_FUNC(jstring, pipelineSetFilterRadius)(JNIEnv *env, jobject, jint radius) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  rvc_pipeline_set_filter_radius(g_pipeline, (int)radius);
  return env->NewStringUTF("");
}

/* Realtime f0 lookahead margin: 0 = earliest 100 ms window (low latency),
 * 1 = one window later (steadier block-tail f0, +100 ms). Applies to the
 * next blocks; no graph reload, like pitch. */
JNI_FUNC(jstring, pipelineSetF0Margin)(JNIEnv *env, jobject, jint margin) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  rvc_pipeline_set_f0_margin(g_pipeline, (int)margin);
  return env->NewStringUTF("");
}

/* Returns the converted audio, or an empty array (lastError has the text). */
JNI_FUNC(jfloatArray, pipelineConvert)(JNIEnv *env, jobject, jfloatArray inAudio) {
  if (!g_pipeline) return env->NewFloatArray(0);
  jsize n = env->GetArrayLength(inAudio);
  jfloat *x = env->GetFloatArrayElements(inAudio, nullptr);
  if (!x) return env->NewFloatArray(0);

  size_t cap = (size_t)n * 4 + 16384;
  std::vector<float> out(cap);
  rvc_stats_t st;
  long m = rvc_pipeline_convert(g_pipeline, x, (size_t)n, out.data(), cap, &st);
  env->ReleaseFloatArrayElements(inAudio, x, JNI_ABORT);
  if (m < 0) return env->NewFloatArray(0);

  jfloatArray r = env->NewFloatArray((jsize)m);
  env->SetFloatArrayRegion(r, 0, (jsize)m, out.data());
  return r;
}

/* JSON of the last convert's timing breakdown, so a headless test can read
 * RTF without a second JNI round trip per stage. */
JNI_FUNC(jstring, pipelineStats)(JNIEnv *env, jobject) {
  if (!g_pipeline) return env->NewStringUTF("{}");
  const rvc_stats_t *s = rvc_pipeline_stats(g_pipeline);
  /* The dbg queue dump runs to several KB on a long convert, so size the
   * buffer from the actual string instead of truncating mid-JSON (which
   * made the Kotlin parser fall back to rtf=1). */
  std::string js;
  js.reserve(512 + strlen(s->dbg));
  char head[512];
  int n = snprintf(head, sizeof(head),
           "{\"ms_total\":%.1f,\"ms_init\":%.1f,\"ms_highpass\":%.1f,"
           "\"ms_f0\":%.1f,\"ms_c0\":%.1f,\"ms_gn\":%.1f,\"ms_r\":%.1f,"
           "\"ms_enc\":%.1f,\"ms_synth\":%.1f,\"ms_sola\":%.1f,\"ms_rms\":%.1f,"
           "\"ms_index\":%.1f,"
           "\"n_blocks\":%ld,\"n_in\":%ld,\"n_out\":%ld,\"graph_calls\":%ld,"
           "\"rtf\":%.3f,\"next_block\":%d,\"next_prep\":%d,\"n_real\":%ld,"
           "\"f0_windows\":%ld,\"f0_folded\":%ld,\"q_next\":%llu,\"q_done\":%llu,"
           "\"q_fail\":%d,\"q_workers\":%d,\"dbg\":\"",
           s->ms_total, s->ms_init, s->ms_highpass, s->ms_f0, s->ms_c0,
           s->ms_gn, s->ms_r, s->ms_enc, s->ms_synth, s->ms_sola, s->ms_rms,
           s->ms_index,
           s->n_blocks, s->n_in, s->n_out, s->graph_calls, s->rtf,
           s->next_block, s->next_prep, s->n_real, s->f0_windows, s->f0_folded,
           s->q_next, s->q_done, s->q_fail, s->q_workers);
  if (n <= 0) return env->NewStringUTF("{}");
  js.append(head, (size_t)n);
  /* Escape what JSON requires, so a stray quote in the dump cannot end the
   * string early either. */
  for (const char *c = s->dbg; *c; c++) {
    switch (*c) {
      case '"': js += "\\\""; break;
      case '\\': js += "\\\\"; break;
      case '\n': js += "\\n"; break;
      case '\r': js += "\\r"; break;
      case '\t': js += "\\t"; break;
      default: js += *c;
    }
  }
  js += "\"}";
  return env->NewStringUTF(js.c_str());
}

JNI_FUNC(void, pipelineDestroy)(JNIEnv *, jobject) {
  if (g_pipeline) { rvc_pipeline_destroy(g_pipeline); g_pipeline = nullptr; }
}

JNI_FUNC(jstring, graphTimings)(JNIEnv *env, jobject) {
  return env->NewStringUTF(qnn_timings_json());
}

/* ---------- streaming pipeline ---------- */

static std::string g_pipeline_err;

/* Feeds n samples (or drains when x is null), growing the scratch buffer as
 * needed until every computable block has been emitted, and returns them all
 * as one FloatArray.  An empty array means failure — see pipelineLastError. */
static jfloatArray stream_drain(JNIEnv *env, const jfloat *x, jsize n) {
  if (!g_pipeline) return env->NewFloatArray(0);
  size_t cap = ((size_t)n / 4000 + 4) * 10000 + 10000;
  std::vector<float> buf(cap);
  std::vector<float> result;
  rvc_stats_t st;
  bool first = true;
  for (;;) {
    const float *px = first ? x : nullptr;
    jsize pn = first ? n : 0;
    first = false;
    long m = rvc_pipeline_feed_stream(g_pipeline, px, (size_t)pn, buf.data(),
                                      cap, &st);
    if (m < 0) {
      g_pipeline_err = rvc_pipeline_last_error(g_pipeline);
      return env->NewFloatArray(0);
    }
    result.insert(result.end(), buf.data(), buf.data() + (size_t)m);
    if ((size_t)m < cap) break;      /* no more computable blocks right now */
    cap *= 2;
    buf.assign(cap, 0.0f);
  }
  jfloatArray r = env->NewFloatArray((jsize)result.size());
  if (result.size()) env->SetFloatArrayRegion(r, 0, (jsize)result.size(), result.data());
  return r;
}

JNI_FUNC(jstring, pipelineStreamStart)(JNIEnv *env, jobject) {
  if (!g_pipeline) return env->NewStringUTF("no pipeline");
  if (rvc_pipeline_start_stream(g_pipeline) != 0)
    return env->NewStringUTF(rvc_pipeline_last_error(g_pipeline));
  return env->NewStringUTF("");
}

/* Appends the audio and returns every block that became computable. */
JNI_FUNC(jfloatArray, pipelineStreamFeed)(JNIEnv *env, jobject, jfloatArray inAudio) {
  if (!g_pipeline) return env->NewFloatArray(0);
  std::vector<float> xin;
  if (inAudio) {
    jsize n = env->GetArrayLength(inAudio);
    if (n > 0) {
      xin.resize((size_t)n);
      env->GetFloatArrayRegion(inAudio, 0, n, xin.data());
    }
  }
  return stream_drain(env, xin.data(), (jsize)xin.size());
}

/* Pads the input so the final block emits and flushes it. */
JNI_FUNC(jfloatArray, pipelineStreamEnd)(JNIEnv *env, jobject) {
  return stream_drain(env, nullptr, 0);
}

JNI_FUNC(jstring, pipelineLastError)(JNIEnv *env, jobject) {
  return env->NewStringUTF(g_pipeline_err.c_str());
}
