/* qnn_executor.cpp — in-process QNN HTP executor.
 *
 * Ported from rvc_npu/qnn_resident.c (the resident multi-context daemon that
 * ran the full 5-stage RVC pipeline bit-exact against qnn-net-run). The daemon
 * framing (u32-length records over stdin/stdout) is replaced by direct C entry
 * points; the QNN sequence and the pinned 4 KB-aligned client buffers are
 * unchanged, because the HTP context binds clientBuf.data across graphExecute
 * calls and the buffers must never move once the context is created.
 */
#include "qnn_executor.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <android/log.h>

#include "QnnInterface.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

#define TAG "rvc_native"

#define MAX_GRAPHS  16
#define MAX_TENSORS 16
#define NAME_LEN    128
#define ERR_LEN     1024

static const QnnInterface_t *IFACE;   /* -> v2_34 */
static Qnn_LogHandle_t      LOG;
static Qnn_BackendHandle_t  BACKEND;
static Qnn_DeviceHandle_t   DEVICE;
static int                  INIT_DONE = 0;
static pthread_mutex_t      LOCK = PTHREAD_MUTEX_INITIALIZER;
static char                 ERR_BUF[ERR_LEN];

static void set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(ERR_BUF, sizeof(ERR_BUF), fmt, ap);
  va_end(ap);
}

const char *qnn_last_error(void) { return ERR_BUF; }

/* Matches qnn_resident's android log path; logs go to logcat under rvc_native. */
static void qlog(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  __android_log_vprint(ANDROID_LOG_INFO, TAG, fmt, ap);
  va_end(ap);
}

static void qlog_cb(const char *fmt, QnnLog_Level_t level, uint64_t ts, va_list ap) {
  char buf[1024];
  int n = snprintf(buf, sizeof(buf), "[qnn:%d ts=%llu] ", (int)level,
                   (unsigned long long)ts);
  if (n > 0 && (size_t)n < sizeof(buf))
    vsnprintf(buf + n, sizeof(buf) - n, fmt, ap);
  __android_log_print(ANDROID_LOG_INFO, TAG, "%s", buf);
}

/* --- per-graph state ----------------------------------------------------- */

struct tensor {
  char     name[NAME_LEN];
  uint32_t id;       /* HTP matches graph tensors by id, not name */
  uint32_t rank, dims[8];
  Qnn_DataType_t     dtype;
  Qnn_TensorDataFormat_t dataFormat;
  Qnn_QuantizeParams_t   quantizeParams;
  size_t   bytes;
  void    *buf;      /* 4 KB-aligned, stable for the context's lifetime */
};

struct graph {
  int      used;
  char     tag[NAME_LEN];
  int      fd;
  void    *map;
  size_t   map_len;
  Qnn_ContextHandle_t ctx;
  Qnn_GraphHandle_t    g;
  uint32_t n_in, n_out;
  struct tensor in[MAX_TENSORS];
  struct tensor out[MAX_TENSORS];
  /* graphExecute-only wall time, excluding host marshalling. */
  uint64_t exec_ns_sum = 0;
  uint64_t exec_ns_min = 0;
  uint64_t exec_ns_max = 0;
  uint64_t exec_n = 0;
};

static struct graph GRAPHS[MAX_GRAPHS];

static struct graph *find_graph(const char *tag) {
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used && !strcmp(GRAPHS[i].tag, tag)) return &GRAPHS[i];
  return NULL;
}

static size_t dtype_bytes(Qnn_DataType_t dt, size_t elems) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      return elems * 4;
    default:
      return 0;   /* unsupported at the boundary — refuse, never miscast */
  }
}

static void free_graph(struct graph *g) {
#define IF IFACE->v2_34
  if (g->g) IF.graphFinalize(g->g, NULL, NULL);
  if (g->ctx) IF.contextFree(g->ctx, NULL);
#undef IF
  if (g->map && g->map != MAP_FAILED) munmap(g->map, g->map_len);
  if (g->fd >= 0) close(g->fd);
  for (uint32_t i = 0; i < g->n_in; i++) free(g->in[i].buf);
  for (uint32_t i = 0; i < g->n_out; i++) free(g->out[i].buf);
  memset(g, 0, sizeof(*g));
  g->fd = -1;
}

/* Copy a tensor descriptor out of the system metadata into a plain struct.
 * Mirrors qnn_min.c's tensor_v2_from_sys. */
static int describe(const Qnn_Tensor_t *src, struct tensor *t) {
  memset(t, 0, sizeof(*t));
  const char *name = NULL;
  uint32_t rank = 0;
  const uint32_t *dims = NULL;
  if (src->version == QNN_TENSOR_VERSION_2) {
    name = src->v2.name;
    rank = src->v2.rank;
    dims = src->v2.dimensions;
    t->dtype = src->v2.dataType;
    t->id = src->v2.id;
    t->dataFormat = src->v2.dataFormat;
    t->quantizeParams = src->v2.quantizeParams;
  } else {
    name = src->v1.name;
    rank = src->v1.rank;
    dims = src->v1.dimensions;
    t->dtype = src->v1.dataType;
    t->id = src->v1.id;
    t->dataFormat = src->v1.dataFormat;
    t->quantizeParams = src->v1.quantizeParams;
  }
  if (!name || strlen(name) >= NAME_LEN || rank > 8) return -1;
  strcpy(t->name, name);
  t->rank = rank;
  size_t elems = 1;
  for (uint32_t i = 0; i < rank; i++) {
    t->dims[i] = dims[i];
    elems *= dims[i];
  }
  size_t b = dtype_bytes(t->dtype, elems);
  if (!b) {
    set_error("tensor '%s' has unsupported dtype %d", name, (int)t->dtype);
    return -1;
  }
  t->bytes = b;
  void *p = NULL;
  if (posix_memalign(&p, 4096, b) != 0 || !p) return -1;
  t->buf = p;
  return 0;
}

/* libQnnSystem metadata: graph name + input/output tensor descriptors. */
static int graph_info(const void *bin, size_t len, const char **graphName,
                      const QnnSystemContext_GraphInfo_t **gi) {
  void *sh = dlopen("libQnnSystem.so", RTLD_NOW);
  if (!sh) {
    set_error("dlopen libQnnSystem.so: %s", dlerror());
    return -1;
  }
  Qnn_ErrorHandle_t (*sysGetProviders)(const QnnSystemInterface_t ***, uint32_t *) =
      (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***, uint32_t *))
          dlsym(sh, "QnnSystemInterface_getProviders");
  if (!sysGetProviders) {
    set_error("dlsym QnnSystemInterface_getProviders: %s", dlerror());
    return -1;
  }
  const QnnSystemInterface_t **list = NULL;
  uint32_t n = 0;
  if (sysGetProviders(&list, &n)) return -1;
  const QnnSystemInterface_t *sel = NULL;
  for (uint32_t i = 0; i < n; i++)
    if (list[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        list[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR)
      sel = list[i];
  if (!sel) {
    set_error("no QNN System API %u.%u provider",
              QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR);
    return -1;
  }
#define SIF sel->v1_9
  QnnSystemContext_Handle_t sysCtx = NULL;
  if (SIF.systemContextCreate(&sysCtx)) return -1;
  const QnnSystemContext_BinaryInfo_t *info = NULL;
  Qnn_ContextBinarySize_t infoSize = 0;
  if (SIF.systemContextGetBinaryInfo(sysCtx, (void *)bin, (uint64_t)len, &info,
                                     &infoSize)) {
    set_error("systemContextGetBinaryInfo failed");
    return -1;
  }
  if (!info) return -1;
  uint32_t numGraphs = 0;
  const QnnSystemContext_GraphInfo_t *graphs = NULL;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      numGraphs = info->contextBinaryInfoV1.numGraphs;
      graphs = info->contextBinaryInfoV1.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      numGraphs = info->contextBinaryInfoV2.numGraphs;
      graphs = info->contextBinaryInfoV2.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      numGraphs = info->contextBinaryInfoV3.numGraphs;
      graphs = info->contextBinaryInfoV3.graphs;
      break;
    default:
      set_error("unknown BinaryInfo version %d", (int)info->version);
      return -1;
  }
  if (numGraphs != 1) {
    set_error("expected 1 graph, found %u", numGraphs);
    return -1;
  }
  *gi = &graphs[0];
  *graphName = graphs[0].graphInfoV1.graphName;
  return 0;
#undef SIF
}

int qnn_init(const char *lib_dir, const char *skel_dir) {
  pthread_mutex_lock(&LOCK);
  if (INIT_DONE) {
    pthread_mutex_unlock(&LOCK);
    return 0;
  }
  /* Under the bionic linker only ADSP_LIBRARY_PATH decides where the HTP stub
   * looks for libQnnHtpV79Skel.so; LD_LIBRARY_PATH is set as well because that
   * is what the Termux-side path relies on. Both must be set before the first
   * call into libQnnHtp.so. */
  setenv("ADSP_LIBRARY_PATH", skel_dir, 1);
  setenv("LD_LIBRARY_PATH", lib_dir, 1);

  void *h = dlopen("libQnnHtp.so", RTLD_NOW);
  if (!h) {
    set_error("dlopen libQnnHtp.so: %s", dlerror());
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  Qnn_ErrorHandle_t (*getProviders)(const QnnInterface_t ***, uint32_t *) =
      (Qnn_ErrorHandle_t(*)(const QnnInterface_t ***, uint32_t *))
          dlsym(h, "QnnInterface_getProviders");
  if (!getProviders) {
    set_error("dlsym QnnInterface_getProviders: %s", dlerror());
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  const QnnInterface_t **providers = NULL;
  uint32_t num = 0;
  if (getProviders(&providers, &num)) {
    set_error("QnnInterface_getProviders failed");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (uint32_t i = 0; i < num; i++) {
    const Qnn_ApiVersion_t *v = &providers[i]->apiVersion;
    qlog("provider[%u] %s core=%u.%u.%u backend=%u.%u.%u", i,
         providers[i]->providerName ? providers[i]->providerName : "(null)",
         v->coreApiVersion.major, v->coreApiVersion.minor, v->coreApiVersion.patch,
         v->backendApiVersion.major, v->backendApiVersion.minor,
         v->backendApiVersion.patch);
    if (v->coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        v->coreApiVersion.minor >= QNN_API_VERSION_MINOR)
      IFACE = providers[i];
  }
  if (!IFACE) {
    set_error("no QNN API %u.%u provider", QNN_API_VERSION_MAJOR,
              QNN_API_VERSION_MINOR);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
#define IF IFACE->v2_34
  Qnn_ErrorHandle_t e;
  e = IF.logCreate(qlog_cb, QNN_LOG_LEVEL_DEBUG, &LOG);
  if (e) {
    set_error("logCreate 0x%llx", (unsigned long long)e);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  e = IF.backendCreate(LOG, NULL, &BACKEND);
  if (e) {
    set_error("backendCreate 0x%llx", (unsigned long long)e);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  e = IF.deviceCreate(LOG, NULL, &DEVICE);
  if (e) {
    set_error("deviceCreate 0x%llx", (unsigned long long)e);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }

  /* Vote the DSP up to the max voltage corner with sleep disabled. Without an
   * explicit power config the skel stays at a low corner and low bus bandwidth,
   * which is why the big graphs (rmvpe ~190 MB of weights) run 3-5x slower than
   * the same .bin in a process that votes. ML-Sharp-QNN does the same thing. */
  QnnDevice_Infrastructure_t infra = NULL;
  if (IF.deviceGetInfrastructure && QNN_SUCCESS == IF.deviceGetInfrastructure(&infra) && infra) {
    QnnHtpDevice_Infrastructure_t *htpInfra = (QnnHtpDevice_Infrastructure_t *)infra;
    QnnHtpDevice_PerfInfrastructure_t perf = htpInfra->perfInfra;
    if (perf.createPowerConfigId && perf.setPowerConfig) {
      uint32_t cfgId = 0;
      if (QNN_HTP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED !=
              perf.createPowerConfigId(0, 0, &cfgId) &&
          cfgId != 0) {
        QnnHtpPerfInfrastructure_PowerConfig_t pc;
        memset(&pc, 0, sizeof(pc));
        pc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        pc.dcvsV3Config.contextId = cfgId;
        pc.dcvsV3Config.setDcvsEnable = 1;
        pc.dcvsV3Config.dcvsEnable = 1;
        /* locked max: min=target=max pins the frequency, sleep off keeps it
         * there between graphExecute calls (this is a short burst of work) */
        pc.dcvsV3Config.setSleepDisable = 1;
        pc.dcvsV3Config.sleepDisable = 1;
        pc.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
        pc.dcvsV3Config.setBusParams = 1;
        pc.dcvsV3Config.busVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.busVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.busVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.setCoreParams = 1;
        pc.dcvsV3Config.coreVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.coreVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        pc.dcvsV3Config.coreVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t *pcs[] = {&pc, NULL};
        Qnn_ErrorHandle_t pe = perf.setPowerConfig(cfgId, pcs);
        if (pe != QNN_SUCCESS) {
          qlog("perf infra: setPowerConfig 0x%llx (non-fatal)",
               (unsigned long long)pe);
          perf.destroyPowerConfigId(cfgId);
        } else {
          qlog("perf infra: locked MAX corner, sleep disabled (cfgId=%u)", cfgId);
        }
      } else {
        qlog("perf infra: createPowerConfigId unsupported");
      }
    } else {
      qlog("perf infra: createPowerConfigId/setPowerConfig unavailable");
    }
  } else {
    qlog("perf infra: deviceGetInfrastructure unavailable");
  }
#undef IF
  INIT_DONE = 1;
  qlog("qnn_init ok: backend ready");
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_load(const char *tag, const char *bin_path) {
  pthread_mutex_lock(&LOCK);
  if (!INIT_DONE) {
    set_error("backend not initialised");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  struct graph *g = find_graph(tag);
  if (g) free_graph(g);
  g = NULL;
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (!GRAPHS[i].used) {
      g = &GRAPHS[i];
      break;
    }
  if (!g) {
    set_error("no free graph slots");
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  memset(g, 0, sizeof(*g));
  g->fd = -1;

  const char *gname = NULL;
  const QnnSystemContext_GraphInfo_t *gi = NULL;

  g->fd = open(bin_path, O_RDONLY);
  if (g->fd < 0) {
    set_error("open %s", bin_path);
    goto fail;
  }
  struct stat st;
  if (fstat(g->fd, &st) != 0) {
    set_error("fstat %s", bin_path);
    goto fail;
  }
  g->map_len = (size_t)st.st_size;
  g->map = mmap(NULL, g->map_len, PROT_READ, MAP_PRIVATE, g->fd, 0);
  if (g->map == MAP_FAILED) {
    set_error("mmap %s", bin_path);
    goto fail;
  }

#define IF IFACE->v2_34
  if (IF.contextCreateFromBinary(BACKEND, DEVICE, NULL, g->map,
                                 (Qnn_ContextBinarySize_t)g->map_len, &g->ctx,
                                 NULL)) {
    set_error("contextCreateFromBinary failed for %s", bin_path);
    goto fail;
  }
  if (graph_info(g->map, g->map_len, &gname, &gi) != 0 || !gname || !gi) {
    set_error("graph metadata for %s", bin_path);
    goto fail;
  }
  if (IF.graphRetrieve(g->ctx, gname, &g->g)) {
    set_error("graphRetrieve '%s' failed", gname);
    goto fail;
  }
  g->n_in = gi->graphInfoV1.numGraphInputs;
  g->n_out = gi->graphInfoV1.numGraphOutputs;
  if (g->n_in > MAX_TENSORS || g->n_out > MAX_TENSORS || g->n_in == 0 ||
      g->n_out == 0) {
    set_error("tensor count in=%u out=%u", g->n_in, g->n_out);
    goto fail;
  }
  for (uint32_t i = 0; i < g->n_in; i++)
    if (describe(&gi->graphInfoV1.graphInputs[i], &g->in[i]) != 0) {
      set_error("input tensor %u of %s", i, bin_path);
      goto fail;
    }
  for (uint32_t i = 0; i < g->n_out; i++)
    if (describe(&gi->graphInfoV1.graphOutputs[i], &g->out[i]) != 0) {
      set_error("output tensor %u of %s", i, bin_path);
      goto fail;
    }
#undef IF

  strcpy(g->tag, tag);
  g->used = 1;
  qlog("loaded '%s' in=%u out=%u (%zu kB)", tag, g->n_in, g->n_out,
       g->map_len / 1024);
  pthread_mutex_unlock(&LOCK);
  return 0;

fail:
  free_graph(g);
  pthread_mutex_unlock(&LOCK);
  return -1;
}

int qnn_describe(const char *tag, QnnTensorCb in_cb, QnnTensorCb out_cb,
                 void *user) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (uint32_t i = 0; i < g->n_in; i++)
    if (in_cb) in_cb(user, g->in[i].name, (int)g->in[i].dtype, g->in[i].rank, g->in[i].dims);
  for (uint32_t i = 0; i < g->n_out; i++)
    if (out_cb) out_cb(user, g->out[i].name, (int)g->out[i].dtype, g->out[i].rank,
                       g->out[i].dims);
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_exec(const char *tag,
             const char *const *in_names, const int *in_dtypes,
             const void *const *in_data, const size_t *in_elems, size_t n_in,
             float **out_data, size_t *out_elems, size_t n_out_max,
             size_t *n_out) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  if (n_in != g->n_in) {
    set_error("'%s' expects %u inputs, got %zu", tag, g->n_in, n_in);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (size_t i = 0; i < n_in; i++) {
    struct tensor *t = NULL;
    for (uint32_t j = 0; j < g->n_in; j++)
      if (!strcmp(g->in[j].name, in_names[i])) {
        t = &g->in[j];
        break;
      }
    if (!t) {
      set_error("unknown input '%s' for '%s'", in_names[i], tag);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    int want = (t->dtype == QNN_DATATYPE_FLOAT_32) ? RVC_QNN_FLOAT32
                                                   : RVC_QNN_INT32;
    if (in_dtypes[i] != want) {
      set_error("input '%s' dtype mismatch (want %d got %d)", in_names[i], want,
                in_dtypes[i]);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    if (in_elems[i] * 4 != t->bytes) {
      set_error("input '%s' size %zu elements != %zu bytes expected",
                in_names[i], in_elems[i], t->bytes / 4);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    memcpy(t->buf, in_data[i], t->bytes);
  }

  Qnn_Tensor_t inT[MAX_TENSORS], outT[MAX_TENSORS];
  for (uint32_t i = 0; i < g->n_in; i++) {
    inT[i].version = QNN_TENSOR_VERSION_2;
    Qnn_TensorV2_t *t = &inT[i].v2;
    memset(t, 0, sizeof(*t));
    t->id = g->in[i].id;
    t->name = g->in[i].name;
    t->type = QNN_TENSOR_TYPE_APP_WRITE;
    t->dataFormat = g->in[i].dataFormat;
    t->dataType = g->in[i].dtype;
    t->quantizeParams = g->in[i].quantizeParams;
    t->rank = g->in[i].rank;
    t->dimensions = g->in[i].dims;
    t->memType = QNN_TENSORMEMTYPE_RAW;
    t->clientBuf.data = g->in[i].buf;
    t->clientBuf.dataSize = (uint32_t)g->in[i].bytes;
  }
  for (uint32_t i = 0; i < g->n_out; i++) {
    outT[i].version = QNN_TENSOR_VERSION_2;
    Qnn_TensorV2_t *t = &outT[i].v2;
    memset(t, 0, sizeof(*t));
    t->id = g->out[i].id;
    t->name = g->out[i].name;
    t->type = QNN_TENSOR_TYPE_APP_READ;
    t->dataFormat = g->out[i].dataFormat;
    t->dataType = g->out[i].dtype;
    t->quantizeParams = g->out[i].quantizeParams;
    t->rank = g->out[i].rank;
    t->dimensions = g->out[i].dims;
    t->memType = QNN_TENSORMEMTYPE_RAW;
    t->clientBuf.data = g->out[i].buf;
    t->clientBuf.dataSize = (uint32_t)g->out[i].bytes;
  }

#define IF IFACE->v2_34
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  Qnn_ErrorHandle_t err = IF.graphExecute(g->g, inT, g->n_in, outT, g->n_out,
                                          NULL, NULL);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
#undef IF
  uint64_t dt_ns = (uint64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000ULL +
                   (uint64_t)ts1.tv_nsec - (uint64_t)ts0.tv_nsec;
  g->exec_n++;
  g->exec_ns_sum += dt_ns;
  if (g->exec_ns_min == 0 || dt_ns < g->exec_ns_min) g->exec_ns_min = dt_ns;
  if (dt_ns > g->exec_ns_max) g->exec_ns_max = dt_ns;
  if (err) {
    set_error("graphExecute 0x%llx on '%s'", (unsigned long long)err, tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }

  if (g->n_out > n_out_max) {
    set_error("caller provided space for %zu outputs, graph has %u", n_out_max,
              g->n_out);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  for (uint32_t i = 0; i < g->n_out; i++) {
    size_t elems = g->out[i].bytes / 4;   /* outputs are float32 */
    float *dst = (float *)malloc(g->out[i].bytes);
    if (!dst) {
      for (uint32_t k = 0; k < i; k++) free(out_data[k]);
      set_error("out of memory for output %u", i);
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    memcpy(dst, g->out[i].buf, g->out[i].bytes);
    out_data[i] = dst;
    out_elems[i] = elems;
  }
  *n_out = g->n_out;
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_free(const char *tag) {
  pthread_mutex_lock(&LOCK);
  struct graph *g = find_graph(tag);
  if (!g) {
    set_error("no such graph '%s'", tag);
    pthread_mutex_unlock(&LOCK);
    return -1;
  }
  free_graph(g);
  qlog("freed '%s'", tag);
  pthread_mutex_unlock(&LOCK);
  return 0;
}

int qnn_graph_count(void) {
  int n = 0;
  pthread_mutex_lock(&LOCK);
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used) n++;
  pthread_mutex_unlock(&LOCK);
  return n;
}

/* Per-graph graphExecute wall time, so a headless run can tell a DSP-bound
 * graph from a host-marshalling-bound one without a debugger attached. */
const char *qnn_timings_json(void) {
  static char buf[2048];
  pthread_mutex_lock(&LOCK);
  size_t off = 0;
  for (int i = 0; i < MAX_GRAPHS; i++) {
    struct graph *g = &GRAPHS[i];
    if (!g->used || g->exec_n == 0) continue;
    double mean_ms = (double)g->exec_ns_sum / (double)g->exec_n / 1e6;
    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
                            "%s{\"tag\":\"%s\",\"n\":%llu,\"mean_ms\":%.2f,"
                            "\"min_ms\":%.2f,\"max_ms\":%.2f,\"total_ms\":%.0f}",
                            off ? "," : "", g->tag,
                            (unsigned long long)g->exec_n, mean_ms,
                            (double)g->exec_ns_min / 1e6,
                            (double)g->exec_ns_max / 1e6,
                            (double)g->exec_ns_sum / 1e6);
    if (off > sizeof(buf) - 128) break;
  }
  pthread_mutex_unlock(&LOCK);
  buf[off] = '\0';
  return buf;
}

void qnn_shutdown(void) {
  pthread_mutex_lock(&LOCK);
  for (int i = 0; i < MAX_GRAPHS; i++)
    if (GRAPHS[i].used) free_graph(&GRAPHS[i]);
  if (INIT_DONE) {
#define IF IFACE->v2_34
    if (DEVICE) IF.deviceFree(DEVICE);
    if (BACKEND) IF.backendFree(BACKEND);
    if (LOG) IF.logFree(LOG);
#undef IF
    INIT_DONE = 0;
  }
  pthread_mutex_unlock(&LOCK);
}
