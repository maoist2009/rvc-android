# 架构 — 数据流与原生管线

## Java/Kotlin → JNI → 原生

```
AudioRecorder(16k mic) ──┐
                         ├─> RvcJni.pipelineStreamFeed ─> rvc_pipeline.cpp
音频文件(16k mono f32) ───┘        (Channel<FloatArray>，feeder 协程)      │
                                                                          ▼
                                                    HuBERT(c0/r/enc)  ─> phone
                                                    RMVPE          ─> pitch/f0
                                                          encflow  ─> m_p, logs_p, g
                                              z_p = m_p + randn*exp(logs_p)*NOISE_SCALE
                                                          flowreverse ─> z
                                                          decnsf_ext  ─> 40k audio
                                                          SOLA 拼接 + 峰值限幅
                                                                  │
  AudioPlayer(40k) <── feeder 收集 emitted blocks ─────────────────┘
```

## 关键常量（ctx-280 布局）

```
t_ph      = 280   推理上下文帧数（HuBERT 帧率 100 fps → 2.8 s）
t_out     = 30    每个 block 输出的帧数（= 12000 样本 @40k = 300 ms）
t_flow    = 54    encflow 输出的帧数（280 帧上下文的最后 54 帧）
flow_head = 226   flow 段起点 = t_ph - t_flow
dec_head  = 24    z 里喂给 decnsf 的起点
CTX_STRIDE= 25    block 步长（帧），相邻 block 重叠 5 帧，SOLA 融合
upp       = 400   上采样倍数（t_out * 400 = 12000 样本）
```

## 原生管线分段（`rvc_pipeline.cpp`）

1. **pipelineInit**：`qnn_load` 7 个 context（扁平 models/），
   读 `conv0_gn.f32`（raw LE f32，512+512），读 voice.json 覆盖默认常量。
   voice.json 的键在 `rvc_jni.cpp` 里按名扫描，缺则回退 mygf_zbhy 默认。
2. **stream feed**：每个 block 前进 CTX_STRIDE 帧，维持 280 帧滑动窗；
   encflow 在两个并行 `ctx_bufs`（par 0/1）上交替执行，用 NSF gate 的构建
   时间掩盖 encflow 的延迟。
3. **block 合成**（`rvc_pipeline.cpp` ~1340 行）：
   encflow → 位置 memcpy `outs[0]`/`outs[1]`/`outs[2]` → 采样
   → flowreverse → `z + r + ch*t_flow + dec_head` 切片
   → decnsf（`precompute_sine_gate(f0[t_ph-t_out:])`）→ SOLA → 峰值限幅。
4. **崩溃处理**：native crash 写 `filesDir/crash.txt`；
   `filesDir/models/qlog_on` 标记开启 `qnn_jobs.log` 逐任务轨迹。

## QNN 执行器（`qnn_executor.cpp`）

- `qnn_init(lib_dir, skel_dir)`：skel 目录必须是 app nativeLibraryDir，
  否则 `deviceCreate` 失败 0x36b1。之后立即推 DCVS MAX-corner 电压投票
  （`QnnHtpPerfInfrastructure` createPowerConfigId / setPowerConfig，
  DCVS_V3 + sleepDisable + bus/core 都设 MAX + ADJUST_UP_DOWN）。
- `qnn_load(tag, bin)`：mmap .bin → `contextCreateFromBinary` →
  `graphRetrieve`（用元数据里的 graphName）→ `describe()` 逐张量
  从内嵌 `QnnSystemContext` blob 取名字/dtype/dims。
- `qnn_exec()`：输入**按名 + dtype** 匹配。形状/字节数不匹配时报
  `input 'X' size N bytes != M expected`——这是错误形状的音色 bin 在 app
  里唯一的可见症状。

## 音色发现与 staging（`VoiceManager.kt`）

- 音色目录：`filesDir/voices/<name>/`，需 8 个图 + voice.json 齐全才算有效
  （`list()` 只列齐全的）。
- `stage()`：清掉 models/ 里自己拥有的文件，把音色目录 symlink 进
  `filesDir/models/`（相对路径 `../voices/<name>/x.bin`），失败回退整目录拷贝。
  ⚠️ 路径相对 filesDir.parent 会变成 `files/files/...`，静默失败。
- symlink 优先是因为音色图集 ~218 MB/音色，每次切换都拷贝太慢。

## 已知限制

- navigation-compose 不在离线 Gradle 缓存里 → 导航用 `enum class Screen` 状态切换。
- shell 起的 FGS 活不过 ~3 分钟（Android 15/16），无头测试脚本要紧跟广播。
- logcat 审计日志会秒级冲掉 buffer → 先 `timeout N logcat > file &` 再广播。
- Termux 自身的 `logcat`/`ps`/`nsenter` 看不到 app 进程，要 `sudo logcat`。
