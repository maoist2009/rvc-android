# DreamLite RVC — 端侧实时语音转换（NPU）

在骁龙 8 Elite（SM8750）上跑 RVC v2 推理，全部算力落在 Hexagon HTP NPU，
不依赖云、不依赖 CPU 推理框架。

- **包名**：`com.dreamlite.rvc`（debug 构建为 `com.dreamlite.rvc.tmp`）
- **输入**：16 kHz 单声道 PCM
- **输出**：40 kHz 单声道 PCM（管线内自带 48k↔40k 重采样）
- **实测**：RTF 0.537（mygf_zbhy，7.9 s 输入 → 310000 样本），q_fail=0

## 目录结构

```
app/src/main/cpp/          原生层
  qnn_executor.{cpp,h}     QNN HTP 后端封装（context 加载 / graph 执行 / DCVS 电压投票）
  rvc_pipeline.{cpp,h}     7 图流式管线（HuBERT→RMVPE→encflow→flowreverse→decnsf）
  index_retrieval.*        faiss IVF-Flat 检索（index_rate > 0 时启用）
  dsp.*                    重采样 / mel / 噪声门
  rvc_jni.cpp              JNI 边界
app/src/main/java/         Kotlin 层
  data/VoiceManager.kt     音色发现 + staging（voices/<name>/ → models/）
  data/SettingsRepository.kt  DataStore 持久化设置
  engine/RvcSession.kt     录音 → 管线 → 播放 的编排
  audio/                   AudioRecorder(16k) / AudioPlayer(40k)
  ui/                      Compose 两屏：Home / Settings
app/src/main/jniLibs/      QNN 运行时 so（HTP v79 skel + stub）
```

## 构建

```bash
./gradlew assembleMonocleschatFreeDebug   # 或对应 flavor
# 产物：app/build/outputs/apk/.../*.apk
```

构建机必须能离线解析依赖（navigation-compose 未纳入缓存，故导航用朴素状态切换，
见 `ui/nav/DreamLiteNavHost.kt` 注释）。

## 运行（无头测试）

shell 无法直接拉起 FGS，必须先起前台服务再发广播：

```bash
PKG=com.dreamlite.rvc.tmp
sudo /system/bin/am start-foreground-service -n $PKG/com.dreamlite.rvc.service.RvcLiveService
sudo /system/bin/am broadcast -a com.dreamlite.rvc.RUN_PIPELINE -n $PKG/com.dreamlite.rvc.PipelineReceiver
# logcat 会被审计日志快速冲掉，先流式落盘再 grep
sudo timeout 60 logcat -b main,crash -d | grep rvc_pipeline_test
```

debug receiver（`app/src/debug/`）提供 `RUN_SMOKE`（只加载图元数据）与
`RUN_PIPELINE`（跑 test_input.f32 并报告 in/out/q_fail/RTF）。

## 音色从哪来

音色不是 app 内置的，需要自己从 RVC v2 checkpoint 编译成 QNN context binary。
完整流程见 [转换教程](tutorial-convert.md)，音色包格式见 [音色包格式](voice-format.md)。

## 性能要点

- **DCVS 电压投票是强制的**：`qnn_executor.cpp::qnn_init` 在 deviceCreate 之后
  立即推 `QnnHtpPerfInfrastructure` 的 MAX corner 配置。不推的话 DSP 停在低压角，
  大图慢 4–5 倍（RTF 2.57 → 0.70）。输出与投票前后逐位相同，只改时钟。
- skel 目录必须传 app 自己的 nativeLibraryDir，否则 deviceCreate 失败 0x36b1。
- 管线是 ctx-280 布局：280 帧上下文，输出最后 54 帧，block 间靠 SOLA 拼接。
  常量在 `rvc_pipeline.cpp` 顶部，voice.json 可覆盖。
