# 转换教程 — 把 RVC v2 checkpoint 变成 app 音色

从一份 RVC v2 `.pth` 到 app 里能唱的音色，四步。每步都有自动校验，
不要跳过校验直接上机。

前置：本机 Termux 有 torch 2.11 + onnx 1.22 + qai-hub 0.52 + QAIRT SDK
（`/data/user/0/com.termux/files/usr/lib/qairt`）。

## 第 0 步：准备

```bash
mkdir -p ~/tmp/<tag>          # tag 例如 mygf_wh
# 把 RVC pth 解压出来：Ai圣园未花.zip 里是 mygf-wh.pth + added_*.index
```

源码用合并好的 RVC fork（已修两个导出崩溃）：
`~/tmp/rvc_merge/infer/module/`。

## 第 1 步：pth → ONNX（3 个图）

用 `~/tmp/voice_build.py`（T30 布局）或 `~/tmp/weihua/export_ctx.py`
（ctx-280 布局，**部署要的是这个**）。导出脚本里发生的事：

1. `load_model()` 装 checkpoint：`SynthesizerTrnMs768NSFsid`，
   `is_half=False`，`del net.enc_q`，strict=False 只过滤 `enc_q.*`。
   缺的键必须是 `enc_q.*` 开头，否则 checkpoint 与类不匹配。
2. `patch_sinegen()`：把 SineGen 的相位累加从 cumsum 换成 matmul 乘法链，
   因为 cumsum 在固定步长下不可导成 ONNX 的静态图；同时去掉 randn。
3. 导出 3 个图（opset 18，`dynamo=True`，静态形状，权重全部内嵌）：
   - `encflow_<tag>_ctx280.onnx` — `EncFlowCtx`，输出取 `m_p[:, :, -t_flow:]`
   - `flowreverse_<tag>_ctx54.onnx`
   - `decnsf_<tag>_T30.onnx` — `DecNSFExt`，NSF gate 作为**输入**外置
4. 内建自检（不通过就别往下走）：
   `gate_np` vs patched `m_source` < 5e-4；
   `DecNSFExt` vs 原生 `GeneratorNSF` < 1e-4（真 gate）/ < 5e-4（我们的 gate）；
   逐图 ORT vs eager torch。
5. 写出确定性 probe：`probe_encflow_<tag>.npz`（phone/pitch/sid/m_p/logs_p/g）。

```bash
cd ~/tmp/<tag> && python3 export_ctx.py --pth mygf-wh.pth --tag mygf_wh
```

**两个 torch.export 原生崩溃的坑**（本机 torch 2.11，已在源码里修好）：
- `commons.convert_pad_shape` 里 `torch.tensor(pad_shape).flip(0).reshape(-1).int().tolist()`
  在 traced forward 内部会 SIGABRT → 改成纯 Python flatten。
- 把 `requires_grad=True` 的张量作为 export 的输入参数会 SIGABRT
  → export 前对所有 args `.detach()`。
- 旧版 TorchScript exporter（`dynamo=False`）在 2.11 上 100% 坏，
  连两行的 embedding 都导不出。必须 `dynamo=True`。

## 第 2 步：ONNX → DLC（云）

本机**没有** ONNX→DLC 转换器，只有 qai-hub 云。

```bash
python3 compile_<tag>.py encflow flowreverse decnsf
```

- 目标设备：`Samsung Galaxy S25 (Family)`（= sm8750-ac）
- `--target_runtime qnn_dlc --truncate_64bit_io`
- 服务地址 `workbench.aihub.qualcomm.com` 会间歇性宕机（TLS 握手超时），
  外面套一层重试循环（`compile_retry.py`，退避到 120 s）。

`--truncate_64bit_io` 是为什么 `input_specs` 里 pitch/sid 声明成 **int64**
而最终 .bin 元数据是 INT_32：形状推断在截断之前跑，声明 int32 会被拒。

## 第 3 步：DLC → context binary（本机）

```bash
qairt-dlc-prepare \
  --backend $QAIRT/lib/aarch64-android/libQairtHtp.so \
  --dlc encflow.dlc --bin encflow.bin
```

环境：

```
ADSP_LIBRARY_PATH=$QAIRT/lib/hexagon-v79/unsigned
LD_LIBRARY_PATH=/vendor/lib64:$QAIRT/lib/aarch64-android
```

## 第 4 步：验证 .bin（上机前必做）

1. **元数据**：`qnn-context-binary-utility --context_binary=X.bin --json_file=X.json`
   导出 sidecar，与已部署音色的 sidecar 逐项比对 name/dtype/dims。
   graphName 不同是正常的，其它必须一致。
2. **执行 + 输出顺序**：用 `qnn-net-run` + YAML v2 input_list（不是旧的
   `name := file` 格式，混合 f32/int32 只有 YAML v2 能正确解析）跑 probe：
   ```bash
   qnn-net-run --backend libQnnHtp.so --retrieve_context encflow.bin \
     --input_list input_list.yaml --use_native_input_files \
     --use_native_output_files
   ```
   `graphName` 必须从 sidecar 读出来填进 YAML，否则 graph 找不到。
   输出在 `out/Result_0/output_N.raw`。
3. **按值判定 output_0 是不是 m_p**：m_p 和 logs_p 形状相同，比绝对差，
   小的那个是 m_p。**必须是 output_0**，因为原生层按位置 memcpy。
4. **精度门槛 2e-2**（不是 1e-3！HTP 内部 fp16，部署中的 zbhy 自己就是
   m_p 5.1e-3 / logs_p 1.1e-3）。

## 第 5 步：staging + 冷启动

```bash
bash ~/tmp/stage_<tag>.sh   # 复制 3 个音色图 + 4 个共用图 + voice.json + index
```

- 复制后 `chown` 到 app uid（见 [音色包格式](voice-format.md) 的陷阱）
- 冷启动测试：`RUN_PIPELINE` 广播，期望
  `in=127060 out=310000 q_fail=0 RTF≈0.5`，无 `crash.txt`
- logcat 会被审计日志在几秒内冲掉，必须**先流式落盘再 grep**，
  不能 sleep 之后 `-d`

## 速查：哪些脚本属于哪一层

| 层 | 脚本 | 作用 |
|---|---|---|
| 导出 | `~/tmp/voice_build.py`、`~/tmp/weihua/export_ctx.py` | pth → ONNX |
| 编译 | `~/tmp/compile_<tag>.py`、`compile_retry.py` | ONNX → 云 DLC → 本机 .bin |
| 验证 | `qnn-context-binary-utility`、`verify_<tag>_bins.py` | 元数据 + 执行 + 输出顺序 |
| 编排（py 侧） | `~/projects/rvc_npu/run_rvc_npu.py`、`qnn_resident.c` | CLI/daemon 两条执行路径 |
| 上机 | `~/tmp/stage_<tag>.sh`、`smoke_test.sh` | staging + 无头测试 |
| 应用 | `app/.../rvc_pipeline.cpp`、`VoiceManager.kt` | 运行时 |

## 已验证音色的 job id

- mygf_zbhy ctx-280：encflow `j5qle1yop`、flowreverse `jgly68ym5`
- mygf_wh decnsf_ext：`jp2rr6v6g`
