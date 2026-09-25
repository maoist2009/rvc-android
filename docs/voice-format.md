# 音色包格式（voice pack）

一个音色 = `filesDir/voices/<name>/` 下的一个目录，app 用 `VoiceManager`
把它 staging 到 `filesDir/models/`（扁平布局，原生层只认 models/）。

## 必需文件

| 文件 | 大小（参考） | 说明 |
|---|---|---|
| `hubert_c0_s23.bin` | ~200 KB | HuBERT conv0 特征提取 |
| `hubert_r_2063_s23.bin` | ~370 MB | HuBERT 残差变换（15 层残差块） |
| `hubert_enc_140_s23.bin` | ~200 KB | HuBERT 编码器 |
| `rmvpe_e2e_s23.bin` | ~14 MB | RMVPE F0 提取（端到端） |
| `encflow.bin` | ~14 MB | posterior encoder + flow（输出 m_p/logs_p/g） |
| `flowreverse.bin` | ~13 MB | flow 逆变换 |
| `decnsf_ext.bin` | ~31 MB | NSF 声码器（gate 外置） |
| `conv0_gn.f32` | 4096 B | HuBERT conv0 的 GroupNorm affine：512 权重 + 512 偏置，小端 float32 |
| `voice.json` | — | 布局常量 + gate 标量（见下） |

## 可选文件

| 文件 | 说明 |
|---|---|
| `retrieval.index` / `index.path` | faiss IVF-Flat 检索索引，`index_rate > 0` 时启用 |

## voice.json

```json
{
  "tag": "mygf_wh",
  "sr": 40000,
  "t_ph": 280, "t_out": 30, "t_flow": 54,
  "flow_head": 226, "dec_head": 24,
  "upp": 400, "sine_amp": 0.1, "dim": 1,
  "l_w": -0.81982421875, "l_b": 0.0005002021789550781,
  "n_spk": 109, "sid": 0
}
```

原生层按 key 扫描（`rvc_jni.cpp::pipelineInit`），缺的键回退到
**mygf_zbhy ctx-280 默认值**。所有键必须与 .bin 编译时的布局一致，
否则 `pipelineInit` 会以
`input 'phone' size N != M bytes expected` 之类的形状不匹配失败。

## IO 契约（ctx-280）

```
encflow     phone(1,280,768) f32, pitch(1,280) INT_32, sid(1,) INT_32
            -> output_0 (1,192,54) f32   ← m_p（posterior 均值）
               output_1 (1,192,54) f32   ← logs_p（对数方差）
               output_2 (1,256,1)  f32   ← 说话人嵌入 g
flowreverse z_p(1,192,54) f32, mask(1,1,54) f32, g(1,256,1) f32
            -> output_0 (1,192,54) f32
decnsf_ext  z(1,192,30) f32, sid(1,) INT_32, sine_gate_in(1,1,12000) f32
            -> output_0 (1,1,12000) f32
```

⚠️ **encflow 的 output_0/output_1 形状完全相同**（都是 `(1,192,54)`）。
原生层按位置 memcpy（`outs[0]` 当 m_p、`outs[1]` 当 logs_p）。QNN 会把输出
重命名为 `output_<ONNX输出索引>`，所以**每换一个音色，必须按值确认 output_0
真的是 m_p**（用确定性的 probe npz 比对，见 [转换教程](tutorial-convert.md)）。
映射错了管线不会报错，只会输出噪声。

## 共用组件

`hubert_*` 和 `rmvpe_*` 四个图对所有音色通用（它们不含音色权重）。
所以多音色时只有 `encflow/flowreverse/decnsf_ext/conv0_gn.f32/voice.json`
是音色专属的，前四个可以从已有音色目录复制（或 app 内共享导入，见导入功能）。

## 已验证的音色

| 音色 | l_w | l_b | 备注 |
|---|---|---|---|
| mygf_zbhy | -0.826171875 | 0.002140045166015625 | 40 kHz 中文女声，端到端 RTF 0.537 |
| mygf_wh（Ai圣园未花） | -0.81982421875 | 0.0005002021789550781 | 40 kHz，RTF 0.553 |

## 部署陷阱

- `models/synthesizer/*_s23.bin` 是 **T30** 布局（phone `(1,30,768)`），
  staging 那一套会让 `pipelineInit` 失败并 q_fail=1、out=0。
  正确的是 ctx-280 布局。文件名相同、形状不同——永远用 ctx-280 那批。
- `su -c cp` 复制进来的文件是 root 所有，app 读不了；
  要 `chown <app_uid>:<app_uid> && chmod 644`。
- 精度基线：**NPU vs ORT ≈ 5e-3（m_p）/ 1e-3（logs_p）**，因为 HTP 编译产物
  内部是 fp16。ONNX-vs-torch 才是 ~2e-6。新音色按 2e-2 判定，不要按 1e-3。
