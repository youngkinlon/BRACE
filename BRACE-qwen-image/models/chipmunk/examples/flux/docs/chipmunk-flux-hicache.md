# Chipmunk‑FLUX 与 HiCache 集成说明（实验性）

> 本文档只针对 `models/chipmunk/examples/flux` 下的 Chipmunk‑FLUX 集成，目的是澄清：
> 
> - Chipmunk‑FLUX 和原生 FLUX 的差异；
> - 当前 `--backend chipmunk --mode HiCache` 的真实行为与已知问题；
> - 推荐的“正确性对比”与“性能评测”用法。

## 1. 模型与路径对比

- **原生 FLUX 路径**
  - 代码位置：`models/flux/src/flux`
  - 运行入口：`--backend flux`（`scripts/sample.sh` / `RUN/multi_gpu_launcher.sh`）
  - 关键特性：
    - 使用 BFL 官方 FLUX 推理代码（Dense Attention + Dense MLP）；
    - HiCache / Taylor 通过 `flux.ideas.denoise_cache` + `flux.modules.cache_functions.*` 实现；
    - 默认 **不开启 `model.compile()`**，属于 PyTorch eager 推理。

- **Chipmunk‑FLUX 路径**
  - 代码位置：`models/chipmunk/examples/flux/src/flux`
  - 运行入口：`--backend chipmunk`（`RUN/multi_gpu_launcher.sh` 经 `python -m flux.cli`）
  - 关键特性：
    - 使用 **Chipmunk 稀疏内核** 替换 FLUX 中的 Attention / MLP（列稀疏 + Delta 计算）；
    - 在 `sampling.py` 中额外引入 `patchify / patchify_rope / unpatchify` 等重排逻辑；
    - 在 `util.load_flow_model` 中调用 `model.compile()`（底层为 `torch.compile`），首帧存在编译冷启动；
    - 通过 `chipmunk-config.yml` 以及 `GLOBAL_CONFIG` 控制定制稀疏模式、patchify、Delta‑Cache 等。

简而言之：**两个路径共享相同的 FLUX 参数配置，但中间算子完全不同**——`models/flux` 是 Dense FLUX，`models/chipmunk` 是“FLUX + Chipmunk 稀疏内核 + 编译/流水线优化”。

## 2. 模式语义与当前实现

在 Chipmunk backend 下，`--mode` 的实际含义大致如下：

- `--backend chipmunk --mode chipmunk`
  - 入口：`models/chipmunk/examples/flux/src/flux/cli.py`
  - 使用 `flux.sampling.denoise(...)`，只启用 Chipmunk 稀疏加速，不走 HiCache/Taylor 预测；
  - 这是当前 **推荐、已验证正常** 的 Chipmunk 路径。

- `--backend chipmunk --mode original`
  - 仍然通过 Chipmunk 的采样代码运行，但强制每一步都做 full 计算（不走 Chipmunk 自身的缓存/预测），主要用于 baseline 对比。

- `--backend chipmunk --mode HiCache / Taylor / …`（实验性）
  - 当前实现为：**Chipmunk 稀疏 FLUX + `models/flux` 的 HiCache 实现**：
    - `cli.py` 中根据 `cache_mode` 分支：
      - `cache_mode ∈ {"chipmunk","original"}` → 走 `flux.sampling.denoise`;
      - 其他模式（如 HiCache/Taylor） → 调用 `flux.ideas.denoise_cache(...)`；
    - 也即：时间维度用 HiCache/Taylor 预测，空间维度仍然是 Chipmunk 的稀疏/patchify 结构。
  - 这一组合 **尚未经过系统调参与稳定性验证**，目前已知行为包括：
    - 在 `--mode HiCache` 下，即便 `interval` 调整为小值（例如 1），输出图像仍可能退化为“彩色噪声”；
    - `interval=1` 并不会退化为 “不使用缓存”，因为 HiCache 的决策逻辑只减少预测跨度，而不会自动回退到 `mode=chipmunk` 的纯稀疏前向。

### 关于 `interval=1` 的说明

在 HiCache 的参考实现（`models/flux/src/flux/modules/cache_functions/cache_init.py` + `cal_type.py`）中：

- `interval` 影响的是 `fresh_threshold` / `cal_threshold`，即“多长间隔插一次 full 步”；
- 只要 `cache_mode != chipmunk/original` 且 `taylor_cache=True`，第二步起仍会根据 `cache_counter` 等状态在 full / taylor_cache / ToCa 之间切换；
- 因此：
  - `mode=HiCache, interval=1` ≠ `mode=chipmunk`；
  - 真正关闭 HiCache / 缓存预测的是 `mode=chipmunk` 或 `mode=original`，而不是把 interval 调到 1。

## 3. 已知问题：Chipmunk + HiCache 叠加

在当前版本中，以下配置组合是 **实验性的且存在明显异常**：

- 命令示例：

```bash
bash RUN/multi_gpu_launcher.sh \
  --backend chipmunk \
  --mode HiCache \
  --model_name flux-dev \
  --prompt_file resources/prompts/prompt.txt \
  --limit 200 \
  --width 1024 --height 1024 \
  --num_steps 50 \
  --interval 1 \
  --max_order 2 \
  --first_enhance 3 \
  --hicache_scale 0.6 \
  --run-name chipmunk_hicache_demo
```

- 典型现象：
  - 与 `--backend chipmunk --mode chipmunk` 对比，HiCache 版本的输出大面积退化为彩色噪声，画面细节与语义完全丢失；
  - 即使 `interval` 调整为 1，也未能恢复正常图像。

**推断原因（待系统验证）：**

- HiCache 的 Hermite/Taylor 预测是在 Dense FLUX 上调参得到的，假设了特征尺度与结构与原版一致；
- Chipmunk 稀疏 FLUX 对 Attention/MLP 做了列稀疏、Delta‑Cache 和 patchify 等变换，导致：
  - 多项式预测的输入分布发生变化；
  - Chipmunk 稀疏核内部也在做“类似预测/差分”的工作；  
  两者叠加后很容易出现数值爆炸或严重失配，具体表现就是“彩色噪声图”。

因此，当前可以认为：**“Chipmunk 稀疏 + HiCache 时间预测”的组合尚未稳定，不建议作为正式实验配置使用。**

## 4. 推荐使用方式

根据目前的验证情况，推荐按下面方式使用各个 backend：

1. **正确性 / 论文对比（HiCache 本身）**
   - 使用 `backend=flux` + `mode=HiCache/Taylor/...`；
   - 示例：  
     ```bash
     bash RUN/multi_gpu_launcher.sh \
       --backend flux \
       --mode HiCache \
       --model_name flux-dev \
       --prompt_file resources/prompts/prompt.txt \
       --limit 200 \
       --width 1024 --height 1024 \
       --num_steps 50 \
       --interval 5 --max_order 2 --first_enhance 3 --hicache_scale 0.6 \
       --run-name flux_hicache_demo
     ```
   - 这条路径使用 Dense FLUX + HiCache，是当前“正确实现”的参考。

2. **性能评测（Chipmunk 稀疏加速）**
   - 使用 `backend=chipmunk --mode chipmunk`（或 `original` 作为 baseline）；
   - 示例：  
     ```bash
     bash RUN/multi_gpu_launcher.sh \
       --backend chipmunk \
       --mode chipmunk \
       --model_name flux-dev \
       --prompt_file resources/prompts/prompt.txt \
       --limit 200 \
       --width 1024 --height 1024 \
       --num_steps 50 \
       --guidance 3.5 \
       --chipmunk_config models/chipmunk/examples/flux/chipmunk-config.yml \
       --run-name chipmunk_baseline
     ```
   - 若不想等待 `torch.compile` 的冷启动，可在命令前加：  
     `TORCHDYNAMO_DISABLE=1`。

3. **暂不建议：Chipmunk + HiCache 叠加**
   - 即 `--backend chipmunk --mode HiCache/Taylor/...`；
   - 当前状态：**实验性，画质不稳定，容易出现彩色噪声，即便 interval=1**；
   - 仅建议在明确实验目的、对结果质量有心理预期的情况下使用，并在实验记录中注明“ Chipmunk 稀疏 + HiCache（未校准）”。

## 5. 后续 TODO / 修复计划（备注）

下面是当前规划中的修复路线，后续改动会在此文档同步：

1. **重新接入 HiCache，保证 “chipmunk‑flux 是 FLUX 的替换件”**
   - 目标：把 Chipmunk‑FLUX 视为 `Flux` 的实现细节替换，而不是换一整条采样管线。
   - 具体做法（计划）：
     - 在 Chipmunk 侧保留 `sampling.denoise` 作为“纯 Chipmunk 模式”（`mode=chipmunk`），不动现有行为；
     - 为 HiCache/Taylor 模式新增一条 **Chipmunk 专用的 `denoise_cache` 变体**，逻辑与 `models/flux/src/flux/ideas/cache_denoise.py` 对齐，但：
       - 只依赖 `Flux` 公共接口（`forward(img, txt, vec, pe, cache_dic, current)`），不擅自重写 Chipmunk 的 patchify/Delta 逻辑；
       - 保证各类 `cache_init/cal_type/derivative_approximation/taylor_utils` 调用与 Dense FLUX 一致，只是底层算子换成稀疏实现。

2. **用 `interval=1` 作为“实现是否正确”的验收标准（不加特殊分支）**
   - 目标：在不加入任何显式 if/else 退化逻辑的前提下，当 `mode=HiCache, interval=1` 时，推理轨迹自然收敛为“每步 full”，数值结果与 `mode=chipmunk` 高度一致（随机性允许范围内）。
   - 具体做法（计划）：
     - 视 Chipmunk‑FLUX 为 FLUX 的“实现替换件”，在同一 HiCache 调度下，对比：
       - `backend=flux, mode=HiCache, interval=1`；
       - `backend=chipmunk, mode=HiCache, interval=1`；
       - `backend=chipmunk, mode=chipmunk`。
     - 如果 HiCache 接入是正确的，则 “HiCache+interval=1” 应该在数值上自然接近 chipmunk baseline，而不需要额外的代码分支去“帮忙退化”。
     - 将这组三元对比作为后续改动的回归测试基线：一旦 `interval=1` 下行为明显偏离 chipmunk，即视为 HiCache 接入存在问题，需要回滚或修正。

3. **矫正 “Chipmunk 稀疏 + HiCache 预测” 的数值行为（interval > 1）**
   - 目标：在 `interval>1` 场景下，Chipmunk+HiCache 既能获得明显加速，又不会退化为彩色噪声。
   - 具体做法（计划）：
     - 以 `backend=flux` + HiCache 作为“正确实现基线”，在同一组 prompt/seed 下对比：
       - 特征范数、Delta 分布（各层 / 各阶差分）；
       - 输出图像的 LPIPS / CLIP score / PSNR 等指标；
     - 依据对比结果，调整：
       - HiCache 的 `hicache_scale/max_order/first_enhance`，使其适配 Chipmunk 稀疏特征的分布；
       - 必要时在 Chipmunk config 中为 HiCache 模式关闭或弱化某些 Delta‑Cache / 稀疏策略，避免“预测叠预测”导致不稳定。

4. **保留护栏：在修复完成前继续标记为实验性**
   - 在上述修复完成、经过系统性验证（包括 `interval=1` 等价性检查）之前：
     - 保持本文档中的“实验性 / 不稳定”声明；
     - 在命令模板与 README 中明确标注：`--backend chipmunk --mode HiCache` 仍属实验配置，正式实验应优先使用：
       - 正确性：`--backend flux --mode HiCache/Taylor/...`
       - 性能：`--backend chipmunk --mode chipmunk`

本页会随着实现推进持续更新；如在代码中对 Chipmunk+HiCache 的接入方式作出重大调整，请在对应 PR 中同步修改此文档。 
