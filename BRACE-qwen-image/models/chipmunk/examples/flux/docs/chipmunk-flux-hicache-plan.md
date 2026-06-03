# Chipmunk‑FLUX + HiCache 接入与验收方案草案

> 本文档只聚焦两件事：
> 1. 如何把 Chipmunk‑FLUX 当成原生 FLUX 的“实现替换件”，在其上正确接入 HiCache 管线；
> 2. 如何用 `interval=1` 做为不添加任何特殊分支的“自检/验收用例”，判断接入是否正确。

## 1. 接入原则：Chipmunk‑FLUX 视作 FLUX 的实现替换件

目标：在代码层面，把 Chipmunk‑FLUX 看成是 `Flux` 的一种实现，而不是一套完全独立的采样逻辑。HiCache 逻辑只关心 `Flux` 的公开接口，不感知底下是 Dense 还是 Sparse。

### 1.1 意图

- **上层（HiCache / Taylor）**：  
  保持 `models/flux/src/flux/ideas/cache_denoise.py` +  
  `models/flux/src/flux/modules/cache_functions/*` 的调度与状态机不变。

- **底层（Flux 实现）**：  
  - `backend=flux` 时，使用 Dense 版 FLUX（当前 `models/flux` 中的实现）；  
  - `backend=chipmunk` 时，使用 Chipmunk‑FLUX（`models/chipmunk/examples/flux/src/flux`），在相同接口下替换内部 Attention/MLP 为稀疏/patchify 版本。

换句话说：**HiCache 只负责“什么时候 full / 什么时候预测”**，  
底层到底是 Dense 还是 Sparse（Chipmunk），都应该通过统一的 `Flux` 接口来屏蔽。

### 1.2 具体接入思路（计划）

1. 保持 `models/flux/src/flux/ideas/cache_denoise.py:denoise_cache` 作为“规范参考实现”，不直接改动该文件；
2. 在 Chipmunk 路径下新增一个 `denoise_cache` 变体（例如放在 `models/chipmunk/examples/flux/src/flux/ideas/cache_denoise_chipmunk.py`），满足：
   - 函数签名与原版 `denoise_cache` 对齐（`img/img_ids/txt/txt_ids/vec/timesteps/guidance/cache_mode/...`）；
   - 内部仍调用 Chipmunk‑FLUX 的 `Flux.forward(...)`，并让 `forward` 尊重 `cache_dic/current`（即 HiCache 的决策）；
   - 不擅自重写 Chipmunk 的 patchify/Delta‑Cache 逻辑，只在调用方式上对齐 Dense FLUX 的接口和语义。
3. 在 `models/chipmunk/examples/flux/src/flux/cli.py` 中：
   - `mode in {"chipmunk","original"}` 继续走现有 `sampling.denoise`（纯 Chipmunk 模式）；  
   - `mode in {"Taylor","HiCache",...}` 切换成调用 Chipmunk 版 `denoise_cache`，而不是 Dense 版的实现。

这一步的验收标准是：在不使用任何特别“护栏”的前提下，Dense FLUX 与 Chipmunk‑FLUX 在相同 HiCache 配置下的行为模式一致（数值差异仅来自稀疏实现本身）。

## 2. 用 `interval=1` 作为不加分支的“自检用例”

需求背景：从 HiCache 的算法设计上讲，`interval=1` 意味着 **每一步都是 full 步**，不会真正触发“间隔跳步预测”。在这种配置下：

- 对 Dense FLUX：`mode=HiCache, interval=1` 应该与 `mode=original` 的采样轨迹尽量接近；
- 对 Chipmunk‑FLUX：`mode=HiCache, interval=1` 应该与 `mode=chipmunk` 的采样轨迹尽量接近；

这里的“接近”是在不添加任何 `if interval==1: ...` 之类分支的前提下，自然产生的结果。

### 2.1 验收目标

1. **Dense FLUX 侧**  
   - 在相同 prompt/seed 下，对比：
     - A：`backend=flux, mode=original`  
     - B：`backend=flux, mode=HiCache, interval=1`  
   - 要求：B 的图像质量与 A 基本一致（可用 LPIPS/CLIP 等指标和肉眼双重评估），说明 HiCache 在 `interval=1` 时自然退化为“每步 full”，没有额外预测误差。

2. **Chipmunk‑FLUX 侧**  
   - 在相同 prompt/seed 下，对比：
     - C：`backend=chipmunk, mode=chipmunk`  
     - D：`backend=chipmunk, mode=HiCache, interval=1`（使用 Chipmunk 版 `denoise_cache`）  
   - 要求：D 的图像质量与 C 基本一致，说明在 Chipmunk‑FLUX 上接入 HiCache 没有破坏 `interval=1` 的“每步 full”语义。

### 2.2 使用方式（作为回归检查）

- 在后续修改 Chipmunk+HiCache 接入逻辑时，始终保留上述 A/B/C/D 四组对比作为回归测试：
  - 一旦发现 `interval=1` 时的 HiCache 结果（B 或 D）出现明显偏差 / 噪声，而 baseline（A 或 C）正常，即视为本次改动破坏了 HiCache 正确性；
  - 通过这种“无特殊分支的 interval=1 对比”，避免用额外 if/else 把问题“兜底”掉，从而掩盖上游实现错误。

这份方案文档仅作为开发/调试 HiCache‑on‑Chipmunk 时的设计约束和验收基线，具体代码变更应在相应 PR 中引用并同步更新本文件。 
