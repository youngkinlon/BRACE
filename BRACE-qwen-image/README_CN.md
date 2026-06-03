<p align="center">
  <h1 align="center">HiCache</h1>
  <p align="center">
    <em>用于 Taylor 式"缓存-预测"扩散加速的可插拔 Scaled-Hermite 升级方案</em>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2508.16984"><img src="https://img.shields.io/badge/arXiv-2508.16984-b31b1b.svg" alt="arXiv"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="License"></a>
    <a href="https://iclr.cc/Conferences/2026"><img src="https://img.shields.io/badge/ICLR-2026-blueviolet.svg" alt="ICLR 2026"></a>
  </p>
  <p align="center">
    <a href="./README.md">English</a> | 简体中文
  </p>
  <p align="center">
    <a href="#快速开始">快速开始</a> •
    <a href="#支持的后端">支持的后端</a> •
    <a href="#背后的故事">背后的故事</a> •
    <a href="#使用方法">使用方法</a> •
    <a href="#实验结果">实验结果</a> •
    <a href="#开源协议">开源协议</a>
  </p>
</p>

---

## 📄 摘要

扩散模型在内容生成方面取得了显著成功，但由于迭代采样的计算成本高昂而受到限制。虽然近期的特征缓存方法通过时序外推来加速推理，但这些方法由于无法建模特征演化的复杂动态而遭受严重的质量损失。

为了解决这个问题，本文提出了 **HiCache**（**H**erm**i**te 多项式特征**缓存**），这是一个无需训练的加速框架，通过将数学工具与经验特性对齐从根本上改进了特征预测。我们的关键洞察是：扩散 Transformer 中的特征导数近似呈现多元高斯特性，这促使我们使用 Hermite 多项式——高斯相关过程的理论最优基函数。此外，我们引入了**双重缩放机制**，在保持预测精度的同时确保数值稳定性，该机制单独应用于 TaylorSeer 时也很有效。

大量实验证明了 HiCache 的优越性：在 FLUX.1-dev 上实现了 **5.55× 加速**，同时超越了基准质量，在文本生成图像、视频生成和超分辨率任务中保持强劲表现。此外，HiCache 可以自然地添加到先前的缓存方法中以增强其性能，*如*将 ClusCa 的图像奖励从 0.9480 提升到 0.9840。

---

## ✨ 特性

- **HiCache 加速** — 基于 Hermite 多项式的高效扩散采样特征缓存
- **多后端支持** — FLUX、Qwen-Image、Qwen-Image-Edit、Chipmunk-Flux
- **灵活部署** — 包含单 GPU 和多 GPU 启动器
- **易于集成** — 简单的 `pip install -e ".[all]"` 安装

> [!NOTE]
> 本仓库**不包含**模型权重/检查点。

> [!TIP]
> **待办事项**
> - [ ] 发布基于 Inf-DiT 的超分辨率加速代码

---

## 🧩 背后的故事

这篇 HiCache，是我人生第一篇顶会“一作”。科研是一场充满不确定性的延迟满足：工作在 2025 年 7 月底做完，先投 AAAI 被拒，随后改投 ICLR，并在 Rebuttal 之后最终被 ICLR 2026 接收；仅“做完 → 榜上有名”，往往就会隔着大半年。

HiCache 背后，也是我从「保研复旦金融专硕」到「申请 AI 普博 / AI 预备博士生」的故事下半场。2025 年 3 月起我决定放弃就业、转向申博，并开始在张老师 EPIC Lab 实习；在研一下仍有金融课程的情况下，我长期在复旦（爱久公寓/邯郸）与上交徐汇之间奔波通勤。那段时间实验室的伙伴与氛围，给了我继续走下去的支撑。

在此之前，我也经历过从金融转向工程：大三保研后为了量化/研究等需求补齐计算机能力，花了七八个月自学后端开发，并在大四暑假去字节做过后端开发实习；这段经历也让我更清楚自己想走的方向。

在工作层面，HiCache 也“差点不会诞生”：它来自一个可能会被师弟（郑世康）抛弃的 Hermite 方向 PlanB。我们先做 TaylorSeer 的超分加速，在视频模型上反复碰壁后转向更强的缓存预测算法；世康实验出 FOCA 与 Hermite 两条线，最终选择主推 FOCA。彼时我面临“必须拿到一作来申请 AI 普博”的压力；在张老师的启发下，我把 Hermite 的思路补齐并做了关键创新（例如 HiCache 的基函数双重缩放 trick），于是从 Hermite 诞生了 HiCache。同期我们也在四个月内完成两篇顶会工作：FOCA（AAAI 中稿）与 HiCache（AAAI 被拒后转投 ICLR 并接收）。

更多个人经历与科研路径记录（FinTechMath）：
- 知乎文章：<https://zhuanlan.zhihu.com/p/1999383695040213067>
- 小红书帖子：<https://www.xiaohongshu.com/discovery/item/6977f8e20000000021029ef5?source=webshare&xhsshare=pc_web&xsec_token=ABpAzypbjtoTrduA2sdYDFRliw8EG3UeRJmUN4Q1f2aIU=&xsec_source=pc_share>

---

## 🚀 快速开始

```bash
# 创建并激活虚拟环境
python3.10 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -e ".[all]"
```

---

## 🎯 支持的后端

| 后端              | 描述                            | 状态 |
| ----------------- | ------------------------------- | ---- |
| **FLUX**          | 文本生成图像                     | ✅    |
| **Qwen-Image**    | 文本生成图像                     | ✅    |
| **Qwen-Image-Edit** | 图像编辑                       | ✅    |
| **Chipmunk-Flux** | 后端实验                         | ✅    |

---

## 📖 使用方法

### 统一启动器（单 GPU / 多 GPU）

#### FLUX

```bash
# 单 GPU: --gpus 0 ; 多 GPU: --gpus 0,1
bash RUN/multi_gpu_launcher.sh --backend flux --mode HiCache --gpus 0 \
  --prompt_file resources/prompts/prompt.txt --output_dir outputs/hicache
```

#### Qwen-Image

```bash
# 推荐：传入你的 Qwen 环境的 python 解释器
bash RUN/multi_gpu_launcher.sh --backend qwen-image --python /path/to/python -- \
  --model_path /path/to/Qwen-Image --output_dir outputs/qwen_image
```

---

## 📊 实验结果

### 文本生成图像（FLUX.1-dev）

HiCache 实现了 **5.55× 加速**，同时图像质量优于基准方法。

<p align="center">
  <img src="docs/figures/compare.png" alt="文本生成图像对比" width="100%">
</p>

<p align="center"><em>不同提示的定性对比。HiCache 相比 TaylorSeer 和其他基准生成了更高保真度的结果。</em></p>

### 细节保留与风格一致性

<p align="center">
  <img src="docs/figures/detail_comparison.png" alt="细节对比" width="100%">
</p>

<p align="center"><em>高频细节保留：HiCache 比竞争方法更好地保留了精细细节。</em></p>

<p align="center">
  <img src="docs/figures/time_shift_contra.png" alt="风格一致性" width="100%">
</p>

<p align="center"><em>在不同加速比下保持一致的风格和干净的背景。</em></p>

### 文本生成视频（HunyuanVideo）

<p align="center">
  <img src="docs/figures/video.png" alt="视频生成对比" width="80%">
</p>

<p align="center"><em>相比其他加速方法，具有更好的时间一致性和帧质量。</em></p>

### 图像超分辨率（Inf-DiT）

<p align="center">
  <img src="docs/figures/ntire_results_comparison.png" alt="超分辨率对比" width="60%">
</p>

<p align="center"><em>HiCache 实现了约 5.93× 理论加速，同时保持了可比的 PSNR 和 SSIM。</em></p>

---

## 📦 权重与路径

### FLUX 权重

将 FLUX 权重放在 `resources/weights/` 下（git 不跟踪）：

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir resources/weights/FLUX.1-dev \
  --local-dir-use-symlinks False
```

### Qwen 权重

Qwen-Image / Qwen-Image-Edit 权重应通过以下方式提供：
- `--model_path` 参数，或
- 环境变量：`QWEN_IMAGE_MODEL_PATH`

---

## 📁 项目结构

```
HiCache/
├── models/                 # 模型实现
│   ├── flux/              # FLUX 后端
│   ├── qwen_image/        # Qwen-Image 后端
│   ├── qwen_image_edit/   # Qwen-Image-Edit 后端
│   ├── chipmunk/          # Chipmunk-Flux 实验
│   └── hicache_fast_impl.py  # 核心 HiCache 实现
├── scripts/               # 实用脚本
├── RUN/                   # 启动脚本
└── resources/             # 提示词、权重、许可证
```

---

## 📄 开源协议

| 组件              | 协议                                                            |
| ----------------- | -------------------------------------------------------------- |
| **本仓库**        | [GPL-3.0](./LICENSE)                                           |
| **模型权重**      | 见 `resources/third_party/model_licenses/`                     |
| **第三方代码**    | 见 `resources/third_party/code_licenses/`（如 Apache-2.0）     |

---

## 📚 引用

如果您觉得 HiCache 有用，请引用我们的论文：

```bibtex
@inproceedings{feng2026hicache,
  title={HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration},
  author={Feng, Liang and Zheng, Shikang and Liu, Jiacheng and Lin, Yuqi and Zhou, Qinming and Cai, Peiliang and Wang, Xinyu and Chen, Junjie and Zou, Chang and Ma, Yue and Zhang, Linfeng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
