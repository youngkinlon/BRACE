# Chipmunk + FLUX 环境与配置清单

> 目标：在本机 GPU 开发机上复现 `CV/HiCache-Flux/models/chipmunk/examples/flux` 的 Chipmunk‑FLUX 推理环境，为后续接入 HiCache 做准备。

## 硬件与系统要求

- GPU：Hopper 架构（H100 / H200），本机为 H200 ✅  
- CUDA Driver 显示版本：`nvidia-smi` 中 `CUDA Version >= 12.4`（当前为 12.4 ✅）  
- CUDA Toolkit：`nvcc --version` 显示 12.4+（当前为 12.4 ✅）

## Conda 环境约定（推荐基于现有 eval 环境克隆）

- 环境父目录：`/mnt/innovator/data/fengLang/envs/`
- 机器上已存在一个经过实际评测的环境：  
  `eval-scicode-conda`（Python 3.10，`torch 2.8.0+cu128` 工作正常）
- 推荐做法：**从该环境克隆出 Chipmunk‑FLUX 专用环境**，既复用稳定的 PyTorch，又保持隔离：

```bash
source /mnt/innovator/code/fengLang/AnyWorkspace/installed/miniconda3/etc/profile.d/conda.sh

# 如存在旧的 chipmunk-flux 环境，可先删除
rm -rf /mnt/innovator/data/fengLang/envs/chipmunk-flux

conda create -p /mnt/innovator/data/fengLang/envs/chipmunk-flux --clone /mnt/innovator/data/fengLang/envs/eval-scicode-conda -y

conda activate /mnt/innovator/data/fengLang/envs/chipmunk-flux
```

> 直接在全新环境里 `conda install pytorch` 时，曾遇到过 `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` 链接错误，克隆已验证环境可以规避这类底层问题。

## Python / PyTorch 版本

- Python：`eval-scicode-conda` 中当前为 3.10.x（满足 FLUX 要求 `>=3.10`）  
- PyTorch：`eval-scicode-conda` 中当前为 `2.8.0+cu128`，可直接复用  
  - 在克隆出的 `chipmunk-flux` 环境中验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 安装 Chipmunk 主仓库

在新环境中：

```bash
cd $PROJECT_ROOT/models/chipmunk
git submodule update --init --recursive

# 按官方建议：构建自定义 CUDA kernel 并安装依赖
pip install -e . --no-build-isolation
```

注意：

- 安装阶段如需访问 PyPI / Hugging Face，请先按服务器文档启用 GPU 代理：  
  `source BrightPedia/Servers/inno_yidian_server_use/enable_gpu_proxy.sh`  
- 运行推理时建议关闭代理，并设置 `NO_PROXY` 覆盖 `localhost,127.0.0.1,::1`（避免后续本地/内网 API 被代理劫持）：  
  `source BrightPedia/Servers/inno_yidian_server_use/disable_gpu_proxy.sh`

### 编译时间与常见日志说明

- 第一次安装 Chipmunk 时，会从源码编译自定义 CUDA kernel 和 `flash_attn`，日志中会出现类似：
  - `Building editable for chipmunk (pyproject.toml) ...`
  - `Building wheel for flash_attn (setup.py) ...`
- 这一步可能持续数分钟属于正常现象，请耐心等待直至出现：
  - `Successfully built chipmunk flash_attn`
  - `Successfully installed chipmunk-1.0.0 ... flash_attn-...`
- 安装过程中可能看到关于 `https://pypi.ngc.nvidia.com` 的 SSL 警告（`WRONG_VERSION_NUMBER`），目前可以忽略，因为实际依赖已经从清华源安装完成。

## 安装 Chipmunk‑FLUX 示例代码

在同一环境中：

```bash
cd $PROJECT_ROOT/models/chipmunk/examples/flux
pip install -e .
```

这一步会安装 FLUX 推理代码及其依赖（`transformers`、`huggingface-hub`、`invisible-watermark` 等）。

## 模型权重与路径约定

建议将模型统一放在：`/mnt/innovator/model/fengLang/`

Chipmunk‑FLUX 至少需要：

- FLUX.1‑dev flow 模型：`black-forest-labs/FLUX.1-dev` 中的 `flux1-dev.safetensors`
- 自编码器（AE）：同仓中的 `ae.safetensors`

在当前代码树中，以下路径已作为默认路径写入 `examples/flux/src/flux/util.py`：

- FLUX.1‑dev flow：`$PROJECT_ROOT/resources/weights/flux.dev/flux1-dev.safetensors`
- FLUX.1‑dev / ‑schnell 共用 AE：`$PROJECT_ROOT/resources/weights/flux.dev/ae.safetensors`
- FLUX.1‑schnell flow：`$PROJECT_ROOT/resources/weights/FLUX.1-schnell/flux1-schnell.safetensors`

如保持上述目录结构不变，**不需要设置任何环境变量**，Chipmunk‑FLUX 会直接使用这些本地权重。

若需覆盖这些默认路径，可以按需设置（可选）：

```bash
export FLUX_DEV=/path/to/custom/flux1-dev.safetensors
export FLUX_SCHNELL=/path/to/custom/flux1-schnell.safetensors
export AE=/path/to/custom/ae.safetensors
```

此外，FLUX 的文本编码模型（T5 与 CLIP）在 HiCache‑Flux 中也已下载好，Chipmunk‑FLUX 的实现已默认优先复用这些本地目录：

- T5 文本编码器：`CV/HiCache-Flux/resources/weights/t5-v1_1-xxl`
- CLIP 文本编码器：`CV/HiCache-Flux/resources/weights/clip-vit-large-patch14`

如需显式指定自定义路径，可设置以下环境变量（可选）：

```bash
export FLUX_T5_LOCAL_PATH=$PROJECT_ROOT/resources/weights/t5-v1_1-xxl
export FLUX_CLIP_LOCAL_PATH=$PROJECT_ROOT/resources/weights/clip-vit-large-patch14
```

> NSFW 过滤模型 `Falconsai/nsfw_image_detection` 也会在首次运行时通过 `transformers.pipeline` 下载，如无外网需提前镜像到本机或在有代理的会话中预下载。

## 测试 Chipmunk‑FLUX 推理

在已激活的 `chipmunk-flux` 环境中：

### 一次性生成一张图片（推荐起步方式）

```bash
cd $PROJECT_ROOT/models/chipmunk/examples/flux
export PROMPT="A very cute cartoon chipmunk dressed up as a ninja holding katanas"
python -m flux.cli --name flux-dev --prompt "$PROMPT" --chipmunk-config ./chipmunk-config.yml
```

生成完成后，日志中会出现类似：

```text
Done in X.XXXs. Saving output/img_0.jpg
```

图片默认保存在当前目录下的 `output/` 目录，例如：`CV/HiCache-Flux/models/chipmunk/examples/flux/output/img_0.jpg`。

### 进入交互式多轮生成（可选）

```bash
cd $PROJECT_ROOT/models/chipmunk/examples/flux
export PROMPT="A very cute cartoon chipmunk dressed up as a ninja holding katanas"
python -m flux.cli --name flux-dev --prompt "$PROMPT" --loop --chipmunk-config ./chipmunk-config.yml
```

加载完成后，会看到：

```text
Next prompt (write /h for help, /q to quit and leave empty to repeat):
```

- 直接回车：重复当前 `PROMPT` 再生成一张图；  
- 输入新文案再回车：用新文案生成；  
- 输入 `/q`：退出。

每次生成完成都会在 `output/` 目录追加写入 `img_{idx}.jpg`。

检查点：

- 能成功加载模型与 AE（无权重路径报错）；  
- 第 1–2 张图像较慢（`torch.compile` 冷启动），第 3 张之后单张时间 ≈ 5s（1280×768，默认稀疏度）；  
- GPU 利用率明显提升，显存占用随 Chipmunk 配置变化合理。

## 后续：接入 HiCache 的注意事项（预留）

- 本仓当前 Chipmunk‑FLUX 代码在 `examples/flux/src/flux`；  
- HiCache‑Flux 代码在 `CV/HiCache-Flux/models/flux/src/flux`，二者各自 fork 了一份 FLUX 逻辑；  
- 在尝试合并 HiCache 时，建议：
  - 以当前仓的 Chipmunk‑FLUX 为基础；  
  - 按 HiCache‑Flux 的 `cache_dic` / `current` 接口逐层移植；  
  - 保持 `LayerCounter.cur_inference_step` 与 HiCache 的 step 计数对齐；  
  - 在本文件记录额外依赖或新配置项。

如有新的环境约定（例如新增 `conda env` 名称、模型目录结构调整），请同步更新本清单。
