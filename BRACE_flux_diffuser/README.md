# BRACE-Flux (Diffusers)

基于 HuggingFace `diffusers` 库的 [BRACE](https://github.com/youngkinlon/BRACE) (Barycentric Rational Approximation with Chebyshev Enhancement) FLUX.1-dev 推理加速实现。

## 原理

BRACE 通过**时序特征缓存 (Temporal Feature Caching)** 来减少 Transformer 计算量：

- **Full steps**: 完整计算所有 transformer blocks，同时缓存中间特征到滑动窗口
- **Predicted steps**: 利用缓存的 historical features，通过 **Barycentric Rational Forecasting (BRACE)** 或 Taylor 外推直接预测特征，跳过 attention 和 MLP 计算

BRACE 采用 **Chebyshev 加权 + 重心有理插值**，相比基于导数的 Taylor 外推，对特征的突变和不规则曲线拟合更稳定。

## 目录结构

```
BRACE_flux_diffuser/
├── diffusers_taylorseer_flux.py      # 推理入口脚本
├── cache_functions/
│   ├── cache_init.py                 # 缓存初始化 (支持 Taylor/BRACE/ToCa/Delta 模式)
│   ├── cal_type.py                   # 逐步计算类型调度 (full/predict)
│   └── force_scheduler.py            # 动态阈值调度
├── forwards/
│   ├── flux_forward.py               # FluxTransformer2DModel forward (单卡)
│   ├── xfuser_flux_forward.py        # xfuser 多卡并行 forward
│   ├── double_transformer_forward.py # Double-stream block forward
│   └── single_transformer_forward.py # Single-stream block forward
└── taylorseer_utils/
    └── __init__.py                   # BRACE/Taylor 预测公式核心实现
```

## 快速开始

### 依赖

```bash
pip install diffusers torch transformers huggingface_hub
# 可选: pip install xfuser (多卡分布式推理)
```

### 运行

```bash
python diffusers_taylorseer_flux.py
```

首次运行会自动下载 `black-forest-labs/FLUX.1-dev` 模型（需要接受 FLUX 许可协议）。

### 配置

在 `cache_functions/cache_init.py` 中修改缓存模式：

```python
mode = 'Taylor'  # BRACE 模式 (taylor_cache=True, capacity=2, fresh_threshold=6)
# 可选值: 'original' (随机缓存), 'ToCa', 'Taylor' (BRACE), 'Delta'
```

关键参数：
- `fresh_threshold`: 两次 full step 之间的最大间隔（越大加速越多，质量可能下降）
- `capacity`: BRACE 滑动窗口大小（默认 2）
- `max_order`: Taylor 模式下导数最高阶数

## 引用

如果您使用了此代码，请引用 BRACE 项目：

```bibtex
@article{zhang2025brace,
  title={BRACE: Barycentric Rational Approximation with Chebyshev Enhancement
         for Efficient Diffusion Model Inference},
  author={Zhang, Feiyu and others},
  year={2025},
}
```

## 致谢

- [FLUX.1](https://github.com/black-forest-labs/flux) - Black Forest Labs
- [TaylorSeer](https://github.com/shaoshitong/TaylorSeer) - 原始 TaylorSeer 框架
- [diffusers](https://github.com/huggingface/diffusers) - HuggingFace Diffusers
