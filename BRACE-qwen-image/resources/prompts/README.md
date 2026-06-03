# Prompts

本目录用于存放推理脚本的 prompt 列表与公开的基准 prompt 数据集。

- 默认 prompt 列表：`prompt.txt`（`scripts/sample.sh` / `RUN/multi_gpu_launcher.sh` 默认使用）
- 数据集：
  - `datasets/PartiPrompts.tsv`：PartiPrompts（带分类/难度等字段的 TSV 原始表）

如果你只需要 PartiPrompts 的纯 prompt 文本（每行一个 prompt），可从 TSV 第一列提取：

```bash
cut -f1 resources/prompts/datasets/PartiPrompts.tsv | tail -n +2 > partiprompts.txt
```
