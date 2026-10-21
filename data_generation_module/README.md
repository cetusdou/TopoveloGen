# data_generation_module

`data_generation_module` 用于构建拓扑景观、模拟轨迹，并生成可用于 RNA velocity 任务的合成数据。

## 目录

- `docs/`：中文文档（含导航首页）
- `examples/`：示例脚本
- `tests/`：测试代码
- `graph_core.py`：图结构与漂移场核心
- `synthesizer_gene.py`：基因模块与动力学生成
- `presets.py`：统一拓扑/模块预设

## 文档入口

请从 [docs/index.md](./docs/index.md) 开始阅读。

## 快速示例

```bash
python data_generation_module/examples/example_generate.py
```

## Module boundaries

- `graph_core.py` defines the graph geometry and latent drift.
- `synthesizer_gene.py` owns transcription fields and RNA-state integration.
- `observation.py` converts continuous abundances to sampled counts using
  capture variation, Poisson or negative-binomial sampling, and dropout.
- `presets.py` assembles reusable topology and gene-module configurations.

The public `GeneSynthesizer` methods still call the observation module, so
existing imports through `data_generation_module` remain valid.
