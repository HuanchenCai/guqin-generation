# 古琴生成实验

用 Stable Audio 3 Medium + DoRA 适配器做古琴音频续写。本仓库只跟踪代码、实验记录和小型元数据；音频、SAME 潜表示、模型检查点和上游代码都不入库。

## 目录

- `work/*.py` — 数据筛选、分窗、编码、训练、生成、评估脚本
- `work/guqin_aligned_412/` — 412 首整曲的分窗清单、分组、噪声扫描结果（音频与潜表示不入库）
- `outputs/*/` — 各轮试听页 `index.html`、实验记录和样本记录（音频不入库）

## 未入库的依赖（`work/` 下的上游克隆，均未修改）

| 目录 | 来源 | commit |
|---|---|---|
| `stable-audio-3` | https://github.com/Stability-AI/stable-audio-3 | `7f3a3a0` |
| `ACE-Step-1.5` | https://github.com/ace-step/ACE-Step-1.5 | `ca1e85f` |
| `magenta-realtime-torch` | https://github.com/multimodalart/magenta-realtime-torch | `6d076ba` |

训练/推理用 `work/stable-audio-3/.venv` 的 Python。原始录音在 `Y:\Music\古琴曲`。
