# MobileBench-OL：移动端 GUI 智能体在线评估框架

![Pipeline](assets/pipeline.png)

本项目提供一个在真实 Android 设备上运行的移动端 GUI 智能体评测框架，支持多种模型/智能体，完成任务轨迹采集与自动化评估。

- 真机交互：通过 uiautomator2 驱动真实设备执行操作
- 统一接口：适配 UI-TARS、GPT-4o、M3A/T3A、Qwen2.5-VL、Mobile Agent V2 等
- 自动评估：基于规则/xpath 对已采集轨迹进行一致性评估
- 可复现：任务与结果按目录落盘，支持断点续跑与多轮补跑

---

## 环境与安装

已在如下环境测试：
- Python 3.10
- Android 真机（已开启开发者选项与 USB 调试），可通过 USB 或 Wi‑Fi ADB 连接

Python 依赖（核心框架部分）：
- uiautomator2
- Pillow (PIL)
- opencv-python (cv2)
- numpy

安装示例：
```bash
pip install uiautomator2 pillow opencv-python numpy
```

### Mobile Agent V2 说明
如需使用 **Mobile Agent V2**，请按官方环境配置指南进行安装与部署：[https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2)

---

## 支持的模型/智能体

| 模型 | `model.name` | 组织/来源 |
|---|---|---|
| [UI-TARS-1.5](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) | `uitars_1_5` | ByteDance |
| GPT-4o | `gpt4o` | OpenAI |
| M3A（来自 [AndroidWorld](https://github.com/google-research/android_world/tree/main)） | `m3a` | Google |
| T3A（文本模式，使用 GPT-4o） | `t3a` | - |
| Qwen2.5-VL | `qwen2_5vl` | Qwen |
| [Mobile Agent V2](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2) | `mobileagentv2` | Qwen |

说明：
- M3A/T3A 通过 Azure OpenAI/GPT‑4o 调用，需在配置文件中提供 Azure 相关参数。
- Mobile Agent V2 需在配置文件中提供 API URL、Token 以及图像理解（Qwen VL）相关参数。

---

## 快速开始

### 1) 连接设备
- 启用开发者选项与 USB 调试
- 通过 `adb devices` 获取设备序列号，将其填入配置文件的 `[device] id`

### 2) 选择或编写配置文件
仓库已提供多份样例配置（位于 `config/`）：
- `interact.conf`（UI‑TARS 示例）
- `interact_uitars15_*.conf`、`interact_gpt4o_*.conf`、`interact_m3a_*.conf`、`interact_t3a_*.conf`、`interact_qwen2_5vl_*.conf`、`interact_mobileagentv2_*.conf` 等
- 评估配置：`evaluate.conf`

### 3) 采集任务轨迹（交互模式）
```bash
python3 run.py --mode interact \
  --config config/interact.conf \
  --subset base \
  --output results/uitars1.5_1114
```
参数含义：
- `mode`：`interact`（采集轨迹）或 `evaluate`（评估既有轨迹）
- `config`：配置文件路径（见下文“配置项说明”）
- `subset`：从基准集选择任务子集，支持 `[base, long-tail, long-horizon, gui-reasoning, noise-robust]`
- `output`：结果输出目录（示例：`results/round1`）

运行时行为（参考 run.py）
- 多轮补跑：按 `retry.retry_rounds` 多轮执行，仅对未成功任务继续尝试（run.py:51‑116）
- 断点续跑：输出目录下维护 `result_list.txt` 缓存（task_executor.py:277‑314）
- 噪声鲁棒评测：当 `subset` 包含 `noise` 时，将依次以 `repeat / unexecuted / delay / popup` 四种噪声类型执行（run.py:117‑156）

### 4) 评估轨迹（评估模式）
```bash
python3 run.py --mode evaluate --config config/evaluate.conf
```
评估配置示例（`config/evaluate.conf`）：
```ini
[evaluation]
trajectory=results/round1
rule=data/top12.csv
reset=false
type=xpath
```
- `trajectory`：待评估轨迹目录（即交互模式的 `output`）
- `rule`：评估规则文件（CSV），示例 `data/top12.csv`
- `reset`：是否执行 reset 评估逻辑
- `type`：当前支持 `xpath`

---

## 配置项说明

以下示例以 `config/interact.conf` 为准（具体键可能因模型而异）：

```ini
[device]
id=设备序列号（通过 adb devices 获取）

[model]
name=模型名称（如 uitars_1_5 / gpt4o / m3a / t3a / qwen2_5vl / mobileagentv2）
url=模型服务的 API URL（不同模型对应不同含义）
# 对于 mobileagentv2：还需 token / qwen_api / caption_model

[retry]
retry_rounds=总执行轮数（对未成功任务继续尝试）
connect_retry=连接失败的自动重连次数
fail_retry=单任务失败的重试次数

[reset]
reset=true/false （是否按 reset 逻辑执行）

[operation]
max_steps=单任务最大交互步数（默认按 golden_steps*3 限制）
back_times=（保留项）
sleep_seconds_per_act=动作间等待秒数

[task]
# 若使用命令行 --output，会覆盖该项
task_file=data/<任务CSV>
output=results/<结果目录>
```

---

## 模型使用示例

以下示例均基于已提供的样例配置（见 config/），请先按需填充设备 id 与模型服务参数。

### UI‑TARS‑1.5
```bash
python3 run.py --mode interact \
  --config config/interact_uitars15_base.conf \
  --subset base \
  --output results/uitars1.5_run
```


提示：若进行噪声鲁棒评测，可将 `--subset` 设为 `noise-robust`，框架将依次执行 `repeat / unexecuted / delay / popup` 四种噪声类型，输出目录不变（run.py:117‑156）。

---

## 目录结构

```text
.
├── assets/
│   └── pipeline.png
├── config/
│   ├── interact.conf
│   ├── interact_*.conf
│   └── evaluate.conf
├── data/
│   ├── top12.csv
│   ├── longtail.csv
│   ├── MobileBench-OL - Long-Horizon.csv
│   └── ...
├── mobilebench/
│   ├── eval/
│   │   ├── evaluator_xpath.py
│   │   └── evaluator_xpath_step_ratio.py
│   ├── models/
│   │   ├── llm_core_*.py
│   │   └── execute.py
│   └── utils/
│       ├── task_executor.py
│       ├── agent*.py
│       ├── adb_executor.py
│       └── ...
├── MobileAgent_new/
│   └── Mobile-Agent-v2/
├── run.py
├── README.md
└── README_EN.md
```

- assets/: 项目图片与流程图。
- config/: 交互/评估配置文件（多个模型的样例）。
- data/: 基准任务 CSV（base/long-tail/long-horizon/gui-reasoning/noise-robust 等）。
- mobilebench/: 核心框架代码
  - eval/: 评估器（xpath/step-ratio 等）
  - models/: 各模型的 LLM 封装与调用
  - utils/: 设备/执行/解析等工具（如 task_executor、adb_executor 等）
- MobileAgent_new/Mobile-Agent-v2/: 可选的 Mobile Agent V2 集成代码与资源。
- run.py: 入口脚本（交互/评估）。
- README.md: 中文文档；README_EN.md: 英文文档。

---

## 数据与输出结构

- 任务 CSV（示例列，至少需包含如下字段）：
  - `task_identifier`，`goal`，`adb_home_page`，`golden_steps`，`key_nodes`
  - 可选：`reset_xpath`，`reset_query`
  - 加载逻辑参考：mobilebench/utils/task_executor.py:319‑337

- 结果目录结构：
  - `results/<run_name>/result_list.txt`：以空格分隔的 `task_id,success` 记录（task_executor.py:305‑314）
  - `results/<run_name>/<task_id>/trajectory.json`：每个任务的轨迹详情（action / image / response / summary / success）（task_executor.py:297‑304）

- 通过率打印：执行结束后输出 `Overall pass rate: xx.xx%`（run.py:114‑116）

---

## 常见问题与排查

- 连接失败（Unable to connect / uiautomator2 不可用）
  - 检查设备是否通过 `adb devices` 正确识别
  - 更换数据线/端口，或使用 Wi‑Fi ADB
  - 多次重连由 `connect_retry` 控制（task_executor.py:60‑77, 363‑379）

- 无任务或 CSV 字段缺失
  - 确认 `task_file` 指向的 CSV 存在且包含必需列（task_executor.py:319‑337）

- 评估无结果
  - 检查 `evaluation.trajectory` 指向交互输出目录；`evaluation.rule` 为规则 CSV

- 噪声鲁棒模式（`subset` 含 `noise`）
  - 将依次执行 `repeat / unexecuted / delay / popup` 四种噪声类型，目录不变（run.py:117‑156）

---

## 许可协议

本项目的数据集遵循 [Creative Commons Attribution‑NonCommercial‑ShareAlike 4.0 International (CC BY‑NC‑SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可。

本项目的源代码遵循 [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) 许可。

许可摘要：
- 署名（Attribution）：需注明来源，提供许可链接，并说明修改
- 非商业（NonCommercial）：禁止商业用途
- 相同方式共享（ShareAlike）：衍生作品须以相同许可发布

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 引用

如您在研究/产品中使用了本仓库的资源，请引用：
```
@article{wu2026mobilebench,
  title={MobileBench-OL: A Comprehensive Chinese Benchmark for Evaluating Mobile GUI Agents in Real-World Environment},
  author={Wu, Qinzhuo and Yang, Zhizhuo and Li, Hanhao and Gao, Pengzhi and Liu, Wei and Luan, Jian},
  journal={arXiv preprint arXiv:2601.20335},
  year={2026}
}
```
