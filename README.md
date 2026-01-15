# An Online Evaluation Framework for Mobile GUI Agents

![Pipeline](assets/pipeline.png)

## Requirements and Installation

This work has been tested in the following environment:
* `python == 3.10`
* `uiautomator2`
* `PIL`
* `cv2`
* `numpy`
* `csv`

### Note for Mobile Agent V2
If you plan to use **Mobile Agent V2**, please follow the official environment configuration instructions provided [here](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2).

## Supported Models

| Model                                                                | Model Name        | Organization |
|----------------------------------------------------------------------|-------------------|--------------|
| [UI-TARS-1.5](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)  | `uitars_1_5`      | Bytedance    |
| GPT-4o                                                               | `gpt4o`           | OpenAI       |
| M3A(from [AndroidWorld](https://github.com/google-research/android_world/tree/main))    | `m3a`             | Google       |
| [Mobile Agent V2](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2) | `mobileagentv2` | Qwen |
 


## Model Deployment


## Trajectory Collection

```commandline
python3 run.py --mode interact --config config/interact.conf --subset base --output results/uitars1.5_1114
```

**parameters**
- `mode`:  `[interact, evaluate] `
- `config`: `config file path `
- `subset`: `[base,long-tail,long-horizon,gui-reasoning,noise-robust]`
- `output`: `output path `






**device**
- `id (str)`:

**model**
- `name (str)`:
- `url (str)`:

**inference**
- `max_steps (int)`:
- `back_times (int)`:
- `sleep_seconds_per_act (int)`:

**task**
- `task_file (str)`:
- `output (str)`:


## Evaluation

```commandline
python3 run.py --mode evaluate --config config/evaluate.conf
```
**evaluation**
- `type (str)`:
- `trajectory (str)`:
- `rule (str)`:


## Citation

If you find the resources in this repository helpful, please cite as:
```
```
