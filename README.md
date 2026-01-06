# An Online Evaluation Framework for Mobile GUI Agents


## Requirements and Installation

This work has been tested in the following environment:
* `python == 3.11`
* `uiautomator2`
* `PIL`
* `cv2`
* `numpy`

## Supported Models

| Model                                                                | Model Name        | Organization |
|----------------------------------------------------------------------|-------------------|--------------|
| [UI-TARS-1.5](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)  | `ui-tars-1.5-7b`  | Bytedance    |


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