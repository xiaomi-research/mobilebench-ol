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
- `mode`:  `[interact, evaluate] `  Running mode. "interact" means the agent will interact with the environment to collect GUI trajectories; "evaluate" means it will evaluate existing  GUI trajectories.
- `config`: `config file path `  Path to the configuration file that contains all other parameter settings.
- `subset`: `[base,long-tail,long-horizon,gui-reasoning,noise-robust]`  Selects which subset of tasks to run from the benchmark. Different subsets test different agent abilities.
- `output`: `output path `  Directory path where the collected trajectory data will be saved. (e.g., "results/round1").


### Config file settings

**device**
- `id (str)`: The device id of physical device. Input "adb devices" in the terminal to obtain device id.

**model**
- `name (str)`: Name of the model/agent used for trajectory collection (e.g., "uitars_1_5", "gpt-4o").
- `url (str)`: Input your own API URL.

**inference**
- `max_steps (int)`: Maximum number of interaction steps allowed for completing a single task before timing out.
- `back_times (int)`: Number of retry attempts allowed when the agent encounters an error or dead end.
- `sleep_seconds_per_act (int)`: Waiting time in seconds between consecutive actions.

**task**
- `task_file (str)`: Path to the file containing the list of tasks to be executed for trajectory collection. 
- `output (str)`: Specific output path for saving individual task results and trajectories.


## Evaluation

```commandline
python3 run.py --mode evaluate --config config/evaluate.conf
```

### Config file settings

**evaluation**
- `type (str)`: xpath
- `trajectory (str)`: Path to GUI trajectories to be evaluated. (e.g., "results/round1").
- `rule (str)`: Path to the rule file. (e.g., "data/top12.csv").


## License

The dataset of this project is licensed under the [**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The source code of the this is licensed under the [**Apache 2.0**](http://www.apache.org/licenses/LICENSE-2.0)  license.

### Summary of Terms
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial**: You may not use the material for commercial purposes.
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.


### License Badge
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Citation

If you find the resources in this repository helpful, please cite as:
```
@article{wu2026mobilebench,
  title={MobileBench-OL: A Comprehensive Chinese Benchmark for Evaluating Mobile GUI Agents in Real-World Environment},
  author={Wu, Qinzhuo and Yang, Zhizhuo and Li, Hanhao and Gao, Pengzhi and Liu, Wei and Luan, Jian},
  journal={arXiv preprint arXiv:2601.20335},
  year={2026}
}
```
