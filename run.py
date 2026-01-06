from __future__ import annotations
import argparse

from configparser import ConfigParser
from pathlib import Path

from mobilebench.utils.task_executor import DeviceManager
from mobilebench.utils.task_executor import AgentFactory
from mobilebench.utils.task_executor import TaskExecutor
from mobilebench.utils.task_executor import ResultSink
from mobilebench.utils.task_executor import load_tasks
from mobilebench.utils.task_executor import try_execute_task_with_retry
from mobilebench.eval import evaluator_xpath as ev
from mobilebench.eval import evaluator_xpath_step_ratio as eval_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GUI agent benchmark.")
    parser.add_argument('--mode', type=str, default='interact', help='Trajectory generation or trajectory evaluation')
    parser.add_argument('--config', type=str, default='config/interact_uitars15_base.conf', help='config file')
    # parser.add_argument('--mode', type=str, default='evaluate', help='Trajectory generation or trajectory evaluation')
    # parser.add_argument('--config', type=str, default='config/evaluate.conf', help='config file')
    parser.add_argument('--subset', type=str, default='base', help='[base,long-tail,long-horizon,gui-reasoning,noise-robust]')
    parser.add_argument('--output', type=str, default='results/ui-tars-1.5_base_0001', help='')

    return parser.parse_args()


def get_task_file(subset):
    if subset == "base":
        task_file = "data/MobileBench-OL - top12.csv"
    if subset == "long-tail":
        task_file = "data/longtail.csv"
    if subset == "long-horizon":
        task_file = "data/MobileBench-OL - Long-Horizon.csv"
    if subset == "gui-reasoning":
        task_file = "data/MobileBench-OL - Explo_already.csv"
    if subset == "noise-robust":
        task_file = "data/MobileBench-OL - top12.csv"
    return task_file


def main():
    args = parse_args()
    mode = args.mode
    config = ConfigParser()
    config.read(args.config, encoding='UTF-8')
    print(config)
    secs = config.sections()
    print(secs)
    if mode == "interact":
        # 读取参数
        SERIAL = config.get('device', 'id')
        MODEL_NAME = config.get('model', 'name')
        MODEL_URL = config.get('model', 'url')
        RETRY_ROUNDS = int(config.get('retry', 'retry_rounds'))
        CONNECT_RETRY = int(config.get('retry', 'connect_retry'))
        FAIL_RETRY = int(config.get('retry', 'fail_retry'))
        reset = config.get('reset', 'reset') == 'true'
        # Traj_NAME = config.get('task', 'output')
        Traj_NAME = args.output
        BASE_DIR = Path(Traj_NAME)

        max_steps = int(config.get('operation', 'max_steps'))
        time_sleep = int(config.get('operation', 'sleep_seconds_per_act'))

        azure_config = None
        if MODEL_NAME.startswith("m3a") and config.has_section('azure'):
            azure_config = {
                'azure_endpoint': config.get('azure', 'azure_endpoint'),
                'api_key': config.get('azure', 'api_key'),
                'model': config.get('azure', 'model'),
                'api_version': config.get('azure', 'api_version'),
                'max_retry': int(config.get('azure', 'max_retry')),
                'temperature': float(config.get('azure', 'temperature')),
                'max_length': int(config.get('azure', 'max_length')),
            }

        if "noise" not in args.subset:
            task_file = get_task_file(args.subset)
            tasks = load_tasks(Path(task_file))

            # -------- Agent 初始化 --------
            dev_mgr = DeviceManager(SERIAL)
            agent = AgentFactory.create(MODEL_NAME, MODEL_URL, dev_mgr.d, azure_config=azure_config)
            executor = TaskExecutor(dev_mgr, agent, time_sleep, max_steps)
            sink = ResultSink(BASE_DIR)

            # -------- 多轮补跑逻辑 --------
            for round_id in range(RETRY_ROUNDS):
                print(f"\n[INFO] The {round_id + 1} task execution started...")
                any_new_success = False

                for task in tasks:
                    if task.identifier in sink.cache:
                        continue

                    traj = try_execute_task_with_retry(task, BASE_DIR, executor, dev_mgr, CONNECT_RETRY, FAIL_RETRY, reset)
                    if traj is not None:
                        sink.save(traj)
                        print(f"[TRAJ SUCCESS] {task.identifier}  ✅")
                        any_new_success = True

                if not any_new_success:
                    print("[INFO] No new tasks were successfully completed this round, ending the process early.")
                    break

            # -------- 总结与评估 --------
            print(f"\n✅ Overall pass rate: {sink.summary():.2f}%")
            ev.re_evaluate_all(Traj_NAME, task_file, reset)
        else:
            for noise_type in ['repeat', 'unexecuted', 'delay', 'popup']:
                # 新增。Agent和task file根据噪声调整，但保存目录不变
                MODEL_NAME = config.get('model', 'name')
                MODEL_NAME = MODEL_NAME + "_" + noise_type
                task_file = get_task_file(args.subset)
                task_file = task_file.split(".csv")[0] + " - " + noise_type + ".csv"
                tasks = load_tasks(Path(task_file))
                print(f"\n===== Executing with noise type: {noise_type} =====")

                # -------- Agent 初始化 --------
                dev_mgr = DeviceManager(SERIAL)
                agent = AgentFactory.create(MODEL_NAME, MODEL_URL, dev_mgr.d, azure_config=azure_config)
                executor = TaskExecutor(dev_mgr, agent, time_sleep, max_steps)
                sink = ResultSink(BASE_DIR)
                needed_to_run = []

                # -------- 多轮补跑逻辑 --------
                for round_id in range(RETRY_ROUNDS):
                    print(f"\n[INFO] The {round_id + 1} task execution started...")
                    any_new_success = False

                    for task in tasks:
                        if task.identifier in sink.cache:
                            continue

                        traj = try_execute_task_with_retry(task, BASE_DIR, executor, dev_mgr, CONNECT_RETRY, FAIL_RETRY, reset)
                        # 下面是存traj.json
                        if traj is not None:
                            sink.save(traj)
                            print(f"[TRAJ SUCCESS] {task.identifier}  ✅")
                            any_new_success = True

                    if not any_new_success:
                        print("[INFO] No new tasks were successfully completed this round, ending the process early.")
                        break

                # -------- 总结与评估 --------
                print(f"\n✅ Overall pass rate: {sink.summary():.2f}%")
                ev.re_evaluate_all(Traj_NAME, task_file, reset)

    elif mode == "evaluate":
        # rule = config.get('evaluation', 'rule')
        rule = get_task_file(args.subset)
        trajectory_path = config.get('evaluation', 'trajectory')
        reset_flag = config.get('evaluation', 'reset') == 'true'
        eval_trajectory.re_evaluate_all(trajectory_path, rule, reset_flag)


if __name__ == "__main__":
    main()
