from __future__ import annotations
import csv
import json
import os
import time
import uiautomator2 as u2

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from mobilebench.utils import agent, agent_noise
from mobilebench.models import llm_core_uitars_1_5
from mobilebench.models import llm_core_uitars
from mobilebench.models import llm_core_gpt4o
from mobilebench.models import llm_core_qwen2_5vl

from mobilebench.utils import adb_executor
from mobilebench.eval import evaluator_xpath as ev


@dataclass
class Task:
    identifier: str
    goal: str
    home_activity: str
    golden_steps: int
    key_nodes: str
    reset_xpath: str
    reset_query: str


@dataclass
class Trajectory:
    task_id: str
    task_goal: str
    history_action: list
    history_image_path: list
    history_response: list
    summary: str
    success: bool


# ---------- 设备管理 ----------

class DeviceManager:
    def __init__(self, serial: str, max_retry: int = 5):
        self.serial = serial
        self.max_retry = max_retry
        self.d = self._connect()

    def is_uiautomator_alive(self, d) -> bool:
        try:
            d.info
            return True
        except Exception as e:
            print(f"[DeviceManager] uiautomator2 may be unavailable: {e}")
            return False

    def _connect(self):
        for attempt in range(self.max_retry):
            try:
                print(f"[DeviceManager] Attempting to connect {self.serial}, the {attempt + 1} times")
                d = u2.connect(self.serial)
                d.set_input_ime(True)
                if self.is_uiautomator_alive(d):
                    print("[DeviceManager] Connection Success")
                return d
            except Exception as e:
                print(f"[DeviceManager] Connection failed: {e}")
                time.sleep(2)
        raise RuntimeError(f"Unable to connect to the device: {self.serial}")

    def reconnect(self):
        print("[DeviceManager] Attempting to reconnect the device...")
        self.d = self._connect()

    # ---------- 高层 API ----------
    def reset(self):
        """回桌面 + 清最近任务栏（可选）"""
        self.d.press("home")

    def clear_background(self, excludes: Optional[List[str]] = None):
        """
        彻底杀掉后台应用。
        Args:
            excludes: 不想被杀掉的包名列表，如 ['com.android.systemui']
        """
        self.d.app_stop_all(excludes or [])
        self.d.press("home")  # 杀完再回桌面，保证 UI 稳定

    def launch_app(self, activity: str):
        adb_executor.launch_app(activity, self.d)

    def stop_app(self, package: str):
        self.d.app_stop(package)

# ---------- Agent 工厂 ----------


class AgentFactory:
    @staticmethod
    def create(model_name: str, url: str, device, azure_config: Optional[Dict] = None):
        if model_name.startswith("uitars_1_5_noise_repeat"):
            return agent_noise.base_agent(device, llm_core_uitars_1_5.uitars1_5_Wrapper(url), "repeat")
        elif model_name.startswith("uitars_1_5_noise_unexecuted"):
            return agent_noise.base_agent(device, llm_core_uitars_1_5.uitars1_5_Wrapper(url), "unexecuted")
        elif model_name.startswith("uitars_1_5_noise_delay"):
            return agent_noise.base_agent(device, llm_core_uitars_1_5.uitars1_5_Wrapper(url), "delay")
        elif model_name.startswith("uitars_1_5_noise_popup"):
            return agent_noise.base_agent(device, llm_core_uitars_1_5.uitars1_5_Wrapper(url), "popup")
        elif model_name.startswith("uitars_1_5"):
            return agent.base_agent(device, llm_core_uitars_1_5.uitars1_5_Wrapper(url))
        elif model_name.startswith("uitars_noise_repeat"):
            return agent_noise.base_agent(device, llm_core_uitars.uitars_Wrapper(url), "repeat")
        elif model_name.startswith("uitars_noise_unexecuted"):
            return agent_noise.base_agent(device, llm_core_uitars.uitars_Wrapper(url), "unexecuted")
        elif model_name.startswith("uitars_noise_delay"):
            return agent_noise.base_agent(device, llm_core_uitars.uitars_Wrapper(url), "delay")
        elif model_name.startswith("uitars_noise_popup"):
            return agent_noise.base_agent(device, llm_core_uitars.uitars_Wrapper(url), "popup")
        elif model_name.startswith("uitars"):
            return agent.base_agent(device, llm_core_uitars.uitars_Wrapper(url))
        elif model_name.startswith("qwen2_5vl_noise_repeat"):
            return agent_noise.base_agent(device, llm_core_qwen2_5vl.qwen2_5vl_Wrapper(url), "repeat")
        elif model_name.startswith("qwen2_5vl_noise_unexecuted"):
            return agent_noise.base_agent(device, llm_core_qwen2_5vl.qwen2_5vl_Wrapper(url), "unexecuted")
        elif model_name.startswith("qwen2_5vl_noise_delay"):
            return agent_noise.base_agent(device, llm_core_qwen2_5vl.qwen2_5vl_Wrapper(url), "delay")
        elif model_name.startswith("qwen2_5vl_noise_popup"):
            return agent_noise.base_agent(device, llm_core_qwen2_5vl.qwen2_5vl_Wrapper(url), "popup")
        elif model_name.startswith("qwen2_5vl"):
            return agent.base_agent(device, llm_core_qwen2_5vl.qwen2_5vl_Wrapper(url))
        elif model_name.startswith("m3a_noise") or model_name.startswith("t3a_noise"):
            # M3A/T3A Noise模式 - 统一agent with noise
            # 解析噪声类型：model_name格式为 "m3a_noise_delay" 或 "t3a_noise_unexecuted"
            parts = model_name.split("_")
            mode = parts[0]  # "m3a" 或 "t3a"
            noise_type = parts[2] if len(parts) > 2 else "delay"  # 默认delay

            use_image = (mode == "m3a")  # M3A用图片，T3A纯文本

            from mobilebench.models import llm_core_m3a
            from mobilebench.utils import agent_m3a_t3a_noise

            if azure_config is None:
                raise ValueError(f"{mode.upper()} Noise requires azure_config parameter.")

            llm = llm_core_m3a.m3a_Wrapper(  # M3A和T3A都用gpt-4o
                use_azure=True,
                azure_endpoint=azure_config['azure_endpoint'],
                api_key=azure_config['api_key'],
                model=azure_config['model'],
                api_version=azure_config['api_version'],
                max_retry=azure_config['max_retry'],
                temperature=azure_config['temperature'],
                max_length=azure_config['max_length'],
            )

            return agent_m3a_t3a_noise.m3a_t3a_agent_noise(
                device, llm,
                use_image=use_image,
                noise_type=noise_type,
                noise_ratio=0.2,
                debug=False
            )
        elif model_name.startswith("m3a") or model_name.startswith("t3a"):
            # M3A/T3A 正常模式 - 统一agent
            use_image = model_name.startswith("m3a")  # M3A用图片，T3A纯文本

            from mobilebench.models import llm_core_m3a
            from mobilebench.utils import agent_m3a_t3a

            if azure_config is None:
                mode_name = "M3A" if use_image else "T3A"
                raise ValueError(f"{mode_name} requires azure_config parameter. Please add [azure] section in config file.")

            llm = llm_core_m3a.m3a_Wrapper(  # M3A和T3A都用gpt-4o
                use_azure=True,
                azure_endpoint=azure_config['azure_endpoint'],
                api_key=azure_config['api_key'],
                model=azure_config['model'],
                api_version=azure_config['api_version'],
                max_retry=azure_config['max_retry'],
                temperature=azure_config['temperature'],
                max_length=azure_config['max_length'],
            )

            return agent_m3a_t3a.m3a_t3a_agent(device, llm, use_image=use_image, debug=False)
        elif model_name.startswith("gpt4o"):
            return agent.base_agent(device, llm_core_gpt4o.GPT4oWrapper())
        
        elif model_name.startswith("mobileagentv2_noise"):
            from mobilebench.models import llm_core_mobileagent_v2
            from mobilebench.utils import agent_mobileagent_v2_noise
            parts = url.split("|")
            api_url = parts[0] if len(parts) > 0 else url
            token = parts[1] if len(parts) > 1 else ""
            qwen_api = parts[2] if len(parts) > 2 else ""
            caption_model = parts[3] if len(parts) > 3 else "qwen-vl-max-latest"
            llm = llm_core_mobileagent_v2.MobileAgentV2Wrapper(api_url, token)
            noise_type = ""
            for candidate in ("repeat", "unexecuted", "delay", "popup"):
                if model_name.startswith(f"mobileagentv2_noise_{candidate}"):
                    noise_type = candidate
                    break
            return agent_mobileagent_v2_noise.base_agent(device, llm, qwen_api, caption_model, noise_type)
        elif model_name.startswith("mobileagentv2"):
            from mobilebench.models import llm_core_mobileagent_v2
            from mobilebench.utils import agent_mobileagent_v2
            parts = url.split("|")
            api_url = parts[0] if len(parts) > 0 else url
            token = parts[1] if len(parts) > 1 else ""
            qwen_api = parts[2] if len(parts) > 2 else ""
            caption_model = parts[3] if len(parts) > 3 else "qwen-vl-max-latest"
            llm = llm_core_mobileagent_v2.MobileAgentV2Wrapper(api_url, token)
            return agent_mobileagent_v2.base_agent(device, llm, qwen_api, caption_model)

        raise ValueError(f"Unknown model {model_name}")

# ---------- Task 执行 ----------


class TaskExecutor:
    def __init__(self, device_mgr: DeviceManager, agent, time_sleep=3, maximum_steps=20):
        self.device_mgr = device_mgr
        self.agent = agent
        self.time_sleep = time_sleep
        self.maximum_steps = maximum_steps

    def run(self, task: Task, save_dir: Path, reset: bool = False) -> Trajectory:
        self.agent.clear()

        print("task.goal", task.goal)
        self.device_mgr.clear_background()
        time.sleep(2 * self.time_sleep)
        os.makedirs(str(save_dir), exist_ok=True)
        self.agent.save_home_page(path=str(save_dir))

        self.device_mgr.launch_app(task.home_activity)
        time.sleep(4 * self.time_sleep)

        max_steps = min(task.golden_steps * 3, self.maximum_steps)
        os.makedirs(save_dir, exist_ok=True)
        if reset:
            for _ in range(max_steps):
                ok, stepdata = self.agent.step(task.reset_query, path=str(save_dir))
                if ok:
                    break
            success = evaluator_xpath.evaluate(task.reset_xpath, stepdata)
        else:
            for _ in range(max_steps):
                ok, stepdata = self.agent.step(task.goal, path=str(save_dir))
                if ok:
                    break
            time.sleep(self.time_sleep)
            success = evaluator_xpath.evaluate(task.key_nodes, stepdata)

        traj = Trajectory(
            task_id=task.identifier,
            task_goal=task.goal if not reset else task.reset_query,
            history_action=stepdata["history_action"],
            history_image_path=stepdata["history_image_path"],
            history_response=stepdata["history_response"],
            summary=stepdata["summary"],
            success=success,
        )
        return traj


class evaluator_xpath:
    @staticmethod
    def evaluate(task_rule: str, stepdata: dict) -> bool:
        return ev.evaluate(task_rule, stepdata)


class ResultSink:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, bool] = self._load_cache()

    def _load_cache(self):
        fp = os.path.join(self.base_dir, "result_list.txt")
        if not os.path.exists(fp):
            return {}
        fp = open(fp, encoding="utf-8")
        content = fp.read()  # 读取全部内容
        data = {}
        for line in content.strip().split(" "):  # 按空格分割
            if not line:
                continue
            key, value = line.split(",")
            data[key] = value.lower() == "true"
        return data

    def save(self, traj: Trajectory):
        task_dir = os.path.join(self.base_dir, traj.task_id)
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "trajectory.json"), "w", encoding="utf-8") as f:
            json.dump(traj.__dict__, f, ensure_ascii=False, indent=2)
        self.cache[traj.task_id] = traj.success
        self._flush_cache()

    def _flush_cache(self):
        fp = open(os.path.join(self.base_dir, "result_list.txt"), "w", encoding="utf-8")
        fp.write(" ".join(f"{k},{v}" for k, v in self.cache.items()))
    # 事后评估整轮通过率

    def summary(self):
        if not self.cache:
            return 0.0
        passed = sum(self.cache.values())
        return passed * 100 / len(self.cache)

# ---------- CSV Loader ----------


def load_tasks(csv_path: Path) -> List[Task]:
    tasks: List[Task] = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["key_nodes"]:
                continue
            tasks.append(
                Task(
                    identifier=row["task_identifier"],
                    goal=row["goal"],
                    home_activity=row["adb_home_page"],
                    golden_steps=int(row["golden_steps"]),
                    key_nodes=row["key_nodes"],
                    reset_xpath=row.get("reset_xpath", "") or "",
                    reset_query=row.get("reset_query", "") or "",
                )
            )
    return tasks


def load_tasks_without_apps(csv_path, app_list):
    tasks: List[Task] = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["key_nodes"]:
                continue
            if row['task_app'] in app_list:
                continue
            tasks.append(
                Task(
                    identifier=row["task_identifier"],
                    goal=row["goal"],
                    home_activity=row["adb_home_page"],
                    golden_steps=int(row["golden_steps"]),
                    key_nodes=row["key_nodes"],
                    reset_xpath=row.get("reset_xpath", "") or "",
                    reset_query=row.get("reset_query", "") or "",
                )
            )
    return tasks


def run_with_reconnect(executor, task, task_dir, reset, dev_mgr, connect_retry=2):
    """
    带连接失败自动重连的任务执行包装器。
    如果抛异常，会尝试最多 connect_retry 次重连后继续执行。
    """
    for attempt in range(connect_retry):
        return executor.run(task, task_dir, reset)
        try:
            return executor.run(task, task_dir, reset)
        except Exception as e:
            print(f"[ERROR] Connection Failed（ The {attempt + 1}/{connect_retry} times）: {e}")
            if attempt < connect_retry - 1:
                print("[INFO] Tyr reconnect...")
                dev_mgr.reconnect()
                time.sleep(2)
            else:
                raise  # 最后一次仍失败就抛出


def try_execute_task_with_retry(task: Task, BASE_DIR: str, executor: TaskExecutor, dev_mgr: DeviceManager,
                                connent_retry: int, fail_retry: int, reset: bool) -> Optional['Trajectory']:
    """
    包装任务执行逻辑，支持 FAIL_RETRY 次失败重试（非连接错误），
    并保证无论成功失败，最後一次 traj 都能返回。
    """
    task_dir = os.path.join(BASE_DIR, task.identifier)

    for fail_attempt in range(fail_retry):
        # 每次重新执行任务前清空旧数据
        if os.path.exists(task_dir):
            for f in os.listdir(task_dir):
                file_path = os.path.join(task_dir, f)  # Create full path
                try:
                    if os.path.isfile(file_path):  # Check if it's a file (not a directory)
                        os.remove(file_path)
                    # If you want to handle directories too, you could add:
                    # elif os.path.isdir(file_path):
                    #     shutil.rmtree(file_path)
                except Exception as e:
                    print(f"[WARN] Delete files failed: {file_path}, {e}")
        traj = run_with_reconnect(executor, task, task_dir, reset, dev_mgr, connect_retry=connent_retry)
        # try:
        #     traj = run_with_reconnect(executor, task, task_dir, reset, dev_mgr, connect_retry=connent_retry)
        # except Exception as e:
        #     print(f"[FAIL] Connection failed, task skipped.: {e}")
        #     return None

        if not traj.success:
            print(f"[WARN] Execution failed, checking if any retry attempts left {fail_attempt + 1}/{fail_retry}")
            if fail_attempt < fail_retry - 1:
                print("[INFO] Retry task ...")
                dev_mgr.reconnect()
                time.sleep(2)
                continue
            else:
                print(f"[FAIL] Multiple failures, still not success：{task.identifier}")
                return traj  # 返回失败 traj，供分析或保存
        else:
            return traj

    return None
