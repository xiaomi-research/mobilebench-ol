"""
MobileAgent v2 LLM Core
形式来源：参考 llm_core_uitars_1_5.py 的消息处理和响应解析架构
逻辑来源：参考 run_agent.py 中的 MobileAgent prompt 和 inference_chat 调用
"""
import sys
import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加 MobileAgent 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "MobileAgent_new" / "Mobile-Agent-v2"))

from MobileAgent.api import inference_chat
from MobileAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from MobileAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image


class mobileagent_v2_message_handler(object):
    """
    形式来源：参考 uitars_1_5_message_handler 的消息处理结构
    逻辑来源：参考 run_agent.py 中的 prompt 构建和历史管理
    """
    
    def __init__(self):
        self.add_info = "The required app is already open. Please do not use the \"Open app\" or \"Home\" actions."
        self.keyboard_height_ratio = 0.8
        
    def process_message(
        self,
        task: str,
        image_path: str,
        perception_infos: List[Dict],
        width: int,
        height: int,
        keyboard: bool,
        history: Optional[Dict[str, List[str]]] = None,
        last_image_path: Optional[str] = None,
        reflection_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        处理消息，构建 MobileAgent 的 prompt
        
        形式来源：参考 uitars_1_5_message_handler.process_message 的参数和返回结构
        逻辑来源：参考 run_agent.py 中的 prompt 构建逻辑
        
        Args:
            task: 任务描述
            image_path: 当前截图路径
            perception_infos: 感知信息列表
            width: 屏幕宽度
            height: 屏幕高度
            keyboard: 是否显示键盘
            history: 历史记录字典
            last_image_path: 上一张截图路径（用于反思）
            reflection_data: 反思所需的额外数据
        
        Returns:
            包含 chat 对象和其他必要信息的字典
        """
        # 从历史记录中提取信息
        summary_history = []
        action_history = []
        thought_history = []
        completed_requirements = ""
        memory = ""
        error_flag = False
        
        if history:
            # 从 history_response 中提取之前的 thought, summary, action
            response_list = history.get("history_response", [])
            for response in response_list[:-1]:  # 排除最后一个（当前轮）
                try:
                    thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").strip()
                    summary = response.split("### Operation ###")[-1].replace("\n", " ").strip()
                    action = response.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").strip()
                    
                    thought_history.append(thought)
                    summary_history.append(summary)
                    action_history.append(action)
                except Exception:
                    pass

        # 提取当前的 summary 和 action（如果有）
        summary = ""
        action = ""
        if reflection_data:
            summary = reflection_data.get("summary", "")
            action = reflection_data.get("action", "")
            error_flag = reflection_data.get("error_flag", False)
            completed_requirements = reflection_data.get("completed_requirements", "")
            memory = reflection_data.get("memory", "")
        
        # 构建 action prompt（逻辑来源：run_agent.py）
        prompt_action = get_action_prompt(
            task,
            perception_infos,
            width,
            height,
            keyboard,
            summary_history,
            action_history,
            summary,
            action,
            self.add_info,
            error_flag,
            completed_requirements,
            memory
        )
        
        # 初始化 chat（逻辑来源：run_agent.py）
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, image_path)
        
        return {
            "chat": chat_action,
            "type": "action",
            "width": width,
            "height": height,
            "thought_history": thought_history,
            "summary_history": summary_history,
            "action_history": action_history
        }
    
    def process_reflection_message(
        self,
        task: str,
        last_perception_infos: List[Dict],
        perception_infos: List[Dict],
        width: int,
        height: int,
        last_keyboard: bool,
        keyboard: bool,
        summary: str,
        action: str,
        last_image_path: str,
        current_image_path: str,
    ) -> Dict[str, Any]:
        """
        处理反思消息
        
        形式来源：参考 process_message 的结构
        逻辑来源：参考 run_agent.py 中的反思逻辑
        """
        prompt_reflect = get_reflect_prompt(
            task,
            last_perception_infos,
            perception_infos,
            width,
            height,
            last_keyboard,
            keyboard,
            summary,
            action,
            self.add_info
        )
        
        chat_reflect = init_reflect_chat()
        chat_reflect = add_response_two_image("user", prompt_reflect, chat_reflect, [last_image_path, current_image_path])
        
        return {
            "chat": chat_reflect,
            "type": "reflect"
        }
    
    def process_memory_message(self, insight: str, chat_action) -> Dict[str, Any]:
        """
        处理记忆消息
        
        形式来源：参考 process_message 的结构
        逻辑来源：参考 run_agent.py 中的记忆逻辑
        """
        prompt_memory = get_memory_prompt(insight)
        chat_action = add_response("user", prompt_memory, chat_action)
        
        return {
            "chat": chat_action,
            "type": "memory"
        }
    
    def process_planning_message(
        self,
        task: str,
        thought_history: List[str],
        summary_history: List[str],
        action_history: List[str],
        completed_requirements: str,
    ) -> Dict[str, Any]:
        """
        处理规划消息
        
        形式来源：参考 process_message 的结构
        逻辑来源：参考 run_agent.py 中的规划逻辑
        """
        prompt_planning = get_process_prompt(
            task,
            thought_history,
            summary_history,
            action_history,
            completed_requirements,
            self.add_info
        )
        
        chat_planning = init_memory_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning)
        
        return {
            "chat": chat_planning,
            "type": "planning"
        }
    
    def process_response(self, content: str, response_type: str = "action") -> Dict[str, Any]:
        """
        处理 LLM 响应
        
        形式来源：参考 uitars_1_5_message_handler.process_response 的结构
        逻辑来源：参考 run_agent.py 中的响应解析逻辑
        
        Args:
            content: LLM 返回的文本
            response_type: 响应类型（action/reflect/memory/planning）
        
        Returns:
            解析后的动作字典
        """
        result = {
            "raw_response": content,
            "action": "invalid",
            "params": {},
            "normalized_params": {}
        }
        
        if response_type == "action":
            # 解析 action 响应（逻辑来源：run_agent.py）
            try:
                thought = content.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
                summary = content.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
                action = content.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
                
                result.update({
                    "thought": thought,
                    "summary": summary,
                    "action_text": action
                })
                
                # 解析具体的动作类型
                if "Open app" in action:
                    app_name = action.split("(")[-1].split(")")[0]
                    result.update({
                        "action": "open_app",
                        "params": {"app_name": app_name},
                        "normalized_params": {"app_name": app_name}
                    })
                
                elif "Tap" in action:
                    coordinate = action.split("(")[-1].split(")")[0].split(", ")
                    x, y = int(coordinate[0]), int(coordinate[1])
                    result.update({
                        "action": "click",
                        "params": {"position": [x, y]},
                        "normalized_params": {"position": [x, y]}
                    })
                
                elif "Swipe" in action:
                    coordinate1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
                    coordinate2 = action.split("), (")[-1].split(")")[0].split(", ")
                    x1, y1 = int(coordinate1[0]), int(coordinate1[1])
                    x2, y2 = int(coordinate2[0]), int(coordinate2[1])
                    result.update({
                        "action": "swipe",
                        "params": {"start": [x1, y1], "end": [x2, y2]},
                        "normalized_params": {"start": [x1, y1], "end": [x2, y2]}
                    })
                
                elif "Type" in action:
                    if "(text)" not in action:
                        text = action.split("(")[-1].split(")")[0]
                    else:
                        text = action.split(" \"")[-1].split("\"")[0]
                    result.update({
                        "action": "type",
                        "params": {"text": text},
                        "normalized_params": {"text": text}
                    })
                
                elif "Back" in action:
                    result.update({
                        "action": "back",
                        "params": {},
                        "normalized_params": {}
                    })
                
                elif "Home" in action:
                    result.update({
                        "action": "home",
                        "params": {},
                        "normalized_params": {}
                    })
                
                elif "Stop" in action:
                    result.update({
                        "action": "terminate",
                        "params": {"text": summary},
                        "normalized_params": {"text": summary}
                    })
                
                else:
                    result.update({
                        "action": "wait",
                        "params": {},
                        "normalized_params": {}
                    })
                    
            except Exception as e:
                print(f"[ERROR] Failed to parse action response: {e}")
                result.update({
                    "action": "invalid",
                    "params": {},
                    "normalized_params": {}
                })
        
        elif response_type == "reflect":
            # 解析反思响应（逻辑来源：run_agent.py）
            try:
                reflect = content.split("### Answer ###")[-1].replace("\n", " ").strip()
                result.update({
                    "reflect": reflect,
                    "reflect_choice": 'A' if 'A' in reflect else ('B' if 'B' in reflect else 'C')
                })
            except Exception:
                result.update({
                    "reflect": "A",
                    "reflect_choice": "A"
                })

        elif response_type == "memory":
            # 解析记忆响应（逻辑来源 : run_agent.py）
            try:
                memory_content = content.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
                result.update({
                    "memory_content": memory_content
                })
            except Exception:
                result.update({
                    "memory_content": ""
                })

        elif response_type == "planning":
            # 解析规划响应（逻辑来源 : run_agent.py）
            try:
                completed_requirements = content.split("### Completed contents ###")[-1].replace("\n", " ").strip()
                result.update({
                    "completed_requirements": completed_requirements
                })
            except Exception:
                result.update({
                    "completed_requirements": ""
                })

        return result


class MobileAgentV2Wrapper():
    """
    MobileAgent v2 包装器
    
    形式来源：参考 uitars1_5_Wrapper 的类结构
    逻辑来源：参考 run_agent.py 中的 API 调用和模型交互
    """
    
    def __init__(
        self,
        url: str,
        token: str,
        reflection_switch: bool = True,
        memory_switch: bool = True,
        max_retry: int = 2,
    ):
        """
        初始化 MobileAgent v2 包装器
        
        Args:
            url: API URL
            token: API token
            reflection_switch: 是否启用反思
            memory_switch: 是否启用记忆
            max_retry: 最大重试次数
        """
        self.url = url
        self.token = token
        self.reflection_switch = reflection_switch
        self.memory_switch = memory_switch
        self.max_retry = max_retry
        self.message_handler = mobileagent_v2_message_handler()
        
        # 状态变量（逻辑来源：run_agent.py）
        self.thought_history = []
        self.summary_history = []
        self.action_history = []
        self.completed_requirements = ""
        self.memory = ""
        self.insight = ""
        self.error_flag = False
        self.last_summary = ""
        self.last_action = ""
    
    def predict_mm(
        self,
        goal: str,
        current_image_path: str,
        perception_infos: List[Dict],
        width: int,
        height: int,
        keyboard: bool,
        history: Optional[Dict] = None
    ) -> tuple:
        """
        执行一次多模态预测
        
        形式来源：参考 uitars1_5_Wrapper.predict_mm 的接口
        逻辑来源：参考 run_agent.py 中的推理流程
        
        Returns:
            (response_text, action_output)
        """
        # 构建 action 消息
        reflection_data = {
            "summary": self.last_summary,
            "action": self.last_action,
            "error_flag": self.error_flag,
            "completed_requirements": self.completed_requirements,
            "memory": self.memory
        }
        
        msg_data = self.message_handler.process_message(
            goal,
            current_image_path,
            perception_infos,
            width,
            height,
            keyboard,
            history,
            reflection_data=reflection_data
        )
        
        # 调用 API（逻辑来源：run_agent.py）
        try:
            output_action = inference_chat(msg_data["chat"], 'gpt-4o', self.token, azure_endpoint="https://ui-agent-exp.openai.azure.com/")
        except Exception as e:
            print(f"[ERROR] GPT-4o API call failed: {e}")
            output_action = "Error: GPT-4o API call failed."
        
        # 解析响应
        action_output = self.message_handler.process_response(output_action, "action")
        
        # 更新状态
        self.last_summary = action_output.get("summary", "")
        self.last_action = action_output.get("action_text", "")
        
        return output_action, action_output
