"""
MobileAgent v2 Agent
形式来源：参考 agent.py 的 base_agent 类结构和 step 函数
逻辑来源：参考 run_agent.py 中的感知信息获取和动作执行
"""
import sys
import os
import time
import copy
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# 添加 MobileAgent 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "MobileAgent" / "Mobile-Agent-v2"))

from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.controller import get_screenshot_u2, get_xml_u2, tap
from MobileAgent.api import inference_chat
from MobileAgent.chat import add_response

from mobilebench.utils import adb_executor
from mobilebench.models.llm_core_mobileagent_v2 import MobileAgentV2Wrapper

# 导入模型加载相关（逻辑来源：run_agent.py）
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download
import torch
import concurrent.futures
import base64
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt


# 全局模型加载标志（逻辑来源：run_agent.py）
_MODELS_LOADED = False
_groundingdino_model = None
_ocr_detection = None
_ocr_recognition = None


def load_models_once():
    """
    只加载一次模型，避免重复加载
    逻辑来源：参考 run_agent.py 中的 load_models_once
    """
    global _MODELS_LOADED, _groundingdino_model, _ocr_detection, _ocr_recognition
    
    if _MODELS_LOADED:
        return
    
    print("Loading models (first time only)...")
    
    # Load OCR and icon detection models
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
    _groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
    _ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    _ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
    
    _MODELS_LOADED = True
    print("Models loaded successfully!")


def merge_text_blocks(text_list, coordinates_list):
    """
    合并文本块
    逻辑来源：直接从 run_agent.py 复制
    """
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(range(len(coordinates_list)), key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue
        
        anchor = i
        
        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i + 1, num_blocks):
            if merge[j]:
                continue

            if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
            sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
            abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = "\n".join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

    return merged_text_blocks, merged_coordinates


def crop_icon(image, box, i, temp_file):
    """
    裁剪图标
    逻辑来源：参考 run_agent.py 中的 crop 函数
    """
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2 - 10 or y1 >= y2 - 10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(os.path.join(temp_file, f"{i}.jpg"))


def encode_image(image_path):
    """
    编码图片为 base64
    逻辑来源：直接从 run_agent.py 复制
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def process_image(image, query, qwen_api, caption_model):
    """
    使用 Qwen VL API 处理图片并生成描述
    逻辑来源：直接从 run_agent.py 复制
    """
    client = OpenAI(
        api_key=qwen_api,
        base_url="http://model.mify.ai.srv/v1",
        default_headers={
            "X-Model-Provider-Id": "tongyi"
        }
    )
    
    base64_image = encode_image(image)
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}
            },
            {
                'type': 'text',
                'text': query
            },
        ]
    }]
    
    response = client.chat.completions.create(
        model=caption_model,
        messages=messages,
        stream=False,
        timeout=10
    )
    return response.choices[0].message.content


def generate_api(images, query, qwen_api, caption_model):
    """
    使用多线程并发调用 Qwen VL API 处理图标
    逻辑来源：直接从 run_agent.py 复制
    """
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query, qwen_api, caption_model): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                response = future.result()
                icon_map[i + 1] = response
            except Exception as e:
                print(f"[generate_api] Icon {i + 1} processing failed after all retries: {e}. Using default.")
                icon_map[i + 1] = "This is an icon."

    return icon_map


class base_agent():
    """
    MobileAgent v2 Agent 类
    
    形式来源：参考 agent.py 中的 base_agent 类结构
    逻辑来源：参考 run_agent.py 中的感知信息获取和动作执行流程
    """

    def __init__(
        self,
        env,
        llm: MobileAgentV2Wrapper,
        qwen_api: str = "",
        caption_model: str = "qwen-vl-max-latest"
    ):
        """
        初始化 Agent
        
        形式来源：参考 agent.py 的 __init__
        逻辑来源：参考 run_agent.py 中的初始化逻辑
        """
        self.llm = llm
        self.env = env
        self.qwen_api = qwen_api
        self.caption_model = caption_model

        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        self.history_action = []
        self.summary = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = 3
        
        # MobileAgent 特有的状态变量（逻辑来源：run_agent.py）
        self.temp_file = f"temp_{int(time.time() * 1000)}"
        self.screenshot_dir = f"screenshot_{int(time.time() * 1000)}"
        self.keyboard_height_ratio = 0.8
        self.last_perception_infos = []
        self.last_keyboard = False
        
        # 创建临时目录
        if not os.path.exists(self.temp_file):
            os.mkdir(self.temp_file)
        if not os.path.exists(self.screenshot_dir):
            os.mkdir(self.screenshot_dir)
        
        # 加载模型（只加载一次）
        load_models_once()

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        """
        形式来源：参考 agent.py
        """
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        """
        形式来源：参考 agent.py
        """
        pass

    def clear(self):
        """
        清空历史记录
        形式来源：参考 agent.py
        """
        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        self.history_action = []
        self.summary = []
        
        # 清空临时文件
        if os.path.exists(self.temp_file):
            shutil.rmtree(self.temp_file)
            os.mkdir(self.temp_file)
    
    def save_home_page(self, path="screenshot/"):
        """
        保存初始页面
        形式来源：参考 agent.py
        逻辑来源：参考 run_agent.py
        """
        step_prefix = os.path.join(path, "step_0")

        # 保存截图
        img_path = step_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        # 保存XML
        xml_path = step_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)
    
    def get_perception_infos(self, screenshot_file: str, step_num: Optional[int] = None, save_dir: Optional[str] = None):
        """
        获取屏幕感知信息
        
        形式来源：新增方法，但参考 agent.py 的风格
        逻辑来源：参考 run_agent.py 中的 get_perception_infos 函数
        
        Args:
            screenshot_file: 截图文件路径
            step_num: 步骤编号（用于保存文件）
            save_dir: 保存目录
        
        Returns:
            (perception_infos, width, height, keyboard)
        """
        # 获取截图
        get_screenshot_u2(self.env, f"{self.screenshot_dir}/screenshot.jpg")
        
        # 保存 screenshot 和 xml（如果指定了步骤编号和保存目录）
        if step_num is not None and save_dir is not None:
            screenshot_path = os.path.join(save_dir, f"step_{step_num}.jpg")
            xml_path = os.path.join(save_dir, f"step_{step_num}.xml")
            
            if not os.path.exists(screenshot_path):
                shutil.copy(f"{self.screenshot_dir}/screenshot.jpg", screenshot_path)
            else:
                # 文件已存在（由噪声逻辑创建），使用已存在的文件作为 screenshot_file
                shutil.copy(screenshot_path, screenshot_file)
            
            get_xml_u2(self.env, xml_path)
        
        # 获取图片尺寸
        width, height = Image.open(screenshot_file).size
        
        # OCR 文本检测（逻辑来源：run_agent.py）
        try:
            text, coordinates = ocr(screenshot_file, _ocr_detection, _ocr_recognition)
        except Exception:
            text, coordinates = [], []

        text, coordinates = merge_text_blocks(text, coordinates)
        
        # 构建感知信息列表
        perception_infos = []
        for i in range(len(coordinates)):
            perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
            perception_infos.append(perception_info)
        
        # 图标检测（逻辑来源：run_agent.py）
        icon_coordinates = det(screenshot_file, "icon", _groundingdino_model)
        
        for i in range(len(icon_coordinates)):
            perception_info = {"text": "icon", "coordinates": icon_coordinates[i]}
            perception_infos.append(perception_info)
        
        # 裁剪图标
        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            if perception_infos[i]['text'] == 'icon':
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)
        
        for i in range(len(image_box)):
            crop_icon(screenshot_file, image_box[i], image_id[i], self.temp_file)
        
        # 获取图标描述（逻辑来源：run_agent.py）
        images = os.listdir(self.temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
            
            for i in range(len(images)):
                images[i] = os.path.join(self.temp_file, images[i])
            
            icon_map = generate_api(images, prompt, self.qwen_api, self.caption_model)

            for i, j in zip(image_id, range(1, len(image_id) + 1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] = "icon: " + icon_map[j]

        # 转换坐标为中心点
        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [
                int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
                int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)
            ]
        
        # 检测键盘（逻辑来源：run_agent.py）
        keyboard = False
        keyboard_height_limit = self.keyboard_height_ratio * height
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB' in perception_info['text'] or 'Clear' in perception_info['text'] or 'Text' in perception_info['text']:
                keyboard = True
                break
        
        # 清空临时文件
        if os.path.exists(self.temp_file):
            shutil.rmtree(self.temp_file)
            os.mkdir(self.temp_file)
        
        return perception_infos, width, height, keyboard

    def step(self, goal: str, path="screenshot/"):
        """
        执行一步
        
        形式来源：参考 agent.py 的 step 函数
        逻辑来源：参考 run_agent.py 中的主循环逻辑
        
        Returns:
            (is_finished, step_data)
        """
        step_data = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }

        step_index = "step_" + str(len(self.history_image_path) + 1)
        step_prefix = os.path.join(path, step_index)

        # 截图文件路径
        img_path = step_prefix + ".jpg"
        xml_path = step_prefix + ".xml"
        screenshot_file = f"{self.screenshot_dir}/screenshot.jpg"
        
        # 获取感知信息（逻辑来源：run_agent.py）
        step_num = len(self.history_image_path) + 1
        perception_infos, width, height, keyboard = self.get_perception_infos(
            screenshot_file,
            step_num,
            path
        )

        # 读取 XML
        if os.path.exists(xml_path):
            with open(xml_path, encoding='utf-8') as f:
                xml_string = f.read()
        else:
            xml_string = self.env.dump_hierarchy()
            with open(xml_path, 'w', encoding="utf-8") as f:
                f.write(xml_string)
        
        # 调用 LLM 进行预测（逻辑来源：run_agent.py）
        history = copy.deepcopy(step_data)
        response, action_output = self.llm.predict_mm(
            goal, img_path, perception_infos, width, height, keyboard, history
        )

        self.history_xml_string.append(xml_string)
        self.history_image_path.append(img_path)
        self.history_response.append(response)
        self.history_action.append(action_output)

        if action_output['action'] == 'terminate':
            self.summary.append('Agent thinks the request has been completed.')
            step_data = {
                'history_xml_string': self.history_xml_string,
                "history_image_path": self.history_image_path,
                "history_response": self.history_response,
                "history_action": self.history_action,
                "summary": self.summary,
            }
            return (True, step_data)

        print("##########model_response#################\n")
        print(response + "\n")
        print("##########execute_action#################\n")
        print(action_output["action"])
        print(action_output["params"])
        print(action_output["normalized_params"])

        # 处理 Open app 动作（逻辑来源：run_agent.py）
        if action_output['action'] == 'open_app':
            app_name = action_output['params'].get('app_name', '')
            text, coordinates = ocr(screenshot_file, _ocr_detection, _ocr_recognition)
            
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [
                        int((coordinates[ti][0] + coordinates[ti][2]) / 2),
                        int((coordinates[ti][1] + coordinates[ti][3]) / 2)
                    ]
                    
                    # 修改 action_output 为 click 动作
                    action_output['action'] = 'click'
                    action_output['params'] = {
                        'position': [
                            name_coordinate[0],
                            name_coordinate[1] - int(coordinates[ti][3] - coordinates[ti][1])
                        ]
                    }
                    action_output['normalized_params'] = action_output['params']
                    break
        
        # 执行动作
        try:
            adb_executor.execute_adb_action(action_output, self.env)
            time.sleep(2.0)
        except Exception as e:
            print('Failed to execute action.')
            print(str(e))
            print("action_output", action_output)
            summary_error = 'Can not execute the action, make sure to select the action with the required '
            summary_error += 'parameters (if any) in the correct JSON format!'
            self.summary.append(summary_error)
            step_data = {
                'history_xml_string': self.history_xml_string,
                "history_image_path": self.history_image_path,
                "history_response": self.history_response,
                "history_action": self.history_action,
                "summary": self.summary,
            }
            return (False, step_data)

        time.sleep(self.wait_after_action_seconds)

        step_data = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }
        return (False, step_data)
