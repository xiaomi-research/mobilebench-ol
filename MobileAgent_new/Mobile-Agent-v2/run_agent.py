"""
Mobile Agent Runner - 独立的agent执行模块
用于批量处理任务，避免run.py的全局变量污染问题
"""

import time
import uiautomator2 as u2


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 运行时间: {end_time - start_time:.6f} 秒")
        return result
    return wrapper


import os
import sys
import time
import copy
import json
import shutil
import base64
import torch
import random
import concurrent.futures
from PIL import Image, ImageDraw
from tenacity import retry, wait_random_exponential, stop_after_attempt

from MobileAgent.api import inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.controller import get_screenshot, get_screenshot_with_path, get_screenshot_u2, get_xml, get_xml_u2, tap, slide, type, back, home
from MobileAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from MobileAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from openai import OpenAI


# 全局模型加载标志
_MODELS_LOADED = False
_groundingdino_model = None
_ocr_detection = None
_ocr_recognition = None
_qwen_model = None
_qwen_tokenizer = None


def load_models_once(caption_call_method, caption_model):
    """只加载一次模型，避免重复加载"""
    global _MODELS_LOADED, _groundingdino_model, _ocr_detection, _ocr_recognition
    global _qwen_model, _qwen_tokenizer

    if _MODELS_LOADED:
        return

    print("Loading models (first time only)...")

    # Load caption model
    device = "cuda"
    torch.manual_seed(1234)
    if caption_call_method == "local":
        if caption_model == "qwen-vl-chat":
            model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
            _qwen_model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map=device, trust_remote_code=True).eval()
            _qwen_model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif caption_model == "qwen-vl-chat-int4":
            qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
            _qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_dir, device_map=device, trust_remote_code=True, use_safetensors=True).eval()
            _qwen_model.generation_config = GenerationConfig.from_pretrained(
                qwen_dir, trust_remote_code=True, do_sample=False)
        else:
            raise ValueError(
                "If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
    elif caption_call_method != "api":
        raise ValueError("You must choose the caption model call function from \"local\" and \"api\"")

    # Load OCR and icon detection models
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
    _groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
    _ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    _ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

    _MODELS_LOADED = True
    print("Models loaded successfully!")


# ==================== 噪声处理辅助函数 ====================

def _execute_action(adb_path, device_id, action_dict):
    """
    执行一个动作

    Args:
        adb_path: ADB路径
        device_id: 设备ID
        action_dict: 动作字典，包含type和相关参数
    """
    action_type = action_dict.get("type")

    if action_type == "Tap" or action_type == "Open app":
        x = action_dict.get("x")
        y = action_dict.get("y")
        tap(adb_path, device_id, x, y)
    elif action_type == "Swipe":
        x1 = action_dict.get("x1")
        y1 = action_dict.get("y1")
        x2 = action_dict.get("x2")
        y2 = action_dict.get("y2")
        slide(adb_path, device_id, x1, y1, x2, y2)
    elif action_type == "Type":
        text = action_dict.get("text")
        type(adb_path, device_id, text)
    elif action_type == "Back":
        back(adb_path, device_id)
    elif action_type == "Home":
        home(adb_path, device_id)

    elif action_type == "wait":
        time.sleep(2)  # 等待2秒


def copy_page_files(source_path, target_path):
    """
    复制页面文件（包括.jpg和.xml）
    使用 os.path.splitext 安全地处理文件扩展名
    """
    # 复制图片文件
    shutil.copy(source_path, target_path)

    # 复制对应的XML文件
    source_xml = os.path.splitext(source_path)[0] + ".xml"
    target_xml = os.path.splitext(target_path)[0] + ".xml"
    if os.path.exists(source_xml):
        shutil.copy(source_xml, target_xml)


def copy_noise_page(predefined_path, keyword, target_path):
    """
    从预定义路径随机选择一个噪声页面并复制

    Args:
        predefined_path: 噪声页面存储路径（如 "noise/fanqieread/"）
        keyword: 噪声类型关键字（如 "delay", "popup"）
        target_path: 目标文件路径（如 "results/task_0/step_1.jpg"）

    Returns:
        选中的噪声文件名，如果没有找到则返回 None
    """
    if not os.path.exists(predefined_path):
        print(f"[WARN] Noise path does not exist: {predefined_path}")
        return None

    # 查找符合条件的文件
    files = []
    for item in os.listdir(predefined_path):
        full_path = os.path.join(predefined_path, item)
        if os.path.isfile(full_path) and item.startswith(keyword) and (item.endswith(".png") or item.endswith(".jpg")):
            files.append(item)

    if not files:
        print(f"[WARN] No noise files found with keyword '{keyword}' in {predefined_path}")
        return None

    print(f"[INFO] Found {len(files)} noise files: {files}")

    # 随机选择一个文件
    random_file = random.choice(files)
    source_path = os.path.join(predefined_path, random_file)

    # 复制文件
    copy_page_files(source_path, target_path)

    print(f"[INFO] Copied noise page: {random_file} -> {target_path}")
    return random_file


def check_files_with_prefix(path, prefix):
    """
    检查指定路径下是否有以prefix开头的文件

    Returns:
        文件名（如果存在），否则返回 False
    """
    if not os.path.exists(path):
        return False

    for filename in os.listdir(path):
        if filename.startswith(prefix) and os.path.isfile(os.path.join(path, filename)):
            return filename
    return False


def check_close_popup(filename, action_type, action_params):
    """
    检查动作是否点击了弹窗关闭区域

    Args:
        filename: 弹窗文件名，包含box坐标信息（如 "step_1_popup_30_80_120_180.jpg"）
        action_type: 动作类型（如 "Tap"）
        action_params: 动作参数（如坐标）

    Returns:
        True 如果点击在弹窗区域内，否则 False
    """
    if action_type != "Tap":
        return False

    # 将 .xml 转换为 .jpg
    if filename.endswith(".xml"):
        filename = filename.replace(".xml", ".jpg")

    # 从文件名中提取 box 坐标
    # 格式: step_N_popup_x1_y1_x2_y2.jpg
    try:
        parts = filename.split("_")
        # 找到 "popup" 的位置
        popup_idx = parts.index("popup")
        x1 = int(parts[popup_idx + 1])
        y1 = int(parts[popup_idx + 2])
        x2 = int(parts[popup_idx + 3])
        y2 = int(parts[popup_idx + 4].split(".")[0])  # 去掉扩展名
    except (ValueError, IndexError):
        print(f"[WARN] Cannot parse popup box from filename: {filename}")
        return False

    # 检查点击坐标是否在弹窗区域内
    try:
        # action_params 可能是字符串 "(x, y)" 或已经解析好的列表
        if isinstance(action_params, str):
            # 从 "(x, y)" 提取坐标
            coords = action_params.strip("()").split(",")
            x = int(coords[0].strip())
            y = int(coords[1].strip())
        else:
            # 假设是列表或元组
            x, y = action_params[0], action_params[1]
    except Exception as e:
        print(f"[WARN] Cannot parse tap coordinates: {action_params}, error: {e}")
        return False

    # 判断是否在区域内
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        print(f"[INFO] Tap ({x}, {y}) is inside popup box ({x1}, {y1}, {x2}, {y2})")
        return True

    return False


# ==================== 原有函数 ====================


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse(
    (coord[0] -
    point_size,
    coord[1] -
    point_size,
    coord[0] +
    point_size,
    coord[1] +
    point_size),
     fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2 - 10 or y1 >= y2 - 10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./temp/{i}.jpg")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def process_image(image, query, qwen_api, caption_model):
    """
    使用 Qwen VL API 处理图片并生成描述

    使用 @retry 装饰器：
    - 最多重试 3 次
    - 指数退避，等待时间在 1-5 秒之间随机
    - 所有重试失败后会抛出异常，由调用者捕获
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


@timer_decorator
def generate_api(images, query, qwen_api, caption_model):
    """
    使用多线程并发调用 Qwen VL API 处理图标

    如果某个图标处理失败（所有重试都失败），则使用默认值 "This is an icon."
    """
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
    executor.submit(
        process_image,
        image,
        query,
        qwen_api,
        caption_model): i for i,
         image in enumerate(images)}

        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                response = future.result()
                icon_map[i + 1] = response
            except Exception as e:
                # 所有重试都失败后，使用默认值
                print(f"[generate_api] Icon {i+1} processing failed after all retries: {e}. Using default.")
                icon_map[i + 1] = "This is an icon."

    return icon_map


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def merge_text_blocks(text_list, coordinates_list):
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(
    range(
        len(coordinates_list)), key=lambda k: (
            coordinates_list[k][1], coordinates_list[k][0]))
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


def save_trajectory(results_dir, instruction, history_actions, history_image_paths, history_responses):
    trajectory = {
        "task_goal": instruction,
        "history_action": history_actions,
        "history_image_path": history_image_paths,
        "history_response": history_responses
    }
    trajectory_path = os.path.join(results_dir, "trajectory.json")
    with open(trajectory_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)


@timer_decorator
def get_perception_infos(adb_path, device_id, screenshot_file, results_dir, step_num,
                        caption_call_method, caption_model, qwen_api, temp_file, device):
    """获取屏幕感知信息"""
    start_time = time.time()
    curr_time = start_time
    # get_screenshot(adb_path, device_id)
    get_screenshot_u2(device)

    # 保存screenshot和xml（如果文件不存在才保存，避免覆盖噪声注入的文件）
    if step_num is not None:
        screenshot_path = os.path.join(results_dir, f"step_{step_num}.jpg")
        xml_path = os.path.join(results_dir, f"step_{step_num}.xml")

        if not os.path.exists(screenshot_path):
            shutil.copy("/home/mi/MyProj/mobilebench-test-copy/screenshot/screenshot.jpg", screenshot_path)
        else:
            # 文件已存在（由噪声逻辑创建），使用已存在的文件作为screenshot_file
            shutil.copy(screenshot_path, screenshot_file)

        get_xml_u2(device, xml_path)

    curr_time = time.time()
    print("截屏耗时:", curr_time - start_time)

    width, height = Image.open(screenshot_file).size

    text, coordinates = ocr(screenshot_file, _ocr_detection, _ocr_recognition)
    text, coordinates = merge_text_blocks(text, coordinates)

    center_list = [[(coordinate[0] + coordinate[2]) / 2, (coordinate[1] + coordinate[3]) / 2]
                     for coordinate in coordinates]
    draw_coordinates_on_image(screenshot_file, center_list)

    curr_time = time.time()
    print("OCR+画坐标耗时:", curr_time - start_time)

    perception_infos = []
    for i in range(len(coordinates)):
        perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    coordinates = det(screenshot_file, "icon", _groundingdino_model)

    for i in range(len(coordinates)):
        perception_info = {"text": "icon", "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    image_box = []
    image_id = []
    for i in range(len(perception_infos)):
        if perception_infos[i]['text'] == 'icon':
            image_box.append(perception_infos[i]['coordinates'])
            image_id.append(i)

    for i in range(len(image_box)):
        crop(screenshot_file, image_box[i], image_id[i])

    curr_time = time.time()
    print("裁剪耗时:", curr_time - start_time)

    images = get_all_files_in_folder(temp_file)
    if len(images) > 0:
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
        icon_map = {}
        prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
        if caption_call_method == "local":
            for i in range(len(images)):
                image_path = os.path.join(temp_file, images[i])
                icon_width, icon_height = Image.open(image_path).size
                if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                    des = "None"
                else:
                    des = generate_local(_qwen_tokenizer, _qwen_model, image_path, prompt)
                icon_map[i + 1] = des
        else:
            for i in range(len(images)):
                images[i] = os.path.join(temp_file, images[i])
            icon_map = generate_api(images, prompt, qwen_api, caption_model)
        for i, j in zip(image_id, range(1, len(image_id) + 1)):
            if icon_map.get(j):
                perception_infos[i]['text'] = "icon: " + icon_map[j]

    curr_time = time.time()
    print("图标描述耗时:", curr_time - start_time)

    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0] +
    perception_infos[i]['coordinates'][2]) /
    2), int((perception_infos[i]['coordinates'][1] +
    perception_infos[i]['coordinates'][3]) /
     2)]

    curr_time = time.time()
    print("总耗时:", curr_time - start_time)

    return perception_infos, width, height


@timer_decorator
def run_agent(params):
    """
    Run Mobile Agent with given parameters

    Args:
        params: dict with keys:
            - device_id: ADB device ID
            - adb_path: Path to ADB executable
            - results_dir: Directory to save results
            - max_steps: Maximum number of steps
            - instruction: Task instruction
            - API_url: GPT-4o API URL
            - token: API token
            - caption_call_method: "api" or "local"
            - caption_model: Model name for captioning
            - qwen_api: Qwen API key
            - reflection_switch: Enable reflection (default True)
            - memory_switch: Enable memory (default True)
            - add_info: Additional operational knowledge (optional)
            - noise_type: Noise type ("repeat", "unexecuted", "delay", "popup", or None)
            - noise_ratio: Probability of noise injection (default 0.2)

    Returns:
        bool: True if task completed successfully (Stop action), False otherwise
    """
    # Extract parameters
    device_id = params['device_id']
    adb_path = params['adb_path']
    results_dir = params['results_dir']
    max_steps = params['max_steps']
    instruction = params['instruction']
    API_url = params['API_url']
    token = params['token']
    caption_call_method = params['caption_call_method']
    caption_model = params['caption_model']
    qwen_api = params['qwen_api']
    reflection_switch = params.get('reflection_switch', True)
    memory_switch = params.get('memory_switch', True)
    # add_info = params.get('add_info', "If you want to tap an icon of an app,
    # use the action \"Open app\". If you want to exit an app, use the action
    # \"Home\"") 这里是原来的版本
    add_info = params.get(
    'add_info',
     "The required app is already open. Please do not use the \"Open app\" or \"Home\" actions.")
    noise_type = params.get('noise_type', None)
    noise_ratio = params.get('noise_ratio', 0.2)

    device = u2.connect(device_id)

    import subprocess
    # Set ADB keyboard input method
    ime_id = "com.android.adbkeyboard/.AdbIME"
    command = f"{adb_path} -s {device_id} shell ime enable {ime_id}"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    command = f"{adb_path} -s {device_id} shell ime set {ime_id}"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    command = f"{adb_path} -s {device_id} shell settings put secure default_input_method {ime_id}"
    subprocess.run(command, capture_output=True, text=True, shell=True)

    # Load models (only once)
    load_models_once(caption_call_method, caption_model)

    # Initialize state variables
    thought_history = []
    summary_history = []
    action_history = []
    history_actions = []
    history_image_paths = []
    history_responses = []
    summary = ""
    action = ""
    completed_requirements = ""
    memory = ""
    insight = ""
    temp_file = "temp"
    screenshot = "screenshot"

    # Create directories
    if not os.path.exists(temp_file):
        os.mkdir(temp_file)
    else:
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)
    if not os.path.exists(screenshot):
        os.mkdir(screenshot)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    error_flag = False
    task_completed = False
    iter = 0

    print(f"\n{'='*60}")
    print(f"Starting task: {instruction}")
    print(f"Results directory: {results_dir}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    while True:
        iter += 1

        # 检查是否超过最大步数
        if iter > max_steps:
            print(f"Reached maximum steps ({max_steps}). Terminating...")
            history_actions.append({
                "action": "terminate",
                "params": {"text": f"Reached maximum steps limit: {max_steps}"}
            })
            break

        if iter == 1:
            screenshot_file = "./screenshot/screenshot.jpg"
            perception_infos, width, height = get_perception_infos(
                adb_path, device_id, screenshot_file, results_dir, iter,
                caption_call_method, caption_model, qwen_api, temp_file, device
            )
            shutil.rmtree(temp_file)
            os.mkdir(temp_file)

            keyboard = False
            keyboard_height_limit = 0.8 * height    # 还限定为只用0.9，无语了
            for perception_info in perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                # if 'ADB Keyboard' in perception_info['text']: # 这里源码是屎，它可能错识别为"ADB Kevboard"！
                # 再改一下吧，因为输入法会变，可能为clear text
                if 'ADB' in perception_info['text'] or 'Clear' in perception_info['text'] or 'Text' in perception_info['text']:
                    keyboard = True
                    break

        prompt_action = get_action_prompt(
    instruction,
    perception_infos,
    width,
    height,
    keyboard,
    summary_history,
    action_history,
    summary,
    action,
    add_info,
    error_flag,
    completed_requirements,
     memory)
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, screenshot_file)

        try:
            output_action = inference_chat(chat_action, 'gpt-4o', API_url, token)
            thought = output_action.split(
                "### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
            summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
            action = output_action.split(
                "### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
            chat_action = add_response("assistant", output_action, chat_action)
        except BaseException:
            output_action = "Error: GPT-4o API call failed."
            thought = "Error: GPT-4o API call failed."
            summary = "Error: GPT-4o API call failed."
            action = "Error: GPT-4o API call failed."
            chat_action = add_response("assistant", output_action, chat_action)

        # 记录图片路径和响应
        history_image_paths.append(os.path.join(results_dir, f"step_{iter}.jpg"))
        history_responses.append(output_action)

        status = "#" * 50 + " Decision " + "#" * 50
        print(status)
        print(output_action)
        print('#' * len(status))

        if memory_switch:
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            try:
                output_memory = inference_chat(chat_action, 'gpt-4o', API_url, token)
                chat_action = add_response("assistant", output_memory, chat_action)
            except BaseException:
                output_memory = "Error: GPT-4o API call failed."
                chat_action = add_response("assistant", output_memory, chat_action)
            status = "#" * 50 + " Memory " + "#" * 50
            print(status)
            print(output_memory)
            print('#' * len(status))
            output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory

        action_to_execute = None
        if "Open app" in action:
            app_name = action.split("(")[-1].split(")")[0]
            text, coordinate = ocr(screenshot_file, _ocr_detection, _ocr_recognition)
            tap_coordinate = [0, 0]
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [int((coordinate[ti][0] + coordinate[ti][2]) / 2),
                                           int((coordinate[ti][1] + coordinate[ti][3]) / 2)]

                    # 执行动作前不执行tap，留给后面统一处理
                    action_to_execute = {
                        "type": "Open app",
                        "x": name_coordinate[0],
                        "y": name_coordinate[1] - int(coordinate[ti][3] - coordinate[ti][1])
                    }
                    history_actions.append({
                        "action": "click",
                        "params": {"position": [action_to_execute["x"], action_to_execute["y"]]}
                    })
                    break

        elif "Tap" in action:
            coordinate = action.split("(")[-1].split(")")[0].split(", ")
            x, y = int(coordinate[0]), int(coordinate[1])

            action_to_execute = {
                "type": "Tap",
                "x": x,
                "y": y
            }
            history_actions.append({
                "action": "click",
                "params": {"position": [x, y]}
            })

        elif "Swipe" in action:
            coordinate1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
            coordinate2 = action.split("), (")[-1].split(")")[0].split(", ")
            x1, y1 = int(coordinate1[0]), int(coordinate1[1])
            x2, y2 = int(coordinate2[0]), int(coordinate2[1])

            action_to_execute = {
                "type": "Swipe",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
            history_actions.append({
                "action": "swipe",
                "params": {"start": [x1, y1], "end": [x2, y2]}
            })

        elif "Type" in action:
            if "(text)" not in action:
                text = action.split("(")[-1].split(")")[0]
            else:
                text = action.split(" \"")[-1].split("\"")[0]

            action_to_execute = {
                "type": "Type",
                "text": text
            }
            history_actions.append({
                "action": "type",
                "params": {"text": text}
            })

        elif "Back" in action:
            action_to_execute = {
                "type": "Back"
            }
            history_actions.append({
                "action": "back",
                "params": {}
            })

        elif "Home" in action:
            action_to_execute = {
                "type": "Home"
            }
            history_actions.append({
                "action": "home",
                "params": {}
            })

        elif "Stop" in action:
            history_actions.append({
                "action": "terminate",
                "params": {"text": summary}
            })
            task_completed = True
            print(f"\n{'='*60}")
            print(f"Task completed successfully!")
            print(f"{'='*60}\n")
            break

        else:
            action_to_execute = None

        # ==================== 执行动作（带噪声注入）====================
        if action_to_execute is not None:
            # 判断是否需要注入噪声
            random_score = random.random()
            should_inject_noise = noise_type is not None and random_score < noise_ratio
            next_step_num = iter + 1

            # 提取app_name用于查找噪声文件
            app_name = os.path.basename(results_dir).split("_")[0]
            noise_path = os.path.join("noise", app_name, "")

            if should_inject_noise:
                print(f"\n[NOISE] Injecting noise: {noise_type}")

                if noise_type == "repeat":
                    # 执行动作两次
                    print("[NOISE] Executing action twice (repeat)")
                    _execute_action(adb_path, device_id, action_to_execute)
                    time.sleep(5.0)
                    _execute_action(adb_path, device_id, action_to_execute)
                    time.sleep(2.0)

                    # 保存repeat后的页面
                    repeat_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")
                    repeat_marker_path = os.path.join(results_dir, f"step_{next_step_num}_repeat.jpg")

                    # get_screenshot_with_path(adb_path, device_id, repeat_path)
                    get_screenshot_u2(device, repeat_path)
                    get_xml_u2(device, os.path.splitext(repeat_path)[0] + ".xml")
                    copy_page_files(repeat_path, repeat_marker_path)

                elif noise_type == "unexecuted":
                    # 不执行动作
                    print("[NOISE] Action not executed (unexecuted)")
                    time.sleep(2.0)

                    # 保存未执行动作的页面（仍然是当前页面）
                    unexec_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")
                    unexec_marker_path = os.path.join(results_dir, f"step_{next_step_num}_unexecuted.jpg")

                    # get_screenshot_with_path(adb_path, device_id, unexec_path)
                    get_screenshot_u2(device, repeat_path)
                    get_xml_u2(device, os.path.splitext(unexec_path)[0] + ".xml")
                    copy_page_files(unexec_path, unexec_marker_path)

                elif noise_type == "delay":
                    # 正常执行动作
                    print("[NOISE] Action executed, will show delay page")
                    _execute_action(adb_path, device_id, action_to_execute)
                    time.sleep(5.0)

                    # 保存真实页面和delay页面
                    delay_display_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")
                    delay_real_path = os.path.join(results_dir, f"step_{next_step_num}_delay.jpg")

                    # 先保存真实执行后的页面
                    # get_screenshot_with_path(adb_path, device_id, delay_real_path)
                    get_screenshot_u2(device, delay_real_path)
                    get_xml_u2(device, os.path.splitext(delay_real_path)[0] + ".xml")

                    # 复制预设delay页面作为显示页面
                    delay_file = copy_noise_page(noise_path, "delay", delay_display_path)
                    if delay_file is None:
                        print("[WARN] No delay page found, using real page")
                        copy_page_files(delay_real_path, delay_display_path)

                elif noise_type == "popup":
                    # 检查上一步是否有popup
                    curr_step = iter
                    popup_file = check_files_with_prefix(results_dir, f"step_{curr_step}_popup")

                    if not popup_file:
                        # 第一次出现popup，正常执行动作
                        print("[NOISE] First popup, executing action")
                        _execute_action(adb_path, device_id, action_to_execute)
                        time.sleep(5.0)

                        popup_display_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")

                        # 先保存真实页面到临时位置
                        popup_real_temp = os.path.join(results_dir, f"step_{next_step_num}_real_temp.jpg")
                        # get_screenshot_with_path(adb_path, device_id, popup_real_temp)
                        get_screenshot_u2(device, popup_real_temp)
                        get_xml_u2(device, os.path.splitext(popup_real_temp)[0] + ".xml")

                        # 复制预设popup页面
                        popup_file_copied = copy_noise_page(noise_path, "popup", popup_display_path)

                        if popup_file_copied:
                            # 从文件名提取box坐标
                            try:
                                parts = popup_file_copied.replace(".png", "").replace(".jpg", "").split("_")
                                popup_idx = -1
                                for i, p in enumerate(parts):
                                    if p == "popup":
                                        popup_idx = i
                                        break

                                if popup_idx >= 0 and len(parts) > popup_idx + 4:
                                    x1, y1, x2, y2 = parts[popup_idx + 1:popup_idx + 5]
                                    popup_real_path = os.path.join(
    results_dir, f"step_{next_step_num}_popup_{x1}_{y1}_{x2}_{y2}.jpg")
                                else:
                                    popup_real_path = os.path.join(
    results_dir, f"step_{next_step_num}_popup_0_0_100_100.jpg")
                            except BaseException:
                                popup_real_path = os.path.join(
    results_dir, f"step_{next_step_num}_popup_0_0_100_100.jpg")

                            copy_page_files(popup_real_temp, popup_real_path)
                            os.remove(popup_real_temp)
                            if os.path.exists(os.path.splitext(popup_real_temp)[0] + ".xml"):
                                os.remove(os.path.splitext(popup_real_temp)[0] + ".xml")
                        else:
                            print("[WARN] No popup page found, using real page")
                            copy_page_files(popup_real_temp, popup_display_path)
                            os.remove(popup_real_temp)
                            if os.path.exists(os.path.splitext(popup_real_temp)[0] + ".xml"):
                                os.remove(os.path.splitext(popup_real_temp)[0] + ".xml")
                    else:
                        # 已存在popup，检查是否点击关闭
                        last_action_record = history_actions[-1] if history_actions else None

                        if last_action_record and check_close_popup(
                            popup_file, "Tap", last_action_record.get("params", {}).get("position", [])):
                            # 点击关闭，执行动作并显示真实页面
                            print("[NOISE] Popup closed, executing action and showing real page")
                            _execute_action(adb_path, device_id, action_to_execute)
                            time.sleep(5.0)

                            # 显示真实页面
                            next_display_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")
                            # get_screenshot_with_path(adb_path, device_id, next_display_path)
                            get_screenshot_u2(device, next_display_path)
                            get_xml_u2(device, os.path.splitext(next_display_path)[0] + ".xml")
                        else:
                            # 未点击关闭，执行动作但继续显示popup
                            print("[NOISE] Popup still active, executing action but keeping popup display")
                            _execute_action(adb_path, device_id, action_to_execute)
                            time.sleep(5.0)

                            curr_display_path = os.path.join(results_dir, f"step_{curr_step}.jpg")
                            next_display_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")

                            # 保存新的真实页面
                            new_real_temp = os.path.join(results_dir, f"step_{next_step_num}_real_temp.jpg")
                            # get_screenshot_with_path(adb_path, device_id, new_real_temp)
                            get_screenshot_u2(device, new_real_temp)
                            get_xml_u2(device, os.path.splitext(new_real_temp)[0] + ".xml")

                            # 提取旧popup的box坐标并创建新的popup文件
                            new_popup_real_path = popup_file.replace(
    f"step_{curr_step}_popup", f"step_{next_step_num}_popup")
                            new_popup_real_path = os.path.join(results_dir, os.path.basename(new_popup_real_path))

                            copy_page_files(new_real_temp, new_popup_real_path)
                            os.remove(new_real_temp)
                            if os.path.exists(os.path.splitext(new_real_temp)[0] + ".xml"):
                                os.remove(os.path.splitext(new_real_temp)[0] + ".xml")

                            # 继续显示popup
                            copy_page_files(curr_display_path, next_display_path)
            else:
                # 不注入噪声，正常执行
                if noise_type == "popup":
                    # popup模式下，即使不注入噪声也要检查popup状态
                    curr_step = iter
                    popup_file = check_files_with_prefix(results_dir, f"step_{curr_step}_popup")

                    if popup_file:
                        last_action_record = history_actions[-1] if history_actions else None

                        if last_action_record and check_close_popup(
                            popup_file, "Tap", last_action_record.get("params", {}).get("position", [])):
                            # 关闭popup，正常执行
                            print("[INFO] Popup closed by normal action")
                            _execute_action(adb_path, device_id, action_to_execute)
                            time.sleep(2.0)
                        else:
                            # popup仍然存在，执行动作但保持popup显示
                            print("[INFO] Popup still active, executing but keeping display")
                            _execute_action(adb_path, device_id, action_to_execute)
                            time.sleep(2.0)

                            curr_display_path = os.path.join(results_dir, f"step_{curr_step}.jpg")
                            next_display_path = os.path.join(results_dir, f"step_{next_step_num}.jpg")

                            # 保存真实页面
                            new_real_temp = os.path.join(results_dir, f"step_{next_step_num}_real_temp.jpg")
                            # get_screenshot_with_path(adb_path, device_id, new_real_temp)
                            get_screenshot_u2(device, new_real_temp)
                            get_xml_u2(device, os.path.splitext(new_real_temp)[0] + ".xml")

                            new_popup_real_path = popup_file.replace(
    f"step_{curr_step}_popup", f"step_{next_step_num}_popup")
                            new_popup_real_path = os.path.join(results_dir, os.path.basename(new_popup_real_path))

                            copy_page_files(new_real_temp, new_popup_real_path)
                            os.remove(new_real_temp)
                            if os.path.exists(os.path.splitext(new_real_temp)[0] + ".xml"):
                                os.remove(os.path.splitext(new_real_temp)[0] + ".xml")

                            # 继续显示popup
                            copy_page_files(curr_display_path, next_display_path)
                    else:
                        # 没有popup，正常执行
                        _execute_action(adb_path, device_id, action_to_execute)
                        time.sleep(2.0)
                else:
                    # 其他模式，正常执行
                    _execute_action(adb_path, device_id, action_to_execute)
                    time.sleep(2.0)

        # ==================== 结束动作执行 ====================

        time.sleep(3)

        last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = "./screenshot/last_screenshot.jpg"
        last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)

        perception_infos, width, height = get_perception_infos(
            adb_path, device_id, screenshot_file, results_dir, iter + 1,
            caption_call_method, caption_model, qwen_api, temp_file, device
        )
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            # if 'ADB Keyboard' in perception_info['text']: # 这里一样改
            if 'ADB' in perception_info['text'] or 'Clear' in perception_info['text'] or 'Text' in perception_info['text']:
                keyboard = True
                break

        if reflection_switch:
            prompt_reflect = get_reflect_prompt(
    instruction,
    last_perception_infos,
    perception_infos,
    width,
    height,
    last_keyboard,
    keyboard,
    summary,
    action,
     add_info)
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image(
    "user", prompt_reflect, chat_reflect, [
        last_screenshot_file, screenshot_file])

            try:
                output_reflect = inference_chat(chat_reflect, 'gpt-4o', API_url, token)
                reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
                chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            except BaseException:
                output_reflect = "Error: GPT-4o API call failed."
                reflect = 'A'
                chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            status = "#" * 50 + " Reflection " + "#" * 50
            print(status)
            print(output_reflect)
            print('#' * len(status))

            if 'A' in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)

                prompt_planning = get_process_prompt(
    instruction,
    thought_history,
    summary_history,
    action_history,
    completed_requirements,
     add_info)
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)
                try:
                    output_planning = inference_chat(chat_planning, 'gpt-4-turbo', API_url, token)
                except BaseException:
                    output_planning = "Error: GPT-4-turbo API call failed."
                chat_planning = add_response("assistant", output_planning, chat_planning)
                status = "#" * 50 + " Planning " + "#" * 50
                print(status)
                print(output_planning)
                print('#' * len(status))
                completed_requirements = output_planning.split(
                    "### Completed contents ###")[-1].replace("\n", " ").strip()

                error_flag = False

            elif 'B' in reflect:
                error_flag = True
                back(adb_path, device_id)

            elif 'C' in reflect:
                error_flag = True

        else:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)

            prompt_planning = get_process_prompt(
    instruction,
    thought_history,
    summary_history,
    action_history,
    completed_requirements,
     add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            try:
                output_planning = inference_chat(chat_planning, 'gpt-4-turbo', API_url, token)
            except BaseException:
                output_planning = "Error: GPT-4-turbo API call failed."
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()

        os.remove(last_screenshot_file)

    save_trajectory(results_dir, instruction, history_actions, history_image_paths, history_responses)

    return task_completed
