"""
MobileAgent v2 Agent with Noise Support
形式来源：参考 agent_noise.py 的噪声注入逻辑和结构
逻辑来源：参考 run_agent.py 中的噪声处理部分（repeat/unexecuted/delay/popup）
"""
import sys
import os
import time
import copy
import shutil
import random
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# 继承自基础 agent
from mobilebench.utils.agent_mobileagent_v2 import base_agent as base_agent_v2
from mobilebench.utils.agent_mobileagent_v2 import _ocr_detection, _ocr_recognition, ocr
from mobilebench.utils import adb_executor


def copy_page_files(source_path, target_path):
    """
    复制页面文件（包括图片和XML）
    逻辑来源：直接从 run_agent.py 复制
    """
    shutil.copy(source_path, target_path)
    
    source_xml = os.path.splitext(source_path)[0] + ".xml"
    target_xml = os.path.splitext(target_path)[0] + ".xml"
    if os.path.exists(source_xml):
        shutil.copy(source_xml, target_xml)


def copy_noise_page(predefined_path, keyword, target_path):
    """
    从预定义路径随机选择一个噪声页面并复制
    逻辑来源：直接从 run_agent.py 复制
    """
    if not os.path.exists(predefined_path):
        print(f"[WARN] Noise path does not exist: {predefined_path}")
        return None
    
    files = []
    for item in os.listdir(predefined_path):
        full_path = os.path.join(predefined_path, item)
        if os.path.isfile(full_path) and item.startswith(keyword) and (item.endswith(".png") or item.endswith(".jpg")):
            files.append(item)
    
    if not files:
        print(f"[WARN] No noise files found with keyword '{keyword}' in {predefined_path}")
        return None
    
    print(f"[INFO] Found {len(files)} noise files: {files}")
    
    random_file = random.choice(files)
    source_path = os.path.join(predefined_path, random_file)
    
    copy_page_files(source_path, target_path)
    
    print(f"[INFO] Copied noise page: {random_file} -> {target_path}")
    return random_file


def check_files_with_prefix(path, prefix):
    """
    检查指定路径下是否有以prefix开头的文件
    逻辑来源：直接从 run_agent.py 复制
    """
    if not os.path.exists(path):
        return False
    
    for filename in os.listdir(path):
        if filename.startswith(prefix) and os.path.isfile(os.path.join(path, filename)):
            return filename
    return False


def check_close_popup(filename, action_output):
    """
    检查动作是否点击了弹窗关闭区域
    
    形式来源：参考 agent_noise.py 中的 check_close_popup
    逻辑来源：参考 run_agent.py 中的 check_close_popup 函数
    """
    if action_output['action'] != 'click':
        return False
    
    # 将 .xml 转换为 .png
    if filename.endswith(".xml"):
        filename = filename.replace(".xml", ".png")
    
    # 从文件名中提取 box 坐标
    # 格式: step_N_popup_x1_y1_x2_y2.png
    try:
        parts = filename.split("_")
        popup_idx = -1
        for i, p in enumerate(parts):
            if p == "popup":
                popup_idx = i
                break
        
        if popup_idx >= 0 and len(parts) > popup_idx + 4:
            x1 = int(parts[popup_idx + 1])
            y1 = int(parts[popup_idx + 2])
            x2 = int(parts[popup_idx + 3])
            y2 = int(parts[popup_idx + 4].split(".")[0])
        else:
            return False
    except (ValueError, IndexError):
        print(f"[WARN] Cannot parse popup box from filename: {filename}")
        return False
    
    # 检查点击坐标是否在弹窗区域内
    try:
        if "position" not in action_output["params"]:
            return False
        point = action_output["params"]["position"]
        x = point[0]
        y = point[1]
    except Exception as e:
        print(f"[WARN] Cannot parse tap coordinates: {action_output}, error: {e}")
        return False
    
    # 判断是否在区域内
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        print(f"[INFO] Tap ({x}, {y}) is inside popup box ({x1}, {y1}, {x2}, {y2})")
        return True
    
    return False


class base_agent(base_agent_v2):
    """
    支持噪声注入的 MobileAgent v2 Agent
    
    形式来源：参考 agent_noise.py 中的 base_agent 结构
    逻辑来源：参考 run_agent.py 中的噪声处理逻辑
    """

    def __init__(
        self,
        env,
        llm,
        qwen_api: str = "",
        caption_model: str = "qwen-vl-max-latest",
        noise_type: str = "delay"  # repeat/unexecuted/delay/popup
    ):
        """
        初始化带噪声的 Agent
        
        形式来源：参考 agent_noise.py 的 __init__
        逻辑来源：参考 run_agent.py 中的噪声参数
        """
        super().__init__(env, llm, qwen_api, caption_model)
        
        self.noise_ratio = 0.2
        self.noise_type = noise_type  # ["repeat", "unexecuted", "delay", "popup"]
    
    def save_page_by_prefix(self, page_prefix="screenshot/step_0"):
        """
        按前缀保存页面
        形式来源：参考 agent_noise.py
        """
        img_path = page_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        xml_path = page_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)
    
    def copy_page1_page2(self, source_path, target_path):
        """
        复制页面文件
        形式来源：参考 agent_noise.py
        """
        copy_page_files(source_path, target_path)

    def step(self, goal: str, path="screenshot/"):
        """
        执行一步（带噪声注入）
        
        形式来源：参考 agent_noise.py 的 step 函数结构
        逻辑来源：参考 run_agent.py 中的噪声注入逻辑
        
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
        img_path = step_prefix + ".png"
        xml_path = step_prefix + ".xml"
        screenshot_file = "./screenshot/screenshot.jpg"
        
        # 获取感知信息
        step_num = len(self.history_image_path) + 1
        
        # 检查是否已存在文件（噪声可能已创建）
        if not os.path.exists(img_path):
            perception_infos, width, height, keyboard = self.get_perception_infos(
                screenshot_file,
                step_num,
                path
            )
        else:
            # 文件已存在，直接使用
            perception_infos, width, height, keyboard = self.get_perception_infos(
                img_path,
                None,  # 不覆盖已存在的文件
                None
            )
        
        # 读取 XML
        if os.path.exists(xml_path):
            with open(xml_path, encoding='utf-8') as f:
                xml_string = f.read()
        else:
            xml_string = self.env.dump_hierarchy()
            with open(xml_path, 'w', encoding="utf-8") as f:
                f.write(xml_string)
        
        # 调用 LLM 进行预测
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

        # 处理 Open app 动作
        if action_output['action'] == 'open_app':
            app_name = action_output['params'].get('app_name', '')
            text, coordinates = ocr(screenshot_file, _ocr_detection, _ocr_recognition)
            
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [
                        int((coordinates[ti][0] + coordinates[ti][2]) / 2),
                        int((coordinates[ti][1] + coordinates[ti][3]) / 2)
                    ]
                    
                    action_output['action'] = 'click'
                    action_output['params'] = {
                        'position': [
                            name_coordinate[0],
                            name_coordinate[1] - int(coordinates[ti][3] - coordinates[ti][1])
                        ]
                    }
                    action_output['normalized_params'] = action_output['params']
                    break
        
        # ==================== 噪声注入逻辑（逻辑来源：run_agent.py）====================
        random_score = random.random()
        
        if random_score < self.noise_ratio:
            print(f"\n[NOISE] Injecting noise: {self.noise_type}")

            step_index_next = "step_" + str(len(self.history_image_path) + 1)
            step_prefix_next = os.path.join(path, step_index_next)
            app_name = path.split(os.path.sep)[-1].split("_")[0]
            noise_path = os.path.join("noise", app_name, "")
            
            if self.noise_type == "repeat":
                # 执行动作两次
                print("[NOISE] Executing action twice (repeat)")
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(5.0)
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(2.0)
                
                # 保存repeat后的页面
                self.save_page_by_prefix(page_prefix=step_prefix_next)
                self.save_page_by_prefix(page_prefix=step_prefix_next + "_repeat")
            
            elif self.noise_type == "unexecuted":
                # 不执行动作
                print("[NOISE] Action not executed (unexecuted)")
                time.sleep(2.0)
                
                # 保存未执行动作的页面（仍然是当前页面）
                self.save_page_by_prefix(page_prefix=step_prefix_next)
                self.save_page_by_prefix(page_prefix=step_prefix_next + "_unexecuted")
            
            elif self.noise_type == "delay":
                # 正常执行动作
                print("[NOISE] Action executed, will show delay page")
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(5.0)
                
                # 保存真实页面和delay页面
                delay_display_path = step_prefix_next + ".png"
                delay_real_path = step_prefix_next + "_delay.png"
                
                # 先保存真实执行后的页面
                self.save_page_by_prefix(page_prefix=step_prefix_next + "_delay")
                
                # 复制预设delay页面作为显示页面
                delay_file = copy_noise_page(noise_path, "delay", delay_display_path)
                if delay_file is None:
                    print("[WARN] No delay page found, using real page")
                    copy_page_files(delay_real_path, delay_display_path)
            
            elif self.noise_type == "popup":
                # 检查上一步是否有popup
                step_index_last = "step_" + str(len(self.history_image_path))
                popup_file = check_files_with_prefix(path, step_index_last + "_popup")
                
                if not popup_file:
                    # 第一次出现popup，正常执行动作
                    print("[NOISE] First popup, executing action")
                    adb_executor.execute_adb_action(action_output, self.env)
                    time.sleep(5.0)
                    
                    popup_display_path = step_prefix_next + ".png"
                    
                    # 先保存真实页面到临时位置
                    popup_real_temp = step_prefix_next + "_real_temp.png"
                    self.save_page_by_prefix(page_prefix=step_prefix_next + "_real_temp")
                    
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
                                popup_real_path = step_prefix_next + f"_popup_{x1}_{y1}_{x2}_{y2}.png"
                            else:
                                popup_real_path = step_prefix_next + "_popup_0_0_100_100.png"
                        except Exception:
                            popup_real_path = step_prefix_next + "_popup_0_0_100_100.png"

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
                    if check_close_popup(popup_file, action_output):
                        # 点击关闭，执行动作并显示真实页面
                        print("[NOISE] Popup closed, executing action and showing real page")
                        adb_executor.execute_adb_action(action_output, self.env)
                        time.sleep(5.0)
                        
                        # 显示真实页面
                        self.save_page_by_prefix(page_prefix=step_prefix_next)
                    else:
                        # 未点击关闭，执行动作但继续显示popup
                        print("[NOISE] Popup still active, executing action but keeping popup display")
                        adb_executor.execute_adb_action(action_output, self.env)
                        time.sleep(5.0)
                        
                        curr_display_path = step_prefix + ".png"
                        next_display_path = step_prefix_next + ".png"
                        
                        # 保存新的真实页面
                        new_real_temp = step_prefix_next + "_real_temp.png"
                        self.save_page_by_prefix(page_prefix=step_prefix_next + "_real_temp")
                        
                        # 提取旧popup的box坐标并创建新的popup文件
                        new_popup_real_path = popup_file.replace(step_index_last + "_popup", step_index_next + "_popup")
                        new_popup_real_path = os.path.join(path, os.path.basename(new_popup_real_path))
                        
                        copy_page_files(new_real_temp, new_popup_real_path)
                        os.remove(new_real_temp)
                        if os.path.exists(os.path.splitext(new_real_temp)[0] + ".xml"):
                            os.remove(os.path.splitext(new_real_temp)[0] + ".xml")
                        
                        # 继续显示popup
                        copy_page_files(curr_display_path, next_display_path)
        else:
            # 不注入噪声，正常执行
            if self.noise_type == "popup":
                # popup模式下，即使不注入噪声也要检查popup状态
                step_index_last = "step_" + str(len(self.history_image_path))
                popup_file = check_files_with_prefix(path, step_index_last + "_popup")
                
                if popup_file:
                    if check_close_popup(popup_file, action_output):
                        # 关闭popup，正常执行
                        print("[INFO] Popup closed by normal action")
                        adb_executor.execute_adb_action(action_output, self.env)
                        time.sleep(2.0)
                    else:
                        # popup仍然存在，执行动作但保持popup显示
                        print("[INFO] Popup still active, executing but keeping display")
                        adb_executor.execute_adb_action(action_output, self.env)
                        time.sleep(2.0)
                        
                        step_index_next = "step_" + str(len(self.history_image_path) + 1)
                        step_prefix_next = os.path.join(path, step_index_next)
                        curr_display_path = step_prefix + ".png"
                        next_display_path = step_prefix_next + ".png"
                        
                        # 保存真实页面
                        new_real_temp = step_prefix_next + "_real_temp.png"
                        self.save_page_by_prefix(page_prefix=step_prefix_next + "_real_temp")
                        
                        new_popup_real_path = popup_file.replace(step_index_last + "_popup", step_index_next + "_popup")
                        new_popup_real_path = os.path.join(path, os.path.basename(new_popup_real_path))
                        
                        copy_page_files(new_real_temp, new_popup_real_path)
                        os.remove(new_real_temp)
                        if os.path.exists(os.path.splitext(new_real_temp)[0] + ".xml"):
                            os.remove(os.path.splitext(new_real_temp)[0] + ".xml")
                        
                        # 继续显示popup
                        copy_page_files(curr_display_path, next_display_path)
                else:
                    # 没有popup，正常执行
                    adb_executor.execute_adb_action(action_output, self.env)
                    time.sleep(2.0)
            else:
                # 其他模式，正常执行
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(2.0)

        time.sleep(self.wait_after_action_seconds)

        step_data = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }
        return (False, step_data)
