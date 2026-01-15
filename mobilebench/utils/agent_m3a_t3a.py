"""
M3A/T3A Unified Agent
- M3A模式：使用图片+SOM标注（multimodal）
- T3A模式：纯文本UI描述（text-only）
两者共享相同的架构，只是输入格式不同
"""
import os
import time
import json
from typing import Dict, List, Any, Optional
from PIL import Image

from mobilebench.utils import adb_executor
from mobilebench.utils.agent import base_agent
from mobilebench.utils import xml_screen_parser_tool


class m3a_t3a_agent(base_agent):
    """
    M3A/T3A统一Agent
    - 当use_image=True时，使用M3A模式（图片+SOM）
    - 当use_image=False时，使用T3A模式（纯文本）
    """

    def __init__(self, env, llm, use_image=True, debug=False):
        super().__init__(env, llm)

        self.use_image = use_image  # True=M3A模式, False=T3A模式
        self.mode_name = "M3A" if use_image else "T3A"

        # DEBUG: 调试模式开关
        self.debug = debug
        if self.debug:
            self.debug_dir = f"./temp_{self.mode_name.lower()}/debug"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[DEBUG] {self.mode_name} Debug mode enabled, outputs will be saved to {self.debug_dir}")

    def _extract_ui_elements_from_xml(self, xml_string: str) -> List[Dict[str, Any]]:
        """使用统一的XML解析工具"""
        ui_elements = xml_screen_parser_tool.extract_ui_elements_from_xml(xml_string)

        if self.debug:
            debug_path = os.path.join(self.debug_dir, "ui_elements_extracted.json")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(ui_elements, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] {self.mode_name}: Extracted {len(ui_elements)} UI elements")

        return ui_elements

    def _validate_ui_element(self, ui_element: Dict, screen_width: int, screen_height: int) -> bool:
        """过滤无效的UI元素"""
        if not ui_element.get('is_visible', True):
            return False

        if ui_element.get('bounds'):
            x1, y1, x2, y2 = ui_element['bounds']

            if (
                x1 >= x2
                or x1 >= screen_width
                or x2 <= 0
                or y1 >= y2
                or y1 >= screen_height
                or y2 <= 0
            ):
                return False

        return True

    def _generate_ui_elements_description(self, ui_elements: List[Dict], screen_width: int, screen_height: int) -> str:
        """
        生成UI元素描述
        统一使用xml_screen_parser_tool.generate_ui_elements_description_list
        """
        return xml_screen_parser_tool.generate_ui_elements_description_list(
            ui_elements, (screen_width, screen_height)
        )

    def _add_som_annotations(
        self,
        image_path: str,
        ui_elements: List[Dict],
        output_path: str,
        screen_width: int,
        screen_height: int
    ) -> str:
        """M3A专用：在截图上添加SOM标注"""
        try:
            import cv2
            img = cv2.imread(image_path)

            for idx, elem in enumerate(ui_elements):
                if not self._validate_ui_element(elem, screen_width, screen_height):
                    continue

                x1, y1, x2, y2 = elem['bounds']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(idx), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(output_path, img)
            return output_path
        except Exception as e:
            print(f"[WARN] SOM annotation failed: {e}, using original image")
            return image_path

    def _convert_index_to_position(
        self,
        index: int,
        ui_elements: List[Dict],
        screen_width: int,
        screen_height: int
    ) -> tuple[int, int]:
        """将UI元素索引转换为屏幕坐标"""
        if index >= len(ui_elements):
            raise ValueError(f"Index {index} out of range (max: {len(ui_elements) - 1})")

        elem = ui_elements[index]
        return elem['center']

    def _convert_action_to_adb(
        self,
        action_output: Dict[str, Any],
        ui_elements: List[Any],
        screen_width: int,
        screen_height: int
    ) -> Dict[str, Any]:
        """将index-based action转换为adb_executor可执行的格式"""

        # 检查是否是M3A/T3A格式
        if not (action_output.get("m3a_format") or action_output.get("t3a_format")):
            return action_output

        action_type = action_output.get("action")
        params = action_output.get("params", {})

        # 处理需要index的action
        if action_type in ["click", "long_press", "type"]:
            index = params.get("index")
            if index is not None:
                try:
                    x, y = self._convert_index_to_position(index, ui_elements, screen_width, screen_height)

                    if action_type == "type":
                        return {
                            "action": action_type,
                            "params": {
                                "position": [x, y],
                                "text": params.get("text", "")
                            },
                            "normalized_params": {
                                "position": [round(x / screen_width, 2), round(y / screen_height, 2)],
                                "text": params.get("text", "")
                            },
                            "reason": action_output.get("reason", "")
                        }
                    else:
                        return {
                            "action": action_type,
                            "params": {"position": [x, y]},
                            "normalized_params": {
                                "position": [round(x / screen_width, 2), round(y / screen_height, 2)]
                            },
                            "reason": action_output.get("reason", "")
                        }
                except ValueError as e:
                    print(f"Error converting index to position: {e}")
                    return action_output

        # 特殊处理 scroll
        if action_type == "scroll":
            direction = params.get("direction")
            index = params.get("index")

            # 1. 如果params没有"index"，那就返回direction-based格式
            if index is None:
                return {
                    "action": "scroll",
                    "params": {
                        "direction": direction,
                    },
                    "normalized_params": {
                        "direction": direction,
                    },
                    "reason": action_output.get("reason", "")
                }

            # 2. 如果有index，计算具体的滑动坐标
            if index < len(ui_elements):
                x1, y1, x2, y2 = ui_elements[index]['bounds']
                scroll_width = x2 - x1
                scroll_height = y2 - y1
            else:
                x1, y1 = 0, 0
                x2, y2 = screen_width, screen_height
                scroll_width = screen_width
                scroll_height = screen_height

            start_x, start_y = (x1 + x2) // 2, (y1 + y2) // 2

            if direction == 'down':
                end_x, end_y = start_x, start_y + int(scroll_height * 0.5)
            elif direction == 'up':
                end_x, end_y = start_x, start_y - int(scroll_height * 0.5)
            elif direction == 'right':
                end_x, end_y = start_x + int(scroll_width * 0.5), start_y
            elif direction == 'left':
                end_x, end_y = start_x - int(scroll_width * 0.5), start_y
            else:
                print(f"Unknown scroll direction: {direction}")
                return action_output

            return {
                "action": "scroll",
                "params": {
                    "start_position": [start_x, start_y],
                    "end_position": [end_x, end_y],
                },
                "normalized_params": {
                    "start_position": [round(start_x / screen_width, 2), round(start_y / screen_height, 2)],
                    "end_position": [round(end_x / screen_width, 2), round(end_y / screen_height, 2)],
                },
                "reason": action_output.get("reason", "")
            }

        # 处理terminate action
        if action_type == "terminate":
            return action_output

        # 处理不需要参数的action
        if action_type in ["home", "back", "wait", "enter"]:
            return {
                "action": action_type,
                "params": {},
                "normalized_params": {},
                "reason": action_output.get("reason", "")
            }

        # 处理open app action
        if action_type == "open":
            app_name = params.get("app_name", "")
            return {
                "action": "open",
                "params": {"app_name": app_name},
                "normalized_params": {"app_name": app_name},
                "reason": action_output.get("reason", "")
            }

        # 处理answer action
        if action_type == "answer":
            text = params.get("text", "")
            return {
                "action": "answer",
                "params": {"text": text},
                "normalized_params": {"text": text},
                "reason": action_output.get("reason", "")
            }

        print(f"Warning: Unknown action type: {action_type}")
        return action_output

    def step(self, goal: str, path="screenshot_m3a_t3a/") -> tuple[bool, Dict[str, Any]]:
        """执行一步（M3A或T3A模式）"""

        step_index = "step_" + str(len(self.history_image_path) + 1)
        step_prefix = os.path.join(path, step_index)

        # ========== 1. 保存截图和XML ==========
        img_path = step_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        xml_path = step_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)

        if self.debug:
            debug_xml_path = os.path.join(self.debug_dir, "current_hierarchy.xml")
            with open(debug_xml_path, 'w', encoding="utf-8") as f:
                f.write(xml_string)
            print(f"[DEBUG] {self.mode_name} Step {step_index}: Saved screenshot and XML")

        # 获取屏幕尺寸
        img = Image.open(img_path)
        screen_width, screen_height = img.size

        # ========== 2. 提取UI元素 ==========
        ui_elements = self._extract_ui_elements_from_xml(xml_string)
        ui_elements_text = self._generate_ui_elements_description(ui_elements, screen_width, screen_height)

        # ========== 3. 生成SOM标注（仅M3A模式）==========
        som_image_path = None
        if self.use_image:
            som_image_path = step_prefix + "_som.png"
            som_image_path = self._add_som_annotations(
                img_path, ui_elements, som_image_path, screen_width, screen_height
            )

        # ========== 4. 构建历史信息 ==========
        history = {
            "summary": self.summary,
        }

        # ========== 5. 调用LLM获取action ==========
        if self.use_image:
            # M3A模式：传入图片（调用predict_mm）
            response, action_output = self.llm.predict_mm(
                goal=goal,
                raw_image_path=img_path,  # 原始截图
                current_image_path=som_image_path,  # SOM标注图
                ui_elements_text=ui_elements_text,
                history=history,
                additional_guidelines=self.additional_guidelines,
            )
        else:
            # T3A模式：纯文本（调用predict）
            response, action_output = self.llm.predict(
                goal=goal,
                ui_elements_text=ui_elements_text,
                history=history,
                additional_guidelines=self.additional_guidelines,
            )

        # ========== 6. 记录历史 ==========
        self.history_xml_string.append(xml_string)
        self.history_image_path.append(img_path)
        self.history_response.append(response)

        # ========== 7. 转换action格式 ==========
        converted_action = self._convert_action_to_adb(
            action_output, ui_elements, screen_width, screen_height
        )

        self.history_action.append(converted_action)

        if self.debug:
            print(f"[DEBUG] {self.mode_name} Action: {action_output}")
            print(f"[DEBUG] Converted to ADB Action: {converted_action}")

        # ========== 8. 检查是否终止 ==========
        if converted_action['action'] == 'terminate':
            summary_text = action_output.get("reason", "Task completed")
            self.summary.append(summary_text)

            step_data = {
                'history_xml_string': self.history_xml_string,
                "history_image_path": self.history_image_path,
                "history_response": self.history_response,
                "history_action": self.history_action,
                "summary": self.summary,
            }
            return (True, step_data)

        # ========== 9. 执行action ==========
        print("##########model_response#################\n")
        print(response + "\n")
        print("##########execute_action#################\n")

        try:
            print(converted_action["action"])
            print(converted_action["params"])
            print(converted_action["normalized_params"])
            adb_executor.execute_adb_action(converted_action, self.env)

        except Exception as e:
            print('Failed to execute action.')
            print(str(e))
            print("action_output", converted_action)
            summary_error = 'Can not execute the action: ' + str(e)
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

        # ========== 10. 获取执行后状态并生成总结 ==========
        after_xml_string = self.env.dump_hierarchy()
        after_ui_elements = self._extract_ui_elements_from_xml(after_xml_string)
        after_ui_elements_text = self._generate_ui_elements_description(
            after_ui_elements, screen_width, screen_height
        )

        try:
            reason = action_output.get("reason", "")
            action_json_str = action_output.get("action_json_str", str(action_output))

            # 根据模式调用不同的summary方法
            if self.use_image:
                # M3A模式：需要传入图片
                after_img_path = step_prefix + "_after.png"
                after_pixels = self.env.screenshot()
                after_pixels.save(after_img_path)

                after_som_path = step_prefix + "_after_som.png"
                after_som_path = self._add_som_annotations(
                    after_img_path, after_ui_elements, after_som_path,
                    screen_width, screen_height
                )

                summary = self.llm.predict_summary(
                    action_json_str=action_json_str,
                    reason=reason,
                    goal=goal,
                    before_ui_elements_text=ui_elements_text,
                    after_ui_elements_text=after_ui_elements_text,
                    before_image_path=som_image_path,
                    after_image_path=after_som_path,
                )
            else:
                # T3A模式：纯文本
                summary = self.llm.predict_summary(
                    action_json_str=action_json_str,
                    reason=reason,
                    goal=goal,
                    before_ui_elements_text=ui_elements_text,
                    after_ui_elements_text=after_ui_elements_text,
                )

            full_summary = f'Action selected: {action_json_str}. {summary}'
            self.summary.append(full_summary)

            print(f"Summary: {summary}")

        except Exception as e:
            print(f"Error generating summary: {e}")
            error_summary = f'Action selected: {action_json_str}. Error generating summary.'
            self.summary.append(error_summary)

        step_data = {
            'history_xml_string': self.history_xml_string,
            "history_image_path": self.history_image_path,
            "history_response": self.history_response,
            "history_action": self.history_action,
            "summary": self.summary,
        }
        return (False, step_data)
