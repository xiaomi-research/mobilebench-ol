from typing import List, Dict, Any, Optional
import re
from mobilebench.utils import action_parser_tool
from mobilebench.models import execute
from PIL import Image

sys_prompt = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task. \n\n"
    "## Output Format\n\n"
    "Thought: ...\n"
    "Action: ...\n\n\n"
    "## Action Space\n"
    "click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')\n"
    # "long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')\n"
    "type(content=\'\')\n"
    "scroll(direction=\'down or up or right or left\')\n"
    # "open_app(app_name=\'\')\n"
    "press_back()\n"
    "press_home()\n"
    "wait()\n"
    "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
    "## Note\n"
    "- Use English in Thought part.\n\n"
    "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
    "## User Instruction\n"
)


class uitars_message_handler(object):
    def process_message(
        self,
        task: str,
        image_path: str,
        history: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": task
            }
        ]
        messages[1]['content'] = sys_prompt + messages[1]['content']

        # ------------- 拼接历史 -------------
        if history:
            response_list = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])

            # 只保留「回复‑截图」成对数据里的最后 9 条
            # 这里会超token，改为7条
            pairs = list(zip(response_list, screenshot_list))[-7:]
            print("#############pairs#################")
            print("pairs", pairs)

            for reply, shot in pairs:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": action_parser_tool.image_to_uri(shot)},
                            },
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                    }
                )

        # ------------- 当前轮输入 -------------
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(image_path)},
                    },
                ],
            }
        )

        img = Image.open(image_path)
        width, height = img.size

        return messages, width, height

    def process_response(self, content, width, height):
        result = action_parser_tool.parse_agent_output_uitars(content)
        extracted_action = result["action"]
        try:
            if "click" in extracted_action:
                x, y = action_parser_tool.extract_xy_from_point(extracted_action)
                print("x,y", x, y)
                # Calculate the new dimensions
                # new_height, new_width = execute.smart_resize(height, width)
                # new_coordinate = (
                #     int(x / new_width * width),
                #     int(y / new_height * height),
                # )
                # x, y = new_coordinate[0], new_coordinate[1]
                # print(f'Original coordinate: {width}, {height}')
                # print(f'Resized dimensions: {new_width}, {new_height}')
                # print("new_coordinate", new_coordinate)
                # print("x,y", x, y)

                normalized_params = {
                    "position": [round(x / 1000, 2), round(y / 1000, 2)],
                    "click_times": 1
                }
                # 模型输出是0-999的相对坐标，输出改为绝对坐标
                params = {
                    "position": [round(x / 1000 * width), round(y / 1000 * height)],
                    "click_times": 1
                }
                result.update({
                    "action": "click",
                    "params": params,
                    "normalized_params": normalized_params
                })

            elif "type" in extracted_action:
                text = re.search(r"content='(.*?)'", extracted_action).group(1)
                text = text.rstrip("\\n")    # 去除末尾的"\\n"
                text = text.rstrip("\n")    # 去除末尾的"\n"
                result.update({
                    "action": "type",
                    "params": {"text": text},
                    "normalized_params": {"text": text}
                })
            elif "scroll" in extracted_action or "swipe " in extracted_action:
                # "scroll(direction=\'down or up or right or left\')\n"
                dir = action_parser_tool.extract_swipe_direction(extracted_action)

                normalized_params = {
                    "position": [0.50, 0.50],
                    "direction": dir
                }
                params = {
                    "position": [round(0.5 * width), round(0.5 * height)],
                    "direction": dir
                }
                result.update({
                    "action": "scroll",
                    "params": params,
                    "normalized_params": normalized_params
                })
            elif "navigate_back" in extracted_action or "press_back" in extracted_action:
                result.update({
                    "action": "back",
                    "params": {},
                    "normalized_params": {}
                })
            elif "navigate_home" in extracted_action or "press_home" in extracted_action:
                result.update({
                    "action": "home",
                    "params": {},
                    "normalized_params": {}
                })
            elif "wait" in extracted_action.lower():
                result.update({
                    "action": "wait",
                    "params": {},
                    "normalized_params": {}
                })
            elif "finished" in extracted_action:
                try:
                    text = re.search(r"content='(.*?)'", extracted_action).group(1)
                except BaseException:
                    text = ""
                result.update({
                    "action": "terminate",
                    "params": {"text": text},
                    "normalized_params": {"text": text}
                })
            elif "open" in extracted_action.lower():
                try:
                    text = re.search(r"content='(.*?)'", extracted_action).group(1)
                except BaseException:
                    text = ""
                result.update({
                    "action": "open",
                    "params": {"app_name": text},
                    "normalized_params": {"app_name": text},
                })
            else:
                raise ValueError(f"Invalid action: {extracted_action}")
        except BaseException:
            result.update({
                "action": "invalid",
                "params": {},
                "normalized_params": {}
            })
        return result


class uitars_Wrapper():

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        url,
        max_retry: int = 2,
        temperature: float = 0.0,
        max_length: int = 256,
    ):

        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.max_length = max_length
        self.url = url
        self.client = execute.OpenAI_Client(url)
        self.message_handler = uitars_message_handler()

    def predict_mm(self, goal, current_image_path, history):
        req_messages, width, height = self.message_handler.process_message(goal, current_image_path, history)
        response = self.client.call(req_messages, temparature=0, max_tokens=512, top_p=0.9)
        output = self.message_handler.process_response(response, width, height)  # 1080, 2400
        return response, output
