from openai import AzureOpenAI
import openai
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import base64
import math
import requests
from PIL import Image
from io import BytesIO
from mobilebench.utils import representation_utils
from mobilebench.utils import m3a_utils
from mobilebench.utils import action_parser_tool
from mobilebench.utils import xml_screen_parser_tool
import numpy as np


PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Complete some tasks described in the request by performing actions on the phone using visual understanding.\n\n'
    'At each step, you will be given the history path, current screenshot (before action) and the task goal.\n'
    'You must analyze the screen and output your action decision:\n'
    '1. A brief reasoning in Chinese: Why and where to take the next action.\n'
    '2. A structured action command in format below.\n\n'
    'Supported Actions:\n'
    '- Click/tap a position on screen: `{{"click(start_point=(x1,y1))"}}`\n'
    '- Scroll the screen: `{{"scroll(start_box=(x1,y1), end_box=(x2,y2))"}}`\n'
    '- Type text into an input field when searching: `{{"type(content=...)''}}`\n'
   # '- Open an app: `{{"action_type": "open_app", "app_name": "<name>"}}`\n'
    '- Press home button: `{{press_home()}}`\n'
    '- Press back button: `{{press_back()}}`\n'
    '- Wait for UI update: `{{wait()}}`\n'
    '- The task is finished: `{{finished(content='')}}`\n'
    'You must only use the above 7 actions. \n'
    'Use coordinates based on your visual understanding of the screenshot.\n'
)


GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}\n\n'
    'the size of the screenshot is 1080 * 2400\n'
    'Here is a history of what you have done so far:\n{history}\n\n'
    'The current screenshot and the same screenshot with bounding boxes'
    ' and labels added are also given to you.\n'
    'Here is a list of detailed'
    ' information for some of the UI elements (notice that some elements in'
    ' this list may not be visible in the current screen and so you can not'
    ' interact with it, can try to scroll the screen to reveal it first),'
    ' the numeric indexes are'
    ' consistent with the ones in the labeled screenshot:\n{ui_elements}\n'
    + GUIDANCE
    + '\nNow output your decision:\n'
    'Thought: （请用中文推理你的意图和位置）\n'
    'Action: (structured action selected from above 7 actions)...\n'
    # + 'for example: Thought: 我需要点击右下角的方块，那是打开个人主页的入口 \n'
    # 'Action : click(start_point=<900,930>)'
)

SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


class Azure_Openai_Client:
    def __init__(self, model, api_key, azure_endpoint, api_version, temperature, max_tokens):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

    def call(self, messages):
        result = ""
        trial = 0
        while trial < 3:
            trial += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result = response.choices[0].message.content
                return result
            except openai.AuthenticationError as e:
                print(f"OpenAI API returned an Authentication Error: {e}")
            except openai.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
            except openai.BadRequestError as e:
                # Handle connection error here
                print(f"Invalid Request Error: {e}")
            except openai.RateLimitError as e:
                # Handle rate limit error
                print(f"OpenAI API request exceeded rate limit: {e}")
                # 等待50s
                print("等待 50s...")
                time.sleep(50)
            except openai.InternalServerError as e:
                # Handle Service Unavailable error
                print(f"Service Unavailable: {e}")
            except openai.APITimeoutError as e:
                # Handle request timeout
                print(f"Request timed out: {e}")
            except openai.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
            except BaseException:
                # Handles all other exceptions
                print("An exception has occured.")
        return result


class gpt4o_message_handler(object):
    def process_message(
        self,
        task: str,
        image_path: str,
        history: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:

        xml_path = image_path.replace(".png", ".xml")
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_string = f.read()
        before_ui_elements = representation_utils.xml_dump_to_ui_elements(xml_string)

        history_summaries = ""
        if history:
            response_list = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])
            pairs = list(zip(response_list, screenshot_list))[-9:]

            for i, (reply, _) in enumerate(pairs):
                history_summaries += f"[Step {i + 1}] {reply.strip()}\n"

        prompt_text = ACTION_SELECTION_PROMPT_TEMPLATE.format(
            goal=task,
            history=history_summaries.strip(),
            additional_guidelines=GUIDANCE,
            ui_elements=before_ui_elements
        )
        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }

        # 起始 messages
        messages: List[Dict[str, Any]] = [sys_prompt_block]
        messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
        }] + messages

        if history:
            response_list = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])

            # 只保留「回复‑截图」成对数据里的最后 9 条
            pairs = list(zip(response_list, screenshot_list))[-9:]

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
                        "content": [{"type": "text", "text": reply}],
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(image_path)},
                    }
                ],
            }
        )
        return messages

    def process_message_som_elements_list(
        self,
        task: str,
        image_path: str,
        xml_string: str,
        history: Optional[Dict[str, List[str]]] = None,
        step_prefix=""
    ) -> List[Dict[str, Any]]:

        before_ui_elements = representation_utils.xml_dump_to_ui_elements(xml_string)
        before_ui_elements_list = xml_screen_parser_tool._generate_ui_elements_description_list(
            before_ui_elements,
            (1080, 2400),
        )
        before_pixels = np.asarray(Image.open(image_path)).copy()
        for index, ui_element in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, (1080, 2400)):
                m3a_utils.add_ui_element_mark(
                    before_pixels,
                    ui_element,
                    index,
                    (1080, 2400),
                    (0, 0, 1080, 2400),
                    0,
                )

        save_path = f"{step_prefix}_som.png"
        image = Image.fromarray(before_pixels)
        image.save(save_path, format='PNG')

        history_summaries = []
        if history:
            response_list = history.get("history_response", [])
            for i, step_info in enumerate(response_list):
                history_summaries.append('Step ' + str(i + 1) + '- ' + step_info)
        history_context = '\n'.join(history_summaries)
        prompt_text = ACTION_SELECTION_PROMPT_TEMPLATE.format(
            goal=task,
            history=history_context,
            ui_elements=before_ui_elements_list
        )

        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }

        # 起始 messages
        messages: List[Dict[str, Any]] = [sys_prompt_block]
        messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
        }] + messages

        if history:
            response_list = history.get("history_response", [])
            screenshot_list = history.get("history_image_path", [])

            # 只保留「回复‑截图」成对数据里的最后 9 条
            pairs = list(zip(response_list, screenshot_list))[-9:]

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
                        "content": [{"type": "text", "text": reply}],
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(image_path)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(save_path)},
                    }
                ],
            }
        )
        return messages

    def process_message_summary(self, history, after_pixels, after_xml_string, goal):

        before_path = history["history_image_path"][-1]
        before_image = Image.open(before_path)
        before_pixels = np.asarray(before_image).copy()
        before_xml_string = history["history_xml_string"][-1]
        reason = history["history_response"][-1]
        action = history["history_action"][-1]

        before_ui_elements = representation_utils.xml_dump_to_ui_elements(before_xml_string)
        before_ui_elements_list = xml_screen_parser_tool._generate_ui_elements_description_list(
            before_ui_elements,
            (1080, 2400),
        )
        after_ui_elements = representation_utils.xml_dump_to_ui_elements(after_xml_string)
        after_ui_elements_list = xml_screen_parser_tool._generate_ui_elements_description_list(
            after_ui_elements, (1080, 2400)
        )
        for index, ui_element in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, (1080, 2400)):
                m3a_utils.add_ui_element_mark(
                    before_pixels,
                    ui_element,
                    index,
                    (1080, 2400),
                    (0, 0, 1080, 2400),
                    0,
                )
        for index, ui_element in enumerate(after_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, (1080, 2400)):
                m3a_utils.add_ui_element_mark(
                    after_pixels,
                    ui_element,
                    index,
                    (1080, 2400),
                    (0, 0, 1080, 2400),
                    0,
                )
        m3a_utils.add_screenshot_label(before_pixels, 'before')
        m3a_utils.add_screenshot_label(after_pixels, 'after')
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            goal=goal,
            action=action,
            reason=reason,
            before_elements=before_ui_elements_list,
            after_elements=after_ui_elements_list
        )
        sys_prompt_block = {
            "role": "user",
            "content": [
                {"type": "text", "text": summary_prompt},
            ],
        }
        messages: List[Dict[str, Any]] = [sys_prompt_block]
        messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
        }] + messages
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(before_pixels)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(after_pixels)},
                    }
                ],
            }
        )
        return messages

    def process_response(self, content, width, height):
        result = action_parser_tool.parse_agent_output(content)
        extracted_action = result["action"]
        try:
            if "click" in extracted_action:
                x, y = action_parser_tool.extract_xy_from_point(extracted_action)
                normalized_params = {
                    "position": [round(x / width, 2), round(y / height, 2)],
                    "click_times": 1
                }
                params = {
                    "position": [x, y],
                    "click_times": 1
                }
                result.update({
                    "action": "click",
                    "params": params,
                    "normalized_params": normalized_params
                })
            elif "type" in extracted_action:
                text = re.search(r"content='(.*?)'", extracted_action).group(1)
                result.update({
                    "action": "type",
                    "params": {"text": text},
                    "normalized_params": {"text": text}
                })
            elif "scroll" in extracted_action or "swipe " in extracted_action:
                x1, y1, x2, y2 = action_parser_tool.extract_swipe_points(extracted_action)
                normalized_params = {
                    "start_position": [round(x1 / width, 2), round(y1 / height, 2)],
                    "end_position": [round(x2 / width, 2), round(y2 / height, 2)],
                    "press_duration": -1
                }
                params = {
                    "start_position": [int(x1), int(y1)],
                    "end_position": [int(x2), int(y2)],
                    "press_duration": -1
                }
                result.update({
                    "action": "swipe",
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


class GPT4oWrapper():
    """OpenAI GPT4 wrapper.

    Attributes:
      openai_api_key: The class gets the OpenAI api key either explicitly, or
        through env variable in which case just leave this empty.
      max_retry: Max number of retries when some error happens.
      temperature: The temperature parameter in LLM to control result stability.
      model: GPT model to use based on if it is multimodal.
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        max_retry: int = 2,
        temperature: float = 0.0,
        max_length: int = 256,
    ):

        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        max_tokens = 3500
        self.max_length = max_length
        model = "gpt-4o"
        api_key = " "
        azure_endpoint = " "
        api_version = "2025-01-01-preview"
        self.client = Azure_Openai_Client(
    model,
    api_key,
    azure_endpoint,
    api_version,
    temperature,
     max_tokens=max_tokens)
        self.message_handler = gpt4o_message_handler()

    def predict_mm_som(self, goal, current_image_path, current_xml_string, history, step_prefix):

        req_messages = self.message_handler.process_message_som_elements_list(
            goal, current_image_path, current_xml_string, history, step_prefix)
        response = self.client.call(req_messages)
        output = self.message_handler.process_response(response, 1080, 2400)
        return response, output

    def predict_mm(self, goal, current_image_path, history):

        req_messages = self.message_handler.process_message(goal, current_image_path, history)
        response = self.client.call(req_messages)
        output = self.message_handler.process_response(response, 1080, 2400)
        return response, output

    def summarize(self, history, after_pixels, after_xml_string, goal):
        summary_messages = self.message_handler.process_message_summary(history, after_pixels, after_xml_string, goal)
        response = self.client.call(summary_messages)
        return response

# gpt = GPT4oWrapper()
# image2 = "result\\gpt4o_m3a_test_longtail\\bili_0\\step_1.png"
# image1 = "result\\gpt4o_m3a_test_longtail\\bili_0\\step_1.png"
# xml_path = "result\\gpt4o_m3a_test_longtail\\bili_0\\step_1.xml"
# history = {}
# history["history_image_path"] = [image1,image2]

# goal = "open app"
# # before_image = Image.open(before_path)
# # before_pixels = np.asarray(before_image).copy()
# after_pixels = np.asarray(Image.open(image2)).copy()
# with open(xml_path, 'r', encoding='utf-8') as f:
#     xml_string = f.read()
# history["history_xml_string"] = [xml_string]
# gpt.summarize(history,after_pixels,xml_string,goal)
