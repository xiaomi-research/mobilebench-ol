"""
M3A LLM Core - 处理M3A的prompt构建和模型调用逻辑
严格按照llm_core_uitars_1_5.py的架构设计
"""

from typing import List, Dict, Any, Optional
import re
import json
from mobilebench.utils import action_parser_tool
from mobilebench.models import execute
from PIL import Image

import ast

# ============ M3A Prompts (从android_world/agents/m3a.py复制) ============
PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message),'
    ' like user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by'
    ' performing actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step.'
    ' At each step, you will be given the current screenshot (including the'
    ' original screenshot and the same screenshot with bounding'
    ' boxes and numeric indexes added to some UI elements) and a history of'
    ' what you have done (in text). Based on these pieces of information and'
    ' the goal, you must choose to perform one of the'
    ' action in the following list (action description followed by the JSON'
    ' format) by outputing the action in the correct JSON format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- If you think the task is not feasible (including cases like you don't"
    ' have enough information or can not perform some necessary actions),'
    ' finish by using the `status` action with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on an element on the screen, similar with the click action'
    ' above, use the numeric label on the bounding box to indicate which'
    ' element you want to long press:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click on'
    ' the target field to start), use the numeric label'
    ' on the bounding box to indicate the target text field:'
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    # '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    # '- Open an app (nothing will happen if the app is not'
    # ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
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
    # '- Use the `open_app` action whenever you want to open an app'
    # ' (nothing will happen if the app is not installed), do not use the'
    # ' app drawer to open an app unless all other ways have failed.\n'
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


# Summary prompt template (参考 m3a.py 的 SUMMARY_PROMPT_TEMPLATE)
SUMMARY_PROMPT_TEMPLATE = (
    'You are tasked with summarizing the outcome of a single action performed'
    ' on an Android phone to complete a user goal/request.\n'
    'The user goal/request is: {goal}.\n'
    'You performed the following action: {action}.\n'
    'The reason for performing this action was: {reason}.\n\n'
    'Before the action, the UI elements on the screen were:\n{before_elements}\n\n'
    'After the action, the UI elements on the screen are:\n{after_elements}\n\n'
    'Please provide a concise summary of what happened after this action was performed.'
    ' Your summary should focus on the changes to the screen and whether the action'
    ' moved you closer to completing the goal.\n\n'
    'Summary of this step: '
)


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}\n\n'
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
    + '{additional_guidelines}'
    + '\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)


class m3a_message_handler(object):
    """处理M3A的消息构建和响应解析"""

    def process_message(
        self,
        goal: str,
        raw_image_path: str,          # 原始截图
        som_image_path: str,          # SOM标注图
        ui_elements_text: str,
        history: Optional[Dict[str, List[str]]] = None,
        additional_guidelines: Optional[List[str]] = None,
    ) -> tuple[List[Dict[str, Any]], int, int]:
        """
        构建M3A的输入消息

        Args:
            goal: 用户任务目标
            raw_image_path: 原始截图路径
            som_image_path: SOM标注截图路径
            ui_elements_text: UI元素列表的文本描述
            history: 历史记录 {"summary": [...]}
            additional_guidelines: 额外的任务指南

        Returns:
            (messages, width, height): 消息列表和图片尺寸

        Note:
            参考m3a.py line 422-425，原版传递 [raw_screenshot, som_screenshot]
        """

        # 构建历史文本
        if history and history.get("summary"):
            summary_list = history["summary"]
            history_text = '\n'.join([
                f'Step {i + 1}- {summary}'
                for i, summary in enumerate(summary_list)
            ])
        else:
            history_text = 'You just started, no action has been performed yet.'

        # 构建额外指南
        extra_guidelines = ''
        if additional_guidelines:
            extra_guidelines = 'For The Current Task:\n'
            for guideline in additional_guidelines:
                extra_guidelines += f'- {guideline}\n'

        # 构建完整prompt
        prompt = ACTION_SELECTION_PROMPT_TEMPLATE.format(
            goal=goal,
            history=history_text,
            ui_elements=ui_elements_text if ui_elements_text else 'Not available',
            additional_guidelines=extra_guidelines,
        )

        # print("⭕M3A Prompt:\n", prompt)

        # 构建消息（传入两张图：raw + som）
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for Android phone operation."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(raw_image_path)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": action_parser_tool.image_to_uri(som_image_path)},
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]

        # 获取图片尺寸（使用SOM图）
        img = Image.open(som_image_path)
        width, height = img.size

        return messages, width, height

    def process_response(self, content: str, width: int, height: int) -> Dict[str, Any]:
        """
        解析M3A的响应

        Args:
            content: LLM返回的文本
            width: 屏幕宽度
            height: 屏幕高度

        Returns:
            action字典，包含action, params, normalized_params, reason等
        """

        # 解析 Reason 和 Action
        reason, action_json_str = self._parse_reason_action(content)

        result = {
            "reason": reason if reason else "",
            "action_json_str": action_json_str if action_json_str else "",
        }

        if not action_json_str:
            result.update({
                "action": "invalid",
                "params": {},
                "normalized_params": {}
            })
            return result

        # 解析JSON action
        try:
            action_dict = self._extract_json(action_json_str)
            action_type = action_dict.get("action_type", "unknown")

            # 根据不同action类型转换为统一格式
            converted = self._convert_m3a_action_to_standard(
                action_type, action_dict, width, height
            )
            result.update(converted)

        except Exception as e:
            print(f"Error parsing action: {e}")
            result.update({
                "action": "invalid",
                "params": {},
                "normalized_params": {}
            })

        return result

    def _parse_reason_action(self, output: str) -> tuple[Optional[str], Optional[str]]:
        """
        从输出中提取Reason和Action
        参考android_world/agents/m3a_utils.py的parse_reason_action_output
        """
        # 输入: "Reason: xxx\nAction: {json...}"
        # 输出: (reason_text, action_json_string)
        # 参考m3a_utils.parse_reason_action_output的实现
        if output is None:
            output = ""
        reason_result = re.search(
            r'Reason:(.*)Action:', output, flags=re.DOTALL
        )
        reason = reason_result.group(1).strip() if reason_result else ""
        action_result = re.search(
            r'Action:(.*)', output, flags=re.DOTALL
        )
        action = action_result.group(1).strip() if action_result else None
        if action:
            extracted = self._extract_json(action)
            if extracted is not None:
                action = json.dumps(extracted)

        return reason, action

    def _extract_json(self, json_str: str) -> Dict[str, Any]:
        """
        提取JSON对象
        参考android_world/agents/agent_utils.py的extract_json
        """
        # 从字符串中提取JSON对象
        # 可以参考agent_utils.extract_json
        """Extracts JSON from string.

        Args:
        s: A string with a JSON in it. E.g., "{'hello': 'world'}" or from CoT:
            "let's think step-by-step, ..., {'hello': 'world'}".

        Returns:
        JSON object.
        """
        pattern = r'\{.*?\}'
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            try:
                return ast.literal_eval(match.group())
            except (SyntaxError, ValueError) as error:
                print(f'Cannot extract JSON, skipping due to error {error}')
                return None
        else:
            print(f'No JSON match in {json_str}')
            return None

    def _convert_m3a_action_to_standard(
        self,
        action_type: str,
        action_dict: Dict[str, Any],
        width: int,
        height: int
    ) -> Dict[str, Any]:
        """
        将M3A的action格式转换为标准格式（兼容adb_executor）

        M3A格式示例:
        - {"action_type": "click", "index": 5}
        - {"action_type": "input_text", "text": "hello", "index": 3}
        - {"action_type": "scroll", "direction": "down", "index": 2}
        - {"action_type": "status", "goal_status": "complete"}

        标准格式示例（参考uitars）:
        - {"action": "click", "params": {"position": [x, y]}, "normalized_params": {...}}

        注意: index到position的转换在agent层完成（需要UI元素信息）
        这里只做基础的action_type映射
        """

        # 映射M3A的action_type到标准格式
        action_type_mapping = {
            "click": "click",
            "long_press": "long_press",
            "input_text": "type",
            "keyboard_enter": "enter",
            # "navigate_home": "home",
            "navigate_back": "back",
            "scroll": "scroll",
            "open_app": "open",
            "wait": "wait",
            "status": "terminate",
            "answer": "answer",
        }

        mapped_action = action_type_mapping.get(action_type, action_type)

        # 返回M3A格式标记，让agent层处理坐标转换
        return {
            "action": mapped_action,
            "params": action_dict,
            "normalized_params": action_dict,
            "m3a_format": True  # 标记这是M3A格式，需要agent层处理index转换
        }


class m3a_Wrapper():
    """M3A的LLM包装器"""

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        url: str = None,
        api_key: Optional[str] = None,
        max_retry: int = 2,
        temperature: float = 0.0,
        max_length: int = 512,
        use_azure: bool = False,
        azure_endpoint: str = None,
        model: str = "gpt-4o",
        api_version: str = "2025-01-01-preview",
    ):
        """
        初始化M3A LLM包装器

        Args:
            url: OpenAI API URL (use_azure=False时使用)
            api_key: API key
            max_retry: 最大重试次数
            temperature: 温度参数
            max_length: 最大token长度
            use_azure: 是否使用Azure OpenAI (默认False)
            azure_endpoint: Azure endpoint (use_azure=True时必须)
            model: 模型名称 (use_azure=True时使用, 默认"gpt-4o")
            api_version: Azure API版本 (默认"2025-01-01-preview")
        """
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.max_length = max_length
        self.use_azure = use_azure

        # 根据use_azure选择不同的client
        if use_azure:
            if not azure_endpoint:
                raise ValueError("azure_endpoint must be provided when use_azure=True")
            if not api_key:
                raise ValueError("api_key must be provided when use_azure=True")

            self.client = execute.AzureOpenAI_Client(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                model=model
            )
            print(f"[M3A] Using Azure OpenAI with model: {model}")
        else:
            if not url:
                raise ValueError("url must be provided when use_azure=False")

            self.url = url
            self.client = execute.OpenAI_Client(url, api_key=api_key)
            print(f"[M3A] Using OpenAI with URL: {url}")

        self.message_handler = m3a_message_handler()

    def predict(
        self,
        goal: str,
        ui_elements_text: str,
        history: Optional[Dict[str, List[str]]] = None,
        additional_guidelines: Optional[List[str]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        调用LLM进行预测（T3A模式：纯文本，无图片）

        Args:
            goal: 任务目标
            ui_elements_text: UI元素文本描述
            history: 历史记录
            additional_guidelines: 额外指南

        Returns:
            (response_text, action_output): 原始响应和解析后的action
        """
        from mobilebench.models import llm_core_t3a不再使用

        # 使用T3A的message handler来构建纯文本prompt
        t3a_handler = llm_core_t3a不再使用.t3a_message_handler()
        text_prompt = t3a_handler.process_message(
            goal, ui_elements_text, history, additional_guidelines
        )

        # 构建消息（纯文本，无图片）
        req_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for Android phone operation."
            },
            {
                "role": "user",
                "content": text_prompt
            }
        ]

        # 调用LLM（纯文本模式），增加最多3次重试，处理可能的API错误
        last_error: Optional[Exception] = None
        while True:
            try:
                response = self.client.call(
                    req_messages,
                    temparature=self.temperature,
                    max_tokens=self.max_length,
                    top_p=0.9,
                )
                # 如果返回中明显包含 Azure 内容过滤错误，则继续重试
                if isinstance(response, str) and "Azure OpenAI API Error" in response and "content_filter" in response:
                    print("[M3A][predict] Azure content filter triggered, retrying...")
                    last_error = RuntimeError("Azure content filter error")
                    continue
                break
            except Exception as e:  # 捕获网络/服务等异常
                print(f"[M3A][predict] LLM call error: {e}, retrying...")
                last_error = e
                continue
        else:
            # 所有重试失败，抛出最后一个错误
            raise last_error if last_error else RuntimeError("LLM call failed after retries")

        # 使用T3A的response parser
        output = t3a_handler.process_response(response)
        if response is None:
            response = "[Error] Empty response from LLM."

        return response, output

    def predict_mm(
        self,
        goal: str,
        raw_image_path: str,          # 新增：原始截图
        current_image_path: str,      # SOM标注图
        ui_elements_text: str,
        history: Optional[Dict[str, List[str]]] = None,
        additional_guidelines: Optional[List[str]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        调用LLM进行预测

        Args:
            goal: 任务目标
            raw_image_path: 原始截图路径（无标注）
            current_image_path: 当前截图路径（带SOM标注）
            ui_elements_text: UI元素文本描述
            history: 历史记录
            additional_guidelines: 额外指南

        Returns:
            (response_text, action_output): 原始响应和解析后的action

        Note:
            参考m3a.py line 422-425，原版传递 [raw_screenshot, som_screenshot]
        """

        # 构建消息（传入双图像）
        req_messages, width, height = self.message_handler.process_message(
            goal, raw_image_path, current_image_path, ui_elements_text,
            history, additional_guidelines
        )

        # 调用LLM，多模态模式，同样加入重试逻辑
        last_error: Optional[Exception] = None
        for _ in range(self.max_retry):
            try:
                response = self.client.call(
                    req_messages,
                    temparature=self.temperature,
                    max_tokens=self.max_length,
                    top_p=0.9,
                )
                if isinstance(response, str) and "Azure OpenAI API Error" in response and "content_filter" in response:
                    print("[M3A][predict_mm] Azure content filter triggered, retrying...")
                    last_error = RuntimeError("Azure content filter error")
                    continue
                break
            except Exception as e:
                print(f"[M3A][predict_mm] LLM call error: {e}, retrying...")
                last_error = e
                continue
        else:
            raise last_error if last_error else RuntimeError("LLM call failed after retries")

        # 解析响应
        output = self.message_handler.process_response(response, width, height)
        if response is None:
            response = "[Error] Empty response from LLM."

        return response, output

    def predict_summary(
        self,
        action_json_str: str,
        reason: str,
        goal: str,
        before_ui_elements_text: str,
        after_ui_elements_text: str,
        before_image_path: str = None,
        after_image_path: str = None,
    ) -> str:
        """
        调用LLM生成本步的总结

        支持两种模式：
        - M3A模式：传入before_image_path和after_image_path（带图片）
        - T3A模式：不传图片（纯文本）

        Args:
            action_json_str: 执行的action的JSON字符串
            reason: 执行该action的原因
            goal: 任务目标
            before_ui_elements_text: 执行前的UI元素描述
            after_ui_elements_text: 执行后的UI元素描述
            before_image_path: 执行前的SOM标注图路径（可选，M3A模式）
            after_image_path: 执行后的SOM标注图路径（可选，M3A模式）

        Returns:
            summary_text: LLM生成的总结文本
        """
        # 构建总结prompt
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            goal=goal,
            action=action_json_str,
            reason=reason,
            before_elements=before_ui_elements_text if before_ui_elements_text else 'Not available',
            after_elements=after_ui_elements_text if after_ui_elements_text else 'Not available',
        )

        # 根据是否有图片选择不同的消息格式
        if before_image_path and after_image_path:
            # M3A模式：带图片
            import base64
            from PIL import Image

            def encode_image(image_path):
                with Image.open(image_path) as img:
                    img_rgb = img.convert('RGB')
                    import io
                    buffer = io.BytesIO()
                    img_rgb.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')

            before_img_base64 = encode_image(before_image_path)
            after_img_base64 = encode_image(after_image_path)

            req_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": summary_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{before_img_base64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{after_img_base64}"}
                        },
                    ]
                }
            ]
        else:
            # T3A模式：纯文本
            req_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for Android phone operation."
                },
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ]

        # 调用LLM，生成总结，同样加入重试逻辑
        last_error: Optional[Exception] = None
        for _ in range(self.max_retry):
            try:
                summary = self.client.call(
                    req_messages,
                    temparature=self.temperature,
                    max_tokens=self.max_length,
                    top_p=0.9,
                )
                if isinstance(summary, str) and "Azure OpenAI API Error" in summary and "content_filter" in summary:
                    print("[M3A][predict_summary] Azure content filter triggered, retrying...")
                    last_error = RuntimeError("Azure content filter error")
                    continue
                break
            except Exception as e:
                print(f"[M3A][predict_summary] LLM call error: {e}, retrying...")
                last_error = e
                continue
        else:
            raise last_error if last_error else RuntimeError("LLM call failed after retries")

        return summary if summary else "Summary generation failed."
