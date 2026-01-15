import base64
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 运行时间: {end_time - start_time:.6f} 秒")
        return result
    return wrapper


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@timer_decorator
@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((requests.exceptions.RequestException, KeyError, ValueError)),
    reraise=True
)
def inference_chat_old(chat, model, api_url, token):
    headers = {
        "Content-Type": "application/json",
        "api-key": token
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    try:
        res = requests.post(api_url, headers=headers, json=data, timeout=60)
        res.raise_for_status()  # Raise HTTPError for bad status codes
        res_json = res.json()
        res_content = res_json['choices'][0]['message']['content']
        return res_content
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response: {e.response.json()}")
            except Exception:
                print(f"Response text: {e.response.text}")
        raise
    except (KeyError, ValueError) as e:
        print(f"Response parsing error: {e}")
        try:
            print(f"Response JSON: {res_json}")
        except Exception:
            print("Could not parse response")
        raise


@timer_decorator
@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(6),
    reraise=True
)
def inference_chat(chat, model, token, api_version="2025-01-01-preview", azure_endpoint="https://ui-agent-exp.openai.azure.com/"):
    """
    使用 Azure OpenAI SDK 进行推理

    Args:
        chat: 对话历史，格式为 [(role, content), ...]
        model: 模型名称
        token: Azure OpenAI API key
        api_version: API 版本
        azure_endpoint: Azure endpoint URL

    Returns:
        str: 模型返回的内容
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=token,
        azure_endpoint=azure_endpoint,
        api_version=api_version
    )

    messages = [{"role": role, "content": content} for role, content in chat]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        seed=1234
    )

    return response.choices[0].message.content
