import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from openai import OpenAI
import requests
import traceback

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma-3-4b-it")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")

client = OpenAI(
    api_key="1234567890",
    base_url=LOCAL_LLM_BASE_URL
)

def is_available():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if LOCAL_LLM_MODEL not in model_ids:
            print(f"[!] Модель {LOCAL_LLM_MODEL} не найдена среди доступных моделей локального сервера.")
            return False
        return True
    except Exception:
        return False

def llm_completion(messages, temperature=0.4, response_format=None, model=None):
    kwargs = {}
    if response_format:
        kwargs['response_format'] = response_format
    try:
        response = client.chat.completions.create(
            model=model or LOCAL_LLM_MODEL,
            messages=messages,
            temperature=temperature,
            stream=False,
            max_tokens=2048,
            **kwargs
        )
        usage = getattr(response, 'usage', None)
        if usage is None and isinstance(response, dict):
            usage = response.get('usage', None)
        if usage is None:
            usage = {}
        return {'content': response.choices[0].message.content, 'usage': usage}
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        traceback.print_exc()
        raise 