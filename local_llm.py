import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from openai import OpenAI
import traceback
from progress import progress

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma-3-4b-it")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
LOCAL_LLM_CONTEXT_WINDOW = int(os.getenv("LOCAL_LLM_CONTEXT_WINDOW", "32768"))  # Размер окна контекста (32K)
LOCAL_LLM_SLIDING_WINDOW = int(os.getenv("LOCAL_LLM_SLIDING_WINDOW", "16384"))  # Размер sliding window (16K)
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096"))  # Максимальное количество токенов в ответе

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

def llm_completion(messages, temperature=0.4, response_format=None, model=None, debug=False):
    kwargs = {}
    if response_format:
        kwargs['response_format'] = response_format
    
    # Добавляем параметры контекста для LMStudio через extra_body
    kwargs['extra_body'] = {
        'context_window': LOCAL_LLM_CONTEXT_WINDOW,
        'sliding_window': LOCAL_LLM_SLIDING_WINDOW
    }
    
    try:
        if debug:
            # В режиме debug используем streaming
            response = client.chat.completions.create(
                model=model or LOCAL_LLM_MODEL,
                messages=messages,
                temperature=temperature,
                stream=True,
                max_tokens=LOCAL_LLM_MAX_TOKENS,
                **kwargs
            )
            content = ""
            current_line = ""
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        current_line += delta.content
                        content += delta.content
                        # Если встретили перенос строки или знак пунктуации, выводим накопленную строку
                        if '\n' in current_line or any(p in current_line for p in '.!?,;:'):
                            progress.write(current_line.strip())
                            current_line = ""
            # Выводим оставшийся текст
            if current_line:
                progress.write(current_line.strip())
            return {'content': content, 'usage': {}}
        else:
            # В обычном режиме без streaming
            response = client.chat.completions.create(
                model=model or LOCAL_LLM_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_tokens=LOCAL_LLM_MAX_TOKENS,
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