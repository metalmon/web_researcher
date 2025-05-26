import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from openai import OpenAI
import traceback
import requests
import time

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# === Rate limit tracking ===
_openrouter_rate_limit = None  # {'requests': int, 'interval': '10s'}
_openrouter_last_reset = 0
_openrouter_request_count = 0

def check_openrouter_limits(debug=False):
    global _openrouter_rate_limit, _openrouter_last_reset, _openrouter_request_count
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    try:
        resp = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            _openrouter_rate_limit = data.get("rate_limit")
            _openrouter_last_reset = time.time()
            _openrouter_request_count = 0
            if debug:
                print(f"[OpenRouter] Rate limit info: {_openrouter_rate_limit}")
            return _openrouter_rate_limit
        else:
            if debug:
                print(f"[OpenRouter] Не удалось получить лимиты: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        if debug:
            print(f"[OpenRouter] Ошибка при получении лимитов: {e}")
        return None

def _respect_openrouter_rate_limit(debug=False):
    global _openrouter_rate_limit, _openrouter_last_reset, _openrouter_request_count
    rate = _openrouter_rate_limit
    if not rate:
        return  # Нет данных — не ограничиваем
    req_limit = rate.get('requests')
    interval = rate.get('interval', '10s')
    seconds = 10
    if interval.endswith('s'):
        seconds = int(interval[:-1])
    elif interval.endswith('m'):
        seconds = int(interval[:-1]) * 60
    elif interval.endswith('h'):
        seconds = int(interval[:-1]) * 3600
    now = time.time()
    if now - _openrouter_last_reset > seconds:
        _openrouter_last_reset = now
        _openrouter_request_count = 0
    if _openrouter_request_count >= req_limit:
        wait = _openrouter_last_reset + seconds - now
        if wait > 0:
            if debug:
                print(f"[OpenRouter] Rate limit: жду {wait:.1f} сек...")
            time.sleep(wait)
        _openrouter_last_reset = time.time()
        _openrouter_request_count = 0
    _openrouter_request_count += 1

def is_available():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if OPENROUTER_MODEL not in model_ids:
            print(f"[!] Модель {OPENROUTER_MODEL} не найдена среди доступных моделей OpenRouter.")
            return False
        return True
    except Exception:
        return False

def llm_completion(messages, temperature=0.4, response_format=None, model=None, debug=False):
    from openai import OpenAIError
    global _openrouter_rate_limit
    kwargs = {}
    if response_format:
        kwargs['response_format'] = response_format
    # Получаем лимиты только если их ещё нет
    if _openrouter_rate_limit is None:
        check_openrouter_limits(debug=debug)
    max_retries = 3
    for attempt in range(max_retries):
        _respect_openrouter_rate_limit(debug=debug)
        try:
            response = client.chat.completions.create(
                model=model or OPENROUTER_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_tokens=2048,
                **kwargs
            )
            if debug:
                print("[OpenRouter RAW RESPONSE]", response)
            usage = getattr(response, 'usage', None)
            if usage is None and isinstance(response, dict):
                usage = response.get('usage', None)
            if usage is None:
                usage = {}
            return {'content': response.choices[0].message.content, 'usage': usage}
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate limit" in err_str.lower() or "402" in err_str) and attempt < max_retries-1:
                # Перезапрашиваем лимиты и пробуем снова
                check_openrouter_limits(debug=debug)
                if debug:
                    print(f"[OpenRouter] Rate limit/credit error, retrying... ({attempt+1})")
                time.sleep(1)
                continue
            print(f"[OpenRouter ERROR] {e}")
            traceback.print_exc()
            raise