import os
from tqdm import tqdm
from colorama import Fore, Style
import time
import openrouter
import local_llm
from prompts_and_schemas import ANALYSIS_PAGE_SELECTION_SCHEMA, ANALYSIS_PAGE_SELECTION_PROMPT

# === Метрики ===
global_metrics = {
    'cloud_prompt_tokens': 0,
    'cloud_completion_tokens': 0,
    'local_prompt_tokens': 0,
    'local_completion_tokens': 0,
}

def accumulate_usage(result, is_cloud):
    if isinstance(result, dict) and 'usage' in result and result['usage']:
        usage = result['usage']
        if not isinstance(usage, dict):
            if hasattr(usage, 'model_dump'):
                usage = usage.model_dump()
            elif hasattr(usage, 'dict'):
                usage = usage.dict()
            else:
                try:
                    usage = dict(usage)
                except Exception:
                    usage = {}
        if is_cloud:
            global_metrics['cloud_prompt_tokens'] += usage.get('prompt_tokens', 0)
            global_metrics['cloud_completion_tokens'] += usage.get('completion_tokens', 0)
        else:
            global_metrics['local_prompt_tokens'] += usage.get('prompt_tokens', 0)
            global_metrics['local_completion_tokens'] += usage.get('completion_tokens', 0)

def sales_generate_completion(
    role=None,
    task=None,
    content=None,
    temperature=0.4,
    response_format=None,
    is_final=False,
    local_only=False,
    force_remote=False,
    model=None,
    messages=None,
    debug=False
):
    """
    Универсальный роутер для sales pipeline: сначала openrouter, если доступен и не запрещён параметрами, иначе fallback на локальную модель.
    Все детали провайдеров инкапсулированы в их модулях.
    Подсчёт usage централизован здесь.
    Если messages передан — используется напрямую (для кеширования и спец. случаев).
    """
    if messages is None:
        messages = [
            {"role": "user", "content": content},
            {"role": "user", "content": f"{task}"},
            {"role": "system", "content": f"You are a {role}."}
        ]
    # Только локально
    if local_only and not force_remote:
        result = local_llm.llm_completion(messages, temperature=temperature, response_format=response_format, model=model, debug=debug)
        accumulate_usage(result, is_cloud=False)
        return result
    # Только openrouter (но fallback есть)
    if force_remote and openrouter.is_available():
        try:
            result = openrouter.llm_completion(messages, temperature=temperature, response_format=response_format, model=model, debug=debug)
            accumulate_usage(result, is_cloud=True)
            return result
        except Exception as e:
            print(f"[!] Ошибка OpenRouter, fallback на локальную модель: {e}")
            result = local_llm.llm_completion(messages, temperature=temperature, response_format=response_format, model=model, debug=debug)
            accumulate_usage(result, is_cloud=False)
            return result
    # Автоматический выбор
    if openrouter.is_available():
        try:
            result = openrouter.llm_completion(messages, temperature=temperature, response_format=response_format, model=model, debug=debug)
            accumulate_usage(result, is_cloud=True)
            return result
        except Exception as e:
            print(f"[!] Ошибка OpenRouter, fallback на локальную модель: {e}")
    result = local_llm.llm_completion(messages, temperature=temperature, response_format=response_format, model=model, debug=debug)
    accumulate_usage(result, is_cloud=False)
    return result 

def select_best_analysis_pages(links, limit, local_only=True, force_remote=False):
    """
    Выбирает наиболее релевантные для sales-анализа страницы из списка ссылок с помощью LLM.
    Возвращает список URL (может быть меньше лимита или пустой).
    """
    prompt = ANALYSIS_PAGE_SELECTION_PROMPT.format(limit=limit, links="\n".join(links))
    messages = [
        {"role": "system", "content": "Ты — эксперт по анализу сайтов и корпоративной культуры."},
        {"role": "user", "content": prompt}
    ]
    response = sales_generate_completion(
        messages=messages,
        response_format=ANALYSIS_PAGE_SELECTION_SCHEMA,
        local_only=not force_remote and local_only
    )
    import json
    try:
        if isinstance(response, dict):
            urls = response.get("urls", [])
        else:
            data = json.loads(response)
            urls = data.get("urls", [])
        if isinstance(urls, list):
            return urls[:limit]
    except Exception:
        # fallback: парсим строки по одной в каждой строке
        return [line.strip() for line in str(response).splitlines() if line.strip().startswith('http')][:limit]
    return [] 