# === Импорты ===
import csv
import json
import os
import re
import sys
import argparse
import time
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from colorama import init, Fore, Style
from prompts_and_schemas import (
    RESPONSE_SCHEMA, SINGLE_PROMPT, MULTI_PROMPTS, REFLEXION_PROMPT,
    FINAL_SALES_SCHEMA, FINAL_SALES_PROMPT, MBTI_PROFILE_SCHEMA, MBTI_PROFILE_PROMPT, SINGLE_PROMPT, REFLECTION_SCHEMA
)
from llm_router import global_metrics, sales_generate_completion, select_best_analysis_pages
from scraper import crawl_and_analyze_url, get_site_links, get_page_content
from utils import extract_content, robust_json_loads, clean_input, is_russian, parse_final_sales_response, normalize_llm_response, is_valid_url, parse_reflection_response, extract_links_from_text, to_absolute

# === ВАЖНО: Логика флагов local_only и force_remote ===
# local_only (-L): использовать только локальную модель для всех запросов, КРОМЕ тех, где явно указан force_remote.
# force_remote (-F): использовать облако (OpenRouter) только для ВАЖНЫХ запросов (финальный анализ, reflection и т.п.), даже если установлен -L.
# Если оба флага не указаны — авто-режим (сначала облако, если доступно, иначе локально).
# force_remote НЕ отменяет полностью local_only, а временно его переопределяет для конкретных запросов.
# ВСЕГДА передавайте оба параметра в важные LLM-запросы: local_only=local_only, force_remote=force_remote

# === Константы и настройки ===
DEFAULT_URL = 'https://frappecrm.ru/ru'
DEFAULT_LIMIT = 3
DEFAULT_MODE = 'single'

# === Анализ одной страницы (single prompt) ===
def analyze_page_single(content, local_only=True):
    """Анализирует одну страницу с помощью одного промпта."""
    prompt_content = f"Текст страницы:\n{content}"
    analysis = sales_generate_completion(
        role="website deep org psych analyzer",
        task=SINGLE_PROMPT,
        content=prompt_content,
        response_format=RESPONSE_SCHEMA,
        local_only=local_only
    )
    return extract_content(analysis)

# === Анализ одной страницы (multi-step) ===
def analyze_page_multi(content, local_only=True):
    """Анализирует одну страницу поэтапно (multi-step prompts)."""
    context = content
    results = []
    for i, prompt in enumerate(MULTI_PROMPTS):
        input_text = f"Текст страницы:\n{context}"
        prompt_with_prev = prompt
        if i > 0:
            prev = '\n'.join([f"Этап {j+1}: {r}" for j, r in enumerate(results)])
            prompt_with_prev = prompt + f"\n\nРезультаты предыдущих этапов:\n{prev}"
        analysis = sales_generate_completion(
            role=f"website org psych analyzer step {i+1}",
            task=prompt_with_prev,
            content=input_text,
            response_format=RESPONSE_SCHEMA,
            local_only=local_only
        )
        results.append(extract_content(analysis))
    return {'steps': results, 'summary': results[-1] if results else ''}

# === Дозаполнение результата, если LLM оборвал текст ===
def complete_result_if_needed(parsed_json, context, role, task, temperature=0.5, max_attempts=3, local_only=False):
    """Дозаполняет результат, если LLM оборвал текст (нет точки в конце)."""
    result = parsed_json.get('result')
    attempts = 0
    while isinstance(result, str) and not result.strip().endswith('.') and attempts < max_attempts:
        continue_prompt = (
            f"{context}\n\n"
            f"Начало текста result (не повторяй его, только продолжи):\n{result}\n"
            f"Продолжи строго с того места, где оборвался текст в ключе result. Верни только оставшуюся часть текста для этого поля, чтобы завершить мысль."
        )
        continuation = sales_generate_completion(
            role=role,
            task=task,
            content=continue_prompt,
            temperature=temperature,
            local_only=local_only
        )
        continuation_text = extract_content(continuation)
        result = result + continuation_text.strip()
        parsed_json['result'] = result
        attempts += 1
    return parsed_json

# === Генерация MBTI-профиля ===
def generate_mbti_profile(summary_text, local_only=True, force_remote=False):
    """Генерирует MBTI-профиль на основе summary текста."""
    messages = [
        {"role": "user", "content": MBTI_PROFILE_PROMPT},
        {"role": "user", "content": summary_text}
    ]
    response = sales_generate_completion(
        messages=messages,
        response_format=MBTI_PROFILE_SCHEMA,
        local_only=local_only,
        force_remote=force_remote
    )
    return extract_content(response)

# === Подсчёт количества LLM-запросов ===
def calculate_requests_needed(num_pages, mode):
    """Возвращает количество LLM-запросов для анализа сайта."""
    if mode == 'multi':
        per_page = len(MULTI_PROMPTS)
    else:
        per_page = 1
    return num_pages * per_page + 3  # +3: MBTI, рефлексия, sales-отчет

# === Интегрирующий анализ (reflection) ===
def run_reflection_analysis(page_summaries, local_only=True, force_remote=False, debug=False):
    """Выполняет интегрирующий анализ (reflection) по результатам страниц."""
    # Собираем summary по всем страницам
    texts = []
    for i, summary in enumerate(page_summaries, 1):
        res = summary.get('result', summary)
        if isinstance(res, dict) and 'result' in res:
            texts.append(res['result'])
        else:
            texts.append(str(res))
    context = '\n\n'.join(texts)
    messages = [
        {"role": "user", "content": REFLEXION_PROMPT},
        {"role": "user", "content": context}
    ]
    response = sales_generate_completion(
        messages=messages,
        response_format=REFLECTION_SCHEMA,
        local_only=local_only,
        force_remote=force_remote
    )
    if debug:
        print("[DEBUG: RAW REFLECTION RESPONSE]", response)
    return extract_content(response)

# === Основной пайплайн ===
def main():
    """Основная точка входа: парсинг аргументов, анализ сайта, вывод результатов."""
    init(autoreset=True)
    start_time = time.time()

    # --- Аргументы командной строки ---
    parser = argparse.ArgumentParser(
        description="Анализ внутренних ценностей и болей компании по сайту. Пример: python sales.py -u example.com -l 5 -m multi"
    )
    parser.add_argument('-u', '--url', type=str, help='URL сайта для анализа (можно с https:// или без). По умолчанию: https://frappecrm.ru/ru', default=DEFAULT_URL)
    parser.add_argument('-l', '--limit', type=int, help='Максимальное количество страниц для анализа (по умолчанию 3)', default=DEFAULT_LIMIT)
    parser.add_argument('-m', '--mode', type=str, choices=['single', 'multi'], help='Режим анализа: single — один многоступенчатый промпт на страницу; multi — последовательные этапы анализа для каждой страницы. По умолчанию: single', default=DEFAULT_MODE)
    parser.add_argument('-L', '--local-only', action='store_true', help='Использовать только локальную модель, игнорировать OpenRouter')
    parser.add_argument('-F', '--force-remote', action='store_true', help='Использовать OpenRouter в финальном промпте')
    parser.add_argument('-v', '--verbose', dest='debug', action='store_true', help='Включить подробный вывод и сохранение промежуточных данных (debug/verbose режим)')
    parser.add_argument('-C', '--critic', action='store_true', help='Только интегрирующий анализ (reflection), без профиля и sales-анализа')
    parser.add_argument('-no', '--no-smart-pages', action='store_true', help='Отключить умный выбор страниц для анализа (по умолчанию включён)')
    args = parser.parse_args()

    # --- Подготовка параметров ---
    raw_url = clean_input(args.url)
    url = raw_url if raw_url.startswith("http://") or raw_url.startswith("https://") else f"https://{raw_url}"
    limit = args.limit if args.limit > 0 else DEFAULT_LIMIT
    mode = args.mode
    local_only = args.local_only
    force_remote = args.force_remote
    debug = args.debug
    critic = args.critic
    smart_pages = not args.no_smart_pages

    print(Fore.YELLOW + f"\nИспользуемый URL: {url}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Лимит страниц: {limit}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Режим анализа: {mode}\n" + Style.RESET_ALL)
    if local_only and not force_remote:
        print(Fore.YELLOW + Style.BRIGHT + "Используется только локальная модель" + Style.RESET_ALL)
    elif force_remote and local_only:
        print(Fore.YELLOW + Style.BRIGHT + "Используется OpenRouter в финальном промпте" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + Style.BRIGHT + "Используется OpenRouter" + Style.RESET_ALL)

    # --- Умный выбор страниц для анализа ---
    selected_pages = None
    if smart_pages:
        all_links = get_site_links(url)
        if not all_links:
            main_content = get_page_content(url)
            all_links = extract_links_from_text(main_content, url)
        pages_to_parse = [url]
        if all_links:
            best_links = select_best_analysis_pages(all_links, limit-1, local_only=local_only)
            for link in best_links:
                abs_link = to_absolute(link, url)
                if abs_link != url and abs_link not in pages_to_parse:
                    pages_to_parse.append(abs_link)
        selected_pages = pages_to_parse[:limit]

    # --- Скрапинг и анализ сайта ---
    result = crawl_and_analyze_url(
        url, limit=limit, mode=mode, local_only=local_only,
        analyze_page_single=analyze_page_single,
        analyze_page_multi=analyze_page_multi,
        complete_result_if_needed=complete_result_if_needed,
        calculate_requests_needed=calculate_requests_needed,
        normalize_llm_response=normalize_llm_response,
        debug=debug,
        selected_pages=selected_pages
    )
    page_summaries = result.get('page_summaries', [])

    # --- Красивый вывод reflection (интегрирующий анализ) ---
    if critic:
        reflection_struct = run_reflection_analysis(page_summaries, local_only=local_only, force_remote=force_remote, debug=debug)
        if reflection_struct:
            print(Fore.CYAN + Style.BRIGHT + "\nИнтегрированный анализ (рефлексия):" + Style.RESET_ALL)
            parsed_ref = parse_reflection_response(reflection_struct)
            if parsed_ref:
                for k, v in parsed_ref.items():
                    print(Fore.YELLOW + f"{k}:" + Style.RESET_ALL)
                    if k == "Анализ по целям" and isinstance(v, list):
                        for i, item in enumerate(v, 1):
                            print(f"  {i}. Цель: {item.get('Цель', '')}")
                            print(f"     Результат: {item.get('Результат', '')}")
                    elif isinstance(v, list):
                        for item in v:
                            print("  -", item)
                    else:
                        print(" ", v)
                    print()
            else:
                print(reflection_struct)
        else:
            print(Fore.RED + "[!] Не удалось получить интегрирующий анализ." + Style.RESET_ALL)
        # Метрики и return
        end_time = time.time()
        elapsed = end_time - start_time
        print(Fore.GREEN + Style.BRIGHT + f"\n=== Метрики выполнения ===" + Style.RESET_ALL)
        print(f"  Общее время выполнения: {elapsed:.2f} сек")
        print(f"  Токены (облако): prompt={global_metrics['cloud_prompt_tokens']}, completion={global_metrics['cloud_completion_tokens']}")
        print(f"  Токены (локально): prompt={global_metrics['local_prompt_tokens']}, completion={global_metrics['local_completion_tokens']}")
        return

    # --- Итоговый результат (result) ---
    # Это агрегированный текстовый результат по всем страницам (до интеграции и финального sales-анализа).
    # Выводится только в debug-режиме для ручной проверки пайплайна.
    res = result.get("result", "-")
    if debug and res and res != "-" and res != [] and res != ["-"]:
        print(Fore.CYAN + Style.BRIGHT + "Итоговый результат:" + Style.RESET_ALL)
        try:
            parsed = json.loads(res)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    print(f"  {k.capitalize()}: {v}")
            else:
                print("  " + str(parsed))
        except Exception:
            print("  " + str(res))

    # --- Финальный sales-анализ (структурировано) ---
    print(Fore.CYAN + Style.BRIGHT + "\nАнализ потребностей и рекомендации по продажам (структурировано):" + Style.RESET_ALL)
    mbti_profile_raw = generate_mbti_profile(str(res), local_only=local_only, force_remote=force_remote)
    mbti_profile = mbti_profile_raw["mbti_profile"] if isinstance(mbti_profile_raw, dict) and "mbti_profile" in mbti_profile_raw else str(mbti_profile_raw)
    content = str(res) + "\nMBTI-профиль ЛПР: " + mbti_profile
    final_sales_response = sales_generate_completion(
        role="corporate sales analyzer",
        task=FINAL_SALES_PROMPT,
        content=content,
        response_format=FINAL_SALES_SCHEMA,
        is_final=True,
        temperature=1,
        local_only=not force_remote and local_only
    )
    final_sales_content = extract_content(final_sales_response)
    if not is_russian(final_sales_content):
        translate_prompt = (
            "Переведи этот текст на русский язык, сохрани стиль и структуру ответа. Не добавляй ничего лишнего.\n" + final_sales_content
        )
        translated = sales_generate_completion(
            role="corporate sales analyzer",
            task="Ты — профессиональный переводчик. Переведи на русский язык, сохрани стиль и структуру.",
            content=translate_prompt,
            response_format=FINAL_SALES_SCHEMA,
            local_only=not force_remote and local_only
        )
        final_sales_content = extract_content(translated)
    parsed_final = parse_final_sales_response(final_sales_content)
    if parsed_final:
        for k, v in parsed_final.items():
            print(Fore.YELLOW + f"\n{k}:" + Style.RESET_ALL)
            if isinstance(v, list):
                for item in v:
                    print(f"  - {item}")
            elif v:
                print(f"  {v}")
    else:
        print(Fore.RED + "[!] Не удалось распарсить финальный анализ. Сырой ответ:" + Style.RESET_ALL)
        print(final_sales_content)
    print(Fore.BLUE + "\n" + "─"*40 + "\n" + Style.RESET_ALL)

    # --- Метрики ---
    end_time = time.time()
    elapsed = end_time - start_time
    print(Fore.GREEN + Style.BRIGHT + f"\n=== Метрики выполнения ===" + Style.RESET_ALL)
    print(f"  Общее время выполнения: {elapsed:.2f} сек")
    print(f"  Токены (облако): prompt={global_metrics['cloud_prompt_tokens']}, completion={global_metrics['cloud_completion_tokens']}")
    print(f"  Токены (локально): prompt={global_metrics['local_prompt_tokens']}, completion={global_metrics['local_completion_tokens']}")

if __name__ == "__main__":
    main()