# === Импорты ===
import json
import argparse
import time
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from colorama import init, Fore, Style
from prompts_and_schemas import (
    RESPONSE_SCHEMA, SINGLE_PROMPT, MULTI_PROMPTS, REFLEXION_PROMPT,
    FINAL_SALES_SCHEMA, FINAL_SALES_PROMPT, MBTI_PROFILE_SCHEMA, MBTI_PROFILE_PROMPT, SINGLE_PROMPT, REFLECTION_SCHEMA
)
from llm_router import global_metrics, sales_generate_completion, select_best_analysis_pages
from scraper import crawl_and_analyze_url, get_site_links, get_page_content
from utils import extract_content, clean_input, is_russian, parse_final_sales_response, normalize_llm_response, is_valid_url, parse_reflection_response, extract_links_from_text, to_absolute
from progress import progress
from outreach import score_lead, generate_outreach_email, save_email_to_file, format_email_for_sending
from contact import extract_contacts_from_site

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
LEAD_SCORE_THRESHOLD = float(os.getenv('LEAD_SCORE_THRESHOLD', '0.5'))  # Порог релевантности лида из env или 0.5 по умолчанию

# === Анализ одной страницы (single prompt) ===
def analyze_page_single(content, local_only=True, debug=False):
    """Анализирует одну страницу с помощью одного промпта."""
    prompt_content = f"Текст страницы:\n{content}"
    analysis = sales_generate_completion(
        role="website deep org psych analyzer",
        task=SINGLE_PROMPT,
        content=prompt_content,
        response_format=RESPONSE_SCHEMA,
        local_only=local_only,
        debug=debug
    )
    progress.update(1)  # Обновляем прогресс после анализа
    return extract_content(analysis)

# === Анализ одной страницы (multi-step) ===
def analyze_page_multi(content, local_only=True, debug=False):
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
            local_only=local_only,
            debug=debug
        )
        results.append(extract_content(analysis))
        progress.update(1)  # Обновляем прогресс после каждого этапа анализа
    return {'steps': results, 'summary': results[-1] if results else ''}

# === Дозаполнение результата, если LLM оборвал текст ===
def complete_result_if_needed(parsed_json, context, role, task, temperature=0.5, max_attempts=3, local_only=False, debug=False):
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
            local_only=local_only,
            debug=debug
        )
        continuation_text = extract_content(continuation)
        result = result + continuation_text.strip()
        parsed_json['result'] = result
        attempts += 1
    return parsed_json

# === Генерация MBTI-профиля ===
def generate_mbti_profile(summary_text, local_only=True, force_remote=False, debug=False):
    """Генерирует MBTI-профиль на основе summary текста."""
    messages = [
        {"role": "user", "content": MBTI_PROFILE_PROMPT},
        {"role": "user", "content": summary_text}
    ]
    response = sales_generate_completion(
        messages=messages,
        response_format=MBTI_PROFILE_SCHEMA,
        local_only=local_only,
        force_remote=force_remote,
        debug=debug
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
        force_remote=force_remote,
        debug=debug
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
    parser.add_argument('-e', '--email', action='store_true', help='Сгенерировать письмо для первого контакта')
    parser.add_argument('-o', '--output', type=str, help='Путь для сохранения сгенерированного письма', default='outreach_email.txt')
    parser.add_argument('-c', '--contacts', action='store_true', help='Извлечь контакты и реквизиты компании')
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

    progress.write(Fore.YELLOW + f"\nИспользуемый URL: {url}" + Style.RESET_ALL)
    progress.write(Fore.YELLOW + f"Лимит страниц: {limit}" + Style.RESET_ALL)
    progress.write(Fore.YELLOW + f"Режим анализа: {mode}\n" + Style.RESET_ALL)
    if local_only and not force_remote:
        progress.write(Fore.YELLOW + Style.BRIGHT + "Используется только локальная модель" + Style.RESET_ALL)
    elif force_remote and local_only:
        progress.write(Fore.YELLOW + Style.BRIGHT + "Используется OpenRouter в финальном промпте" + Style.RESET_ALL)
    else:
        progress.write(Fore.YELLOW + Style.BRIGHT + "Используется OpenRouter" + Style.RESET_ALL)
    progress.write("")  # Пустая строка для разделения

    # --- Умный выбор страниц для анализа ---
    selected_pages = None
    
    # Инициализируем прогресс-бар с базовым количеством шагов
    base_steps = 1  # Учитываем первый шаг, который уже начался
    if smart_pages:
        base_steps += 2  # Получение карты + выбор страниц
    base_steps += 1  # Получение данных
    if mode == 'multi':
        base_steps += limit * len(MULTI_PROMPTS)  # Анализ страниц в multi-режиме
    else:
        base_steps += limit  # Анализ страниц в single-режиме
    if not critic:
        base_steps += 3  # MBTI + Финальный анализ + Перевод (если нужно)
    else:
        base_steps += 1  # Только reflection
    if args.email:
        base_steps += 3  # Оценка лида + Генерация письма + Генерация заголовка

    # Инициализируем прогресс-бар
    progress.update_stage("Анализ сайта", base_steps)

    if smart_pages:
        # 1. Получение карты сайта
        all_links = get_site_links(url)
        
        # Валидация количества ссылок
        if len(all_links) > limit - 1:  # -1 потому что главная страница уже включена
            progress.pbar.write(Fore.YELLOW + f"\nПолучено {len(all_links)} ссылок, что больше запрошенного лимита {limit-1}." + Style.RESET_ALL)
            progress.pbar.write(Fore.YELLOW + "Ограничиваем количество ссылок до запрошенного лимита." + Style.RESET_ALL)
            all_links = all_links[:limit-1]  # Оставляем только нужное количество ссылок
        
        progress.update(1, "Получение карты сайта")
        
        # 2. Выбор лучших страниц
        pages_to_parse = [url]
        if all_links:
            best_links = select_best_analysis_pages(all_links, limit-1, local_only=local_only)
            for link in best_links:
                abs_link = to_absolute(link, url)
                if abs_link != url and abs_link not in pages_to_parse:
                    pages_to_parse.append(abs_link)
        selected_pages = pages_to_parse[:limit]
        progress.update(1, "Выбор страниц")

        # Пересчитываем общее количество шагов на основе выбранных страниц
        total_steps = 1  # Учитываем первый шаг, который уже начался
        total_steps += 2  # Получение карты + выбор страниц (уже выполнено)
        total_steps += 1  # Получение данных
        if mode == 'multi':
            total_steps += len(selected_pages) * len(MULTI_PROMPTS)  # Анализ страниц в multi-режиме
        else:
            total_steps += len(selected_pages)  # Анализ страниц в single-режиме
        if not critic:
            total_steps += 3  # MBTI + Финальный анализ + Перевод (если нужно)
        else:
            total_steps += 1  # Только reflection
        if args.email:
            total_steps += 3  # Оценка лида + Генерация письма + Генерация заголовка

        # Обновляем прогресс-бар с правильным количеством шагов
        progress.update_stage("Анализ сайта", total_steps)

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
    progress.update(1, "Получение данных")

    # --- Красивый вывод reflection (интегрирующий анализ) ---
    if critic:
        reflection_struct = run_reflection_analysis(page_summaries, local_only=local_only, force_remote=force_remote, debug=debug)
        progress.update(1, "Интегрирующий анализ")
        if reflection_struct:
            progress.write(Fore.CYAN + Style.BRIGHT + "\nИнтегрированный анализ (рефлексия):" + Style.RESET_ALL)
            parsed_ref = parse_reflection_response(reflection_struct)
            if parsed_ref:
                for k, v in parsed_ref.items():
                    progress.write(Fore.YELLOW + f"{k}:" + Style.RESET_ALL)
                    if k == "Анализ по целям" and isinstance(v, list):
                        for i, item in enumerate(v, 1):
                            progress.write(f"  {i}. Цель: {item.get('Цель', '')}")
                            progress.write(f"     Результат: {item.get('Результат', '')}")
                    elif isinstance(v, list):
                        for item in v:
                            progress.write("  - " + item)
                    else:
                        progress.write("  " + v)
                    progress.write("")
            else:
                progress.write(reflection_struct)
        else:
            progress.write(Fore.RED + "[!] Не удалось получить интегрирующий анализ." + Style.RESET_ALL)
        # Метрики и return
        end_time = time.time()
        elapsed = end_time - start_time
        progress.write(Fore.GREEN + Style.BRIGHT + f"\n=== Метрики выполнения ===" + Style.RESET_ALL)
        progress.write(f"  Общее время выполнения: {elapsed:.2f} сек")
        progress.write(f"  Токены (облако): prompt={global_metrics['cloud_prompt_tokens']}, completion={global_metrics['cloud_completion_tokens']}")
        progress.write(f"  Токены (локально): prompt={global_metrics['local_prompt_tokens']}, completion={global_metrics['local_completion_tokens']}")
        return

    # --- Итоговый результат (result) ---
    res = result.get("result", "-")
    if debug and res and res != "-" and res != [] and res != ["-"]:
        progress.write(Fore.CYAN + Style.BRIGHT + "Итоговый результат:" + Style.RESET_ALL)
        try:
            parsed = json.loads(res)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    progress.write(f"  {k.capitalize()}: {v}")
            else:
                progress.write("  " + str(parsed))
        except Exception:
            progress.write("  " + str(res))

    # --- Финальный sales-анализ (структурировано) ---
    mbti_profile_raw = generate_mbti_profile(str(res), local_only=local_only, force_remote=force_remote, debug=debug)
    mbti_profile = mbti_profile_raw["mbti_profile"] if isinstance(mbti_profile_raw, dict) and "mbti_profile" in mbti_profile_raw else str(mbti_profile_raw)
    progress.update(1, "Профилирование")

    content = str(res) + "\nMBTI-профиль ЛПР: " + mbti_profile
    final_sales_response = sales_generate_completion(
        role="corporate sales analyzer",
        task=FINAL_SALES_PROMPT,
        content=content,
        response_format=FINAL_SALES_SCHEMA,
        is_final=True,
        temperature=1,
        local_only=not force_remote and local_only,
        debug=debug
    )
    final_sales_content = extract_content(final_sales_response)
    
    # Проверяем и дозаполняем результат, если нужно
    try:
        parsed_json = json.loads(final_sales_content)
        completed_json = complete_result_if_needed(
            parsed_json,
            content,
            role="corporate sales analyzer",
            task=FINAL_SALES_PROMPT,
            temperature=1,
            local_only=not force_remote and local_only,
            debug=debug
        )
        final_sales_content = json.dumps(completed_json, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        # Если не удалось распарсить JSON, оставляем как есть
        pass

    if not is_russian(final_sales_content):
        translate_prompt = (
            "Переведи этот текст на русский язык, сохрани стиль и структуру ответа. Не добавляй ничего лишнего.\n" + final_sales_content
        )
        translated = sales_generate_completion(
            role="corporate sales analyzer",
            task="Ты — профессиональный переводчик. Переведи на русский язык, сохрани стиль и структуру.",
            content=translate_prompt,
            response_format=FINAL_SALES_SCHEMA,
            local_only=not force_remote and local_only,
            debug=debug
        )
        final_sales_content = extract_content(translated)
    progress.update(1, "Финальный анализ")

    parsed_final = parse_final_sales_response(final_sales_content)
    if parsed_final:
        for k, v in parsed_final.items():
            progress.write(Fore.YELLOW + f"\n{k}:" + Style.RESET_ALL)
            if isinstance(v, list):
                for item in v:
                    progress.write(f"  - {item}")
            elif v:
                progress.write(f"  {v}")
    else:
        progress.write(Fore.RED + "[!] Не удалось распарсить финальный анализ. Сырой ответ:" + Style.RESET_ALL)
        progress.write(final_sales_content)
    progress.write(Fore.BLUE + "\n" + "─"*40 + "\n" + Style.RESET_ALL)

    # --- Генерация письма для первого контакта ---
    if args.email:
        progress.update(1, "Оценка релевантности лида")
        score, explanation, communication_scheme = score_lead(parsed_final, local_only=local_only, force_remote=force_remote, debug=debug)
        
        # Выводим оценку лида
        progress.write(Fore.CYAN + Style.BRIGHT + "\nОценка релевантности лида:" + Style.RESET_ALL)
        progress.write(Fore.YELLOW + f"\nОбщая оценка: {score:.2f}" + Style.RESET_ALL)
        if explanation:
            # Маппинг для красивого вывода
            factor_names = {
                "company_fit": "Соответствие компании",
                "budget_potential": "Потенциальный бюджет",
                "urgency": "Срочность потребности",
                "decision_maker": "ЛПР и готовность к изменениям"
            }
            
            for factor, details in explanation.items():
                if factor != "overall_recommendation":
                    display_name = factor_names.get(factor, factor)
                    progress.write(Fore.YELLOW + f"\n{display_name}:" + Style.RESET_ALL)
                    progress.write(f"  Оценка: {details.get('score', 0):.2f}")
                    progress.write(f"  Причина: {details.get('reason', '')}")
            
            if "overall_recommendation" in explanation:
                progress.write(Fore.YELLOW + "\nРекомендация:" + Style.RESET_ALL)
                progress.write(f"  {explanation['overall_recommendation']}")
        
        # Выводим схему коммуникации
        if communication_scheme:
            progress.write(Fore.YELLOW + "\nРекомендуемая схема коммуникации:" + Style.RESET_ALL)
            progress.write(f"  Схема: {communication_scheme.get('scheme', '')}")
            progress.write(f"  Причина: {communication_scheme.get('reason', '')}")
            progress.write(f"  Ключевые фокусы: {communication_scheme.get('key_focus', '')}")
        
        # Генерируем письмо только если оценка выше порога
        if score >= LEAD_SCORE_THRESHOLD:  # Используем порог из env
            progress.update(1, "Составление письма")
            email_content = generate_outreach_email(parsed_final, local_only=local_only, force_remote=force_remote, debug=debug)
            
            # Выводим письмо в терминал
            progress.write(Fore.CYAN + Style.BRIGHT + "\nПисьмо для первого контакта:" + Style.RESET_ALL)
            if isinstance(email_content, dict):
                progress.write(Fore.YELLOW + "\nТема:" + Style.RESET_ALL)
                progress.write(email_content.get('subject', ''))
                progress.write(Fore.YELLOW + "\nТекст письма:" + Style.RESET_ALL)
                formatted_email = format_email_for_sending(email_content)
                progress.write(formatted_email)
            else:
                # Если получили строку, пытаемся распарсить как JSON
                try:
                    email_content = json.loads(email_content)
                    progress.write(Fore.YELLOW + "\nТема:" + Style.RESET_ALL)
                    progress.write(email_content.get('subject', ''))
                    progress.write(Fore.YELLOW + "\nТекст письма:" + Style.RESET_ALL)
                    formatted_email = format_email_for_sending(email_content)
                    progress.write(formatted_email)
                except json.JSONDecodeError:
                    progress.write(email_content)
            
            # Сохраняем в файл только если указан output
            if args.output and args.output != 'outreach_email.txt':  # Не сохраняем если используется значение по умолчанию
                save_email_to_file(email_content, args.output)
                progress.write(Fore.GREEN + f"\nПисьмо также сохранено в файл: {args.output}" + Style.RESET_ALL)
        else:
            progress.write(Fore.RED + f"\nЛид не прошел порог релевантности ({LEAD_SCORE_THRESHOLD}). Письмо не будет сгенерировано." + Style.RESET_ALL)

    # --- Извлечение контактов ---
    if args.contacts:
        progress.write(Fore.CYAN + Style.BRIGHT + "\nИзвлечение контактов и реквизитов:" + Style.RESET_ALL)
        contacts_data = extract_contacts_from_site(url, limit=limit, local_only=local_only, debug=debug)
        
        # Выводим реквизиты компании
        company_details = contacts_data.get("company_details", {})
        if company_details:
            progress.write(Fore.YELLOW + "\nРеквизиты компании:" + Style.RESET_ALL)
            if company_details.get("legal_name"):
                progress.write(f"  Юридическое название: {company_details['legal_name']}")
            if company_details.get("inn"):
                progress.write(f"  ИНН: {company_details['inn']}")
            if company_details.get("ogrn"):
                progress.write(f"  ОГРН: {company_details['ogrn']}")
            if company_details.get("address", {}).get("full"):
                progress.write(f"  Адрес: {company_details['address']['full']}")
        
        # Выводим контакты
        contacts = contacts_data.get("contacts", [])
        if contacts:
            progress.write(Fore.YELLOW + "\nКонтактная информация:" + Style.RESET_ALL)
            for contact in contacts:
                contact_type = contact.get("type", "").upper()
                value = contact.get("value", "")
                context = contact.get("context", "")
                confidence = contact.get("confidence", 0)
                is_personal = contact.get("is_personal", False)
                
                contact_info = f"  {contact_type}: {value}"
                if context:
                    contact_info += f" (контекст: {context})"
                if confidence:
                    contact_info += f" [уверенность: {confidence:.2f}]"
                if is_personal:
                    contact_info += " [личный контакт]"
                progress.write(contact_info)
        
        # Выводим анализ контактов
        contact_analysis = contacts_data.get("contact_analysis", {}).get("contact_profile", {})
        if contact_analysis:
            progress.write(Fore.YELLOW + "\nАнализ контактов:" + Style.RESET_ALL)
            if "contact_types" in contact_analysis:
                progress.write(f"  Типы контактов: {', '.join(contact_analysis['contact_types'])}")
            if "primary_contact" in contact_analysis:
                progress.write(f"  Основной контакт: {contact_analysis['primary_contact']}")
            if "contact_availability" in contact_analysis:
                progress.write(f"  Доступность: {contact_analysis['contact_availability']}")
            if "contact_preferences" in contact_analysis:
                progress.write("  Предпочтения в коммуникации:")
                for pref in contact_analysis["contact_preferences"]:
                    progress.write(f"    - {pref}")
            if "contact_confidence" in contact_analysis:
                progress.write(f"  Уверенность в контактах: {contact_analysis['contact_confidence']:.2f}")

    # --- Метрики ---
    end_time = time.time()
    elapsed = end_time - start_time
    progress.write(Fore.GREEN + Style.BRIGHT + f"\n=== Метрики выполнения ===" + Style.RESET_ALL)
    progress.write(f"  Общее время выполнения: {elapsed:.2f} сек")
    progress.write(f"  Токены (облако): prompt={global_metrics['cloud_prompt_tokens']}, completion={global_metrics['cloud_completion_tokens']}")
    progress.write(f"  Токены (локально): prompt={global_metrics['local_prompt_tokens']}, completion={global_metrics['local_completion_tokens']}")
    
    # Закрываем прогресс-бар
    progress.close()

if __name__ == "__main__":
    main()