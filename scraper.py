import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from firecrawl import FirecrawlApp, ScrapeOptions
from tqdm import tqdm
from colorama import Fore, Style

# === ВАЖНО: Логика флагов local_only и force_remote ===
# local_only (-L): использовать только локальную модель для всех запросов, КРОМЕ тех, где явно указан force_remote.
# force_remote (-F): использовать облако (OpenRouter) только для ВАЖНЫХ запросов (финальный анализ, reflection и т.п.), даже если установлен -L.
# Если оба флага не указаны — авто-режим (сначала облако, если доступно, иначе локально).
# force_remote НЕ отменяет полностью local_only, а временно его переопределяет для конкретных запросов.
# ВСЕГДА передавайте оба параметра в важные LLM-запросы: local_only=local_only, force_remote=force_remote

# Функция для анализа сайта через Firecrawl

def crawl_and_analyze_url(
    url,
    limit=10,
    mode='single',
    local_only=True,
    force_remote=False,
    analyze_page_single=None,
    analyze_page_multi=None,
    complete_result_if_needed=None,
    calculate_requests_needed=None,
    normalize_llm_response=None,
    debug=False,
    selected_pages=None
):
    """
    Все функции анализа (single/multi) и postprocessing передаются как параметры для слабой связанности.
    debug: если True — сохранять страницы на диск и выводить print по этапам.
    selected_pages: если передан список URL, анализировать только их (по одной), иначе limit.
    """
    api_url = os.getenv("SCRAPER_API_URL", "http://localhost:3002")
    app = FirecrawlApp(api_url=api_url)
    if selected_pages:
        data = []
        for page_url in selected_pages:
            crawl_status = app.crawl_url(
                page_url,
                limit=1,
                scrape_options=ScrapeOptions(formats=['markdown'], exclude_paths=['blog/*'], location={"languages": ["ru"]}),
                poll_interval=5
            )
            data.extend(crawl_status.data)
    else:
        crawl_status = app.crawl_url(
            url,
            limit=limit,
            scrape_options=ScrapeOptions(formats=['markdown'], exclude_paths=['blog/*'], location={"languages": ["ru"]}),
            poll_interval=5
        )
        data = crawl_status.data
    combined_results = []
    page_summaries = []  # Для хранения кратких summary по каждой странице
    if debug:
        os.makedirs('outputs', exist_ok=True)
    page_num = 1
    requests_needed = calculate_requests_needed(len(data), mode)
    if debug:
        print(Fore.YELLOW + f"Количество запросов к LLM: {requests_needed}." + Style.RESET_ALL)

    for item in tqdm(data, desc="Анализ страниц"):
        doc = item.model_dump()
        content = None
        for fmt in ['markdown', 'text', 'html']:
            if fmt in doc:
                content = doc[fmt]
                break
        if content:
            if debug:
                with open(f'outputs/page_{page_num}.md', 'w', encoding='utf-8') as f:
                    f.write(content)
            page_num += 1

            if mode == 'multi':
                context = f"Текст страницы:\n{content}"
                analysis = analyze_page_multi(content, local_only=local_only)
                steps = analysis.get('steps', [])
                if len(steps) >= 3:
                    prev = '\n'.join([f"Этап {j+1}: {r}" for j, r in enumerate(steps[:3])])
                    context += f"\n\nРезультаты предыдущих этапов:\n{prev}"
                if steps and isinstance(steps, list) and "result" in steps:
                    result_json = complete_result_if_needed(steps, context,
                                                           role=f"website org psych analyzer step 4", task=None, local_only=local_only)
                    combined_results.append(result_json["result"])
                    page_summaries.append(result_json)
                elif steps:
                    combined_results.append(steps)
                    page_summaries.append({"result": steps})
            else:
                context = "Raw Content: " + content
                analysis = analyze_page_single(content, local_only=local_only)
                if analysis and isinstance(analysis, dict) and "result" in analysis:
                    result_json = complete_result_if_needed(analysis, context,
                                                           role="website deep org psych analyzer", task=None, local_only=local_only)
                    combined_results.append(result_json["result"])
                    page_summaries.append(result_json)
                elif analysis:
                    combined_results.append(analysis)
                    page_summaries.append({"result": analysis})
    final_text = '\n'.join(str(r) for r in combined_results if r)
    return {
        "result": final_text,
        "page_summaries": page_summaries
    }

def get_site_links(url):
    """Возвращает список ссылок с сайта (карта сайта через Firecrawl)."""
    api_url = os.getenv("SCRAPER_API_URL", "http://localhost:3002")
    app = FirecrawlApp(api_url=api_url)
    site_map = app.map_url(
        url,
        limit=1,
        search=None
    )
    return getattr(site_map, 'links', [])

def get_page_content(url):
    """Возвращает текст главной страницы (markdown или html)."""
    api_url = os.getenv("SCRAPER_API_URL", "http://localhost:3002")
    app = FirecrawlApp(api_url=api_url)
    crawl_status = app.crawl_url(
        url,
        limit=1,
        scrape_options=ScrapeOptions(formats=['markdown', 'html'], onlyMainContent=False),
        poll_interval=5
    )
    doc = crawl_status.data[0].model_dump() if hasattr(crawl_status.data[0], 'model_dump') else crawl_status.data[0]
    content = doc.get('markdown') or doc.get('html') or ''
    return content 