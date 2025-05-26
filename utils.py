def extract_content(result):
    """Извлекает текстовый контент из результата LLM-ответа (dict или str)."""
    if isinstance(result, dict) and 'content' in result:
        return result['content']
    return result

def robust_json_loads(s):
    """Устойчиво парсит JSON-строку, очищая спецсимволы и обрезки."""
    import re
    s = s.replace('"', '"').replace("'", "'")
    s = re.sub(r'<.*?>', '', s)
    s = s.strip()
    if s.startswith('{{') and s.endswith('}}'):
        s = s[1:-1]
    first = s.find('{')
    last = s.rfind('}')
    if first != -1 and last != -1 and last > first:
        s = s[first:last+1]
    try:
        import json
        return json.loads(s)
    except Exception as e:
        print(f"[!] Не удалось распарсить JSON: {e}\nСырой ответ:\n{s}")
        return None

def clean_input(s):
    """Удаляет неотображаемые и управляющие символы из строки."""
    cleaned = ''.join(c for c in s if c.isprintable()).strip()
    removed = [c for c in s if not c.isprintable()]
    if removed:
        print(f"[!] Ввод содержал неотображаемые символы: {[ord(c) for c in removed]}")
    return cleaned

def is_russian(text):
    """Проверяет, содержит ли текст хотя бы 80% кириллических символов среди букв."""
    import re
    letters = re.findall(r'[A-Za-zА-Яа-яЁё]', text)
    if not letters:
        return False
    cyrillic = [c for c in letters if 'А' <= c <= 'я' or c in 'Ёё']
    return len(cyrillic) / len(letters) > 0.8

def parse_final_sales_response(s):
    """Парсит финальный sales-ответ, сопоставляя английские ключи с русскими."""
    import re, json
    try:
        data = json.loads(s)
    except Exception:
        match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                return None
        else:
            return None
    key_map = {
        'company_profile': 'Профиль компании',
        'core_values': 'Ключевые ценности',
        'internal_pains': 'Внутренние боли',
        'hidden_concerns': 'Скрытые опасения',
        'org_culture_traits': 'Особенности корпоративной культуры',
        'key_success_factor': 'Ключевой фактор успеха',
        'key_needs': 'Ключевые потребности',
        'what_to_offer': 'Что можно предложить',
        'important_considerations': 'Важные моменты',
        'sales_recommendations': 'Рекомендации по продажам',
        'overall_assessment': 'Общий вывод',
        'mbti_profile': 'MBTI-профиль ЛПР'
    }
    result = {}
    for en, ru in key_map.items():
        if en in data:
            result[ru] = data[en]
        else:
            found = next((v for k, v in data.items() if k.startswith(en)), None)
            result[ru] = found
    return result

def normalize_llm_response(text):
    """Преобразует LLM-ответ в Markdown-вид, обрабатывает списки и JSON-блоки."""
    import re, json
    start_tag_json = "```json\n"
    end_tag_json = "\n```"
    processed_text = ""
    current_pos = 0
    while start_tag_json in text[current_pos:] and end_tag_json in text[current_pos:]:
        start_index_full = text.find(start_tag_json, current_pos)
        start_index_content = start_index_full + len(start_tag_json)
        end_index_full = text.find(end_tag_json, start_index_content)
        if end_index_full == -1:
            processed_text += text[current_pos:]
            break
        processed_text += text[current_pos:start_index_full]
        json_content = text[start_index_content:end_index_full]
        try:
            data = json.loads(json_content)
            markdown_from_json = ""
            if isinstance(data, dict):
                for key, value in data.items():
                    markdown_from_json += f"**{key.replace('_', ' ').capitalize()}:**\n"
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, (str, int, float, bool, type(None))):
                                markdown_from_json += f"- {item}\n"
                            elif isinstance(item, dict):
                                markdown_from_json += "- " + ", ".join([f"**{k}:** {v}" for k, v in item.items()]) + "\n"
                            else:
                                markdown_from_json += f"- {item}\n"
                        markdown_from_json += "\n"
                    elif isinstance(value, dict):
                         markdown_from_json += "- " + ", ".join([f"**{k.replace('_', ' ').capitalize()}:** {v}" for k, v in value.items()]) + "\n\n"
                    else:
                        markdown_from_json += f"{value}\n\n"
            elif isinstance(data, list):
                 markdown_from_json += "**Данные из списка JSON:**\n"
                 for item in data:
                      if isinstance(item, (str, int, float, bool, type(None))):
                           markdown_from_json += f"- {item}\n"
                      elif isinstance(item, dict):
                           markdown_from_json += "- " + ", ".join([f"**{k.replace('_', ' ').capitalize()}:** {v}" for k, v in item.items()]) + "\n"
                      else:
                           markdown_from_json += f"- {item}\n"
                 markdown_from_json += "\n"
            else:
                 markdown_from_json += f"**Данные из JSON:** {data}\n\n"
            processed_text += markdown_from_json.strip() + "\n"
            current_pos = end_index_full + len(end_tag_json)
        except json.JSONDecodeError:
            processed_text += text[start_index_full:end_index_full + len(end_tag_json)]
            current_pos = end_index_full + len(end_tag_json)
    processed_text += text[current_pos:]
    list_pattern = re.compile(r'\[.*?\]')
    temp_text_after_lists = ""
    last_end = 0
    for match in list_pattern.finditer(processed_text):
        start, end = match.span()
        list_string = match.group(0)
        temp_text_after_lists += processed_text[last_end:start]
        try:
            clean_list_string = list_string.strip()
            if clean_list_string.startswith('[') and clean_list_string.endswith(']'):
                 clean_list_string = clean_list_string.replace("'", '"')
            try:
                parsed_list = json.loads(clean_list_string)
            except json.JSONDecodeError:
                 temp_text_after_lists += list_string
                 last_end = end
                 continue
            if isinstance(parsed_list, list):
                markdown_list = ""
                for item in parsed_list:
                    markdown_list += f"- {item}\n"
                temp_text_after_lists += "\n" + markdown_list + "\n"
            else:
                 temp_text_after_lists += list_string
        except Exception as e:
             temp_text_after_lists += list_string
        last_end = end
    temp_text_after_lists += processed_text[last_end:]
    correction_pattern = re.compile(r'(^|\s),(\s*\*\*[^\*]+\*\*:)', re.MULTILINE)
    final_text = correction_pattern.sub(r'\1\2', temp_text_after_lists)
    return final_text.strip()

def is_valid_url(url):
    """Проверяет, что строка — валидный http(s) URL."""
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and result.netloc
    except Exception:
        return False

def parse_reflection_response(s):
    """Парсит reflection-ответ по новой схеме: analysis — массив целей/результатов, остальные ключи — отдельные секции."""
    import re, json
    try:
        data = json.loads(s)
    except Exception:
        match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                return None
        else:
            return None
    key_map = {
        'summary': 'Краткая сводка',
        'profile': 'Психологический портрет',
        'values': 'Ценности',
        'pains': 'Боли',
        'fears': 'Страхи',
        'conflicts': 'Внутренние конфликты',
        'motivation': 'Мотивация',
        'needs': 'Потребности',
        'hidden_needs': 'Скрытые потребности',
        'features': 'Особенности',
        'strengths': 'Сильные стороны',
        'weaknesses': 'Слабые стороны',
        'opportunities': 'Возможности',
        'threats': 'Угрозы'
    }
    result = {}
    # Сначала обработаем analysis
    analysis = data.get('analysis')
    if analysis and isinstance(analysis, list):
        result['Анализ по целям'] = []
        for i, item in enumerate(analysis, 1):
            obj = {}
            obj['Цель'] = item.get('objective', '')
            obj['Результат'] = item.get('result', '')
            result['Анализ по целям'].append(obj)
    # Теперь остальные ключи
    for k, v in data.items():
        if k == 'analysis':
            continue
        ru = key_map.get(k, k)
        result[ru] = v
    return result

def extract_links_from_text(text, base_url):
    """Извлекает все http(s) и относительные ссылки из текста, возвращает абсолютные URL на том же домене."""
    links = set()
    # Абсолютные ссылки
    for m in re.findall(r'https?://[^\s\'"<>]+', text):
        links.add(m.split('#')[0].rstrip('/'))
    # Относительные ссылки ("/about", "contacts/")
    for m in re.findall(r'href=[\'\"]?(/[^\'\" >]+)', text):
        abs_url = urllib.parse.urljoin(base_url, m)
        links.add(abs_url.split('#')[0].rstrip('/'))
    # Фильтруем только те, что на том же домене
    domain = urllib.parse.urlparse(base_url).netloc
    links = [l for l in links if urllib.parse.urlparse(l).netloc == domain]
    return list(links)

def to_absolute(url, base_url):
    """Преобразует относительный url в абсолютный относительно base_url."""
    return urllib.parse.urljoin(base_url, url) 