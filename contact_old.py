import csv
import json
import os
import re
import sys
import argparse
import urllib.parse

from dotenv import load_dotenv
load_dotenv()
from firecrawl import FirecrawlApp, ScrapeOptions
from openai import OpenAI
from tqdm import tqdm
from colorama import init, Fore, Style
from llm_router import sales_generate_completion

#print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
# Initialize FirecrawlApp and OpenAI
app = FirecrawlApp(api_url="http://localhost:3002")
client = OpenAI(
    api_key="1234567890",
    base_url="http://192.168.88.250:1234/v1"
)

# === JSON Schema для контакта ===
CONTACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "contact_result",
        "schema": {
            "type": "object",
            "properties": {
                "contacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "full_name": {"type": "string", "description": "ФИО или имя контакта"},
                            "role": {"type": "string", "description": "Роль или должность"},
                            "email": {"type": "string", "description": "Email"},
                            "phone": {"type": "string", "description": "Телефон"},
                            "whatsapp": {"type": "string", "description": "WhatsApp"},
                            "telegram": {"type": "string", "description": "Telegram"},
                            "vk": {"type": "string", "description": "ВКонтакте"},
                            "comment": {"type": "string", "description": "Комментарий или пояснение от модели"}
                        },
                        "required": []
                    },
                    "description": "Список найденных контактов на странице"
                }
            },
            "required": ["contacts"]
        }
    }
}

CONTACT_PROMPT = (
    "Ты — эксперт по анализу сайтов и поиску контактных данных. "
    "Твоя задача — найти и структурировать контакты реальных сотрудников компании по тексту страницы сайта. "
    "Ты можешь использовать только текст страницы, не используй никакие другие источники информации. "
)

user_prompt_template = (
    "Добавляй только те контакты, которые реально присутствуют в тексте страницы.\n"
    "В первую очередь ищи и добавляй ЛПР (лиц, принимающих решения: директора, руководителей, владельцев, топ-менеджеров, начальников отделов и т.п.). Если ЛПР не найдены, добавь контактные данные организации (общий email, телефон, мессенджеры, указанные для связи с компанией).\n"
    "Каждый контакт в списке ДОЛЖЕН БЫТЬ УНИКАЛЬНЫМ по совокупности всех полей. Добавляй каждый уникальный контакт только один раз.\n"
    "Если не удалось определить имя или роль, оставь эти поля пустыми (не используй значения вроде 'Не указано', 'Неизвестно', 'N/A', 'Unknown').\n"
    "Если контакт уже есть в списке, не добавляй его повторно.\n"
    "Если не удалось определить имя или роль, добавь такой контакт только один раз, даже если способов связи несколько.\n"
    "Добавляй только контакты реальных людей, работающих в компании, которую анализируешь.\n"
    "Включай только реальных сотрудников компании, пропуская виртуальных персонажей, чат-ботов и автоматизированные системы.\n"
    "Указывай только роли и должности внутри компании, которую анализируешь.\n"
    "Для каждого контакта указывай имя/ФИО (если есть), роль (если есть), все найденные способы связи, комментарий (если есть).\n"
    "В поле telegram указывай только персональные Telegram-аккаунты сотрудников (например, @username или https://t.me/username).\n"
    "В поле vk указывай только персональные страницы ВКонтакте (например, https://vk.com/id12345 или https://vk.com/username).\n"
    "Пропускай группы, каналы, чаты, приглашения, ботов и любые ссылки на Telegram или VK, которые не являются персональными аккаунтами (например, https://t.me/+abc123, https://t.me/joinchat/..., https://vk.com/club..., https://vk.com/public...).\n"
    "Примеры корректных telegram: @ivan_ivanov, https://t.me/ivan_ivanov\n"
    "Примеры некорректных telegram (их пропускай): https://t.me/+FtRxC_6tfdA1ZDhi, https://t.me/joinchat/AAAAAE..., https://t.me/s/pro100robot, https://t.me/bot?start=abc\n"
    "Примеры корректных vk: https://vk.com/id12345, https://vk.com/ivan_ivanov\n"
    "Примеры некорректных vk (их пропускай): https://vk.com/club12345, https://vk.com/public12345, https://vk.com/app..., https://vk.com/event....\n"
    "Добавляй только те контакты, у которых есть хотя бы один реальный способ связи (email, телефон, whatsapp, telegram, vk).\n"
    "Если на странице есть только общие контактные данные организации (например, info@..., office@..., общий телефон или мессенджер), добавь их отдельным контактом с пустыми полями ФИО и роль.\n"
    "Пример контакта ЛПР: {{\"full_name\": \"Иван Иванов\", \"role\": \"Генеральный директор\", ...}}\n"
    "Пример общего контакта организации: {{\"full_name\": \"\", \"role\": \"\", \"email\": \"info@company.ru\", ...}}\n"
    "Ответ верни строго в виде JSON-объекта по схеме: contacts — массив объектов с полями full_name, role, email, phone, whatsapp, telegram, vk, comment.\n"
    "Если контактов нет — верни пустой массив. Ответ только на русском языке, без англоязычных вставок.\n\n"
    "Текст страницы:\n{page_text}"
)

def unified_generate_completion(role, task, content, temperature=0.1, response_format=None, is_final=False, local_only=False):
    messages = [
        {"role": "system", "content": f"You are a {role}. {task}"},
        {"role": "user", "content": content}
    ]
    return sales_generate_completion(
        role=role,
        task=task,
        content=content,
        temperature=temperature,
        response_format=response_format,
        is_final=is_final,
        local_only=local_only,
        messages=messages
    )

def normalize_email(email):
    return email.strip().lower() if email else None

def normalize_phone(phone):
    if not phone:
        return None
    # Оставляем только цифры
    digits = re.sub(r'\D', '', phone)
    # Российские номера: если начинается с 8, заменяем на 7
    if len(digits) == 11 and digits.startswith('8'):
        digits = '7' + digits[1:]
    # Добавляем плюс, если длина >= 10
    if len(digits) >= 10:
        return f'+{digits}'
    return None

def normalize_whatsapp(wa):
    if not wa:
        return None
    digits = re.sub(r'\D', '', wa)
    if len(digits) >= 10:
        return f'+{digits}'
    return None

def normalize_telegram(tg):
    if not tg:
        return None
    tg = tg.strip()
    # Если это ссылка вида https://t.me/username, извлекаем username
    match = re.search(r'(?:https?://)?t\.me/([a-zA-Z0-9_]{4,})', tg)
    if match:
        tg = match.group(1)
    # Если начинается с @, убираем @ для унификации
    tg = tg.lstrip('@')
    return '@' + tg.lower()

def normalize_vk(vk):
    if not vk:
        return None
    vk = vk.strip().lower()
    # Оставляем только https://vk.com/username или https://vk.com/id12345
    match = re.search(r'(?:https?://)?vk\.com/([a-zA-Z0-9_\.]+)', vk)
    if match:
        return f'https://vk.com/{match.group(1)}'
    return None



def extract_phones(text):
    # Сохраняем tel: ссылки отдельно
    tel_links = re.findall(r'tel:([+\d][\d\s\-\(\)]{9,}\d)', text)
    # Удаляем все URL, кроме tel:
    text_wo_urls = re.sub(r'https?://\S+', ' ', text)
    text_wo_urls = re.sub(r'ftp://\S+', ' ', text_wo_urls)
    # Дальше ищем телефоны только в очищенном тексте
    phone_pattern = re.compile(
        r'(?:\+7|8|7)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{2}[\s\-\(\)]*\d{2}'
        r'|\+\d{1,3}[\s\-\(\)]*\d{1,4}[\s\-\(\)]*\d{3,4}[\s\-\(\)]*\d{2,4}[\s\-\(\)]*\d{2,4}'
    )
    phones = set()
    # Добавляем телефоны из tel: ссылок
    for tel in tel_links:
        digits = re.sub(r'\D', '', tel)
        if (len(digits) == 11 and (digits.startswith('7') or digits.startswith('8'))):
            phones.add(tel.strip())
        elif (len(digits) >= 10 and len(digits) <= 15 and tel.strip().startswith('+')):
            phones.add(tel.strip())
    # Добавляем телефоны из обычного текста (без URL)
    for match in phone_pattern.findall(text_wo_urls):
        digits = re.sub(r'\D', '', match)
        if (len(digits) == 11 and (digits.startswith('7') or digits.startswith('8'))):
            phones.add(match.strip())
        elif (len(digits) >= 10 and len(digits) <= 15 and match.strip().startswith('+')):
            phones.add(match.strip())
    return phones

def extract_contacts_regex(text):
    # Email: user@domain.com
    emails = set(re.findall(r'\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b', text))
    
    # Телефоны: только реальные номера
    phones = set()
    for phone in extract_phones(text):
        norm = normalize_phone(phone)
        if norm:
            phones.add(norm)
    
    # WhatsApp: wa.me/номер
    whatsapps = set()
    for match in re.findall(r'wa\.me/(\d{10,15})', text):
        norm = normalize_whatsapp(match)
        if norm:
            whatsapps.add(norm)
    
    # Telegram: ссылки и @username, но не email
    telegrams = set()
    telegrams.update(re.findall(r'https?://t\.me/([a-zA-Z0-9_]{4,})', text))
    telegrams.update(re.findall(r'(?<![\w@])@([a-zA-Z0-9_]{4,})\b(?!\s*\.)', text))  # @username, не email
    telegrams = {normalize_telegram(tg) for tg in telegrams}
    
    # VK: https://vk.com/username или https://vk.com/id12345
    vks = set()
    for match in re.findall(r'https?://vk\.com/[a-zA-Z0-9_\.]+', text):
        norm = normalize_vk(match)
        if norm:
            vks.add(norm)
    
    
    contacts = []
    for email in emails:
        contacts.append({"email": normalize_email(email)})
    for phone in phones:
        if not any(c.get("phone") == phone for c in contacts):
            contacts.append({"phone": phone})
    for wa in whatsapps:
        if not any(c.get("whatsapp") == wa for c in contacts):
            contacts.append({"whatsapp": wa})
    for tg in telegrams:
        if not any(c.get("telegram") == tg for c in contacts):
            contacts.append({"telegram": tg})
    for vk in vks:
        if not any(c.get("vk") == vk for c in contacts):
            contacts.append({"vk": vk})
    return contacts

def premerge_contacts(contacts):
    merged = []
    for c in contacts:
        found = None
        for m in merged:
            if (
                (c.get("email") and m.get("email") == c.get("email")) or
                (c.get("phone") and m.get("phone") == c.get("phone")) or
                (c.get("whatsapp") and m.get("whatsapp") == c.get("whatsapp")) or
                (c.get("telegram") and m.get("telegram") == c.get("telegram")) or
                (c.get("vk") and m.get("vk") == c.get("vk"))

            ):
                found = m
                break
        if found:
            found.update({k: v for k, v in c.items() if v})
        else:
            merged.append(c.copy())
    return merged

def contacts_to_lines(contacts):
    lines = []
    for c in contacts:
        if c.get("email"):
            lines.append(f"email: {c['email']}")
        if c.get("phone"):
            lines.append(f"phone: {c['phone']}")
        if c.get("whatsapp"):
            lines.append(f"whatsapp: {c['whatsapp']}")
        if c.get("telegram"):
            lines.append(f"telegram: {c['telegram']}")
        if c.get("vk"):
            lines.append(f"vk: {c['vk']}")

    return lines

def validate_llm_contacts_by_regex(llm_contacts, text, verbose=False):
    regex_contacts = extract_contacts_regex(text)
    found_emails = {normalize_email(c['email']) for c in regex_contacts if c.get('email')}
    found_phones = {normalize_phone(c['phone']) for c in regex_contacts if c.get('phone')}
    found_whatsapps = {normalize_whatsapp(c['whatsapp']) for c in regex_contacts if c.get('whatsapp')}
    found_telegrams = {normalize_telegram(c['telegram']) for c in regex_contacts if c.get('telegram')}
    found_vks = {normalize_vk(c['vk']) for c in regex_contacts if c.get('vk')}
    validated = []
    for c in llm_contacts:
        new_c = c.copy()
        removed = []
        if new_c.get('email') and normalize_email(new_c['email']) not in found_emails:
            removed.append(('email', new_c['email']))
            new_c['email'] = ''
        if new_c.get('phone') and normalize_phone(new_c['phone']) not in found_phones:
            removed.append(('phone', new_c['phone']))
            new_c['phone'] = ''
        if new_c.get('whatsapp') and normalize_whatsapp(new_c['whatsapp']) not in found_whatsapps:
            removed.append(('whatsapp', new_c['whatsapp']))
            new_c['whatsapp'] = ''
        if new_c.get('telegram') and normalize_telegram(new_c['telegram']) not in found_telegrams:
            removed.append(('telegram', new_c['telegram']))
            new_c['telegram'] = ''
        if new_c.get('vk') and normalize_vk(new_c['vk']) not in found_vks:
            removed.append(('vk', new_c['vk']))
            new_c['vk'] = ''
        if removed and verbose:
            print(f"[DEBUG] У контакта удалены несуществующие способы связи: {removed} | {c}")
        if new_c.get('email') or new_c.get('phone') or new_c.get('whatsapp') or new_c.get('telegram') or new_c.get('vk'):
            validated.append(new_c)
        elif verbose:
            print(f"[DEBUG] Контакт удалён как галлюцинация: {c}")
    return validated

def is_virtual_contact(contact):
    stopwords = [
        'цифровой', 'бот', 'аватар', 'виртуальный', 'робот', 'чат-бот', 'автоматизированный',
        'digital', 'assistant', 'avatar', 'virtual', 'ai', 'искусственный интеллект'
    ]
    fields = [contact.get('full_name', ''), contact.get('role', ''), contact.get('comment', '')]
    text = ' '.join(fields).lower()
    return any(word in text for word in stopwords)

def enrich_contacts_with_llm(text, local_only=True, verbose=False):
    if verbose:
        print(Fore.BLUE + "\nЗапуск LLM для страницы (без передачи контактов):" + Style.RESET_ALL)
    prompt = user_prompt_template.format(page_text=text)
    response = sales_generate_completion(
        role="contact enricher",
        task=CONTACT_PROMPT,
        content=prompt,
        temperature=0.4,
        response_format=CONTACT_SCHEMA,
        is_final=True,
        local_only=local_only
    )
    if verbose:
        print(Fore.CYAN + "\n[DEBUG] Raw LLM response for contact enrichment:" + Style.RESET_ALL)
        if isinstance(response, dict):
            debug_response = {k: v for k, v in response.items() if k != 'usage'}
            print(json.dumps(debug_response, indent=2, ensure_ascii=False))
        else:
            print(str(response))
    try:
        if isinstance(response, dict):
            content = response.get("content", "")
            if content:
                try:
                    data = json.loads(content)
                    llm_contacts = data.get("contacts", [])
                except json.JSONDecodeError as e:
                    if verbose:
                        print(Fore.RED + f"[DEBUG] Error parsing content JSON: {e}" + Style.RESET_ALL)
                    llm_contacts = response.get("contacts", [])
            else:
                llm_contacts = response.get("contacts", [])
        elif isinstance(response, str):
            try:
                data = json.loads(response)
                llm_contacts = data.get("contacts", [])
            except json.JSONDecodeError as e:
                if verbose:
                    print(Fore.RED + f"[DEBUG] Error parsing response string: {e}" + Style.RESET_ALL)
                llm_contacts = []
        else:
            llm_contacts = []
    except Exception as e:
        if verbose:
            print(Fore.RED + f"[DEBUG] Error parsing LLM response: {e}" + Style.RESET_ALL)
        match = re.search(r'\{.*\}', str(response), re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                llm_contacts = data.get("contacts", [])
            except Exception as e:
                if verbose:
                    print(Fore.RED + f"[DEBUG] Error parsing regex match: {e}" + Style.RESET_ALL)
                llm_contacts = []
        else:
            llm_contacts = []
    if verbose:
        print(Fore.GREEN + f"\n[DEBUG] Parsed contacts from LLM: {llm_contacts}" + Style.RESET_ALL)
    llm_contacts = validate_llm_contacts_by_regex(llm_contacts, text, verbose=verbose)
    if verbose:
        print(Fore.YELLOW + f"\n[DEBUG] Contacts after regex validation: {llm_contacts}" + Style.RESET_ALL)
    if not any(c.get("telegram") for c in llm_contacts):
        if verbose:
            print(Fore.RED + "[DEBUG] ВНИМАНИЕ: LLM не вернул ни одного Telegram!" + Style.RESET_ALL)
            print("[DEBUG] Первые 500 символов текста страницы:")
            print(text[:500])
    merged = merge_contacts([llm_contacts])
    if verbose:
        print(Fore.MAGENTA + f"\n[DEBUG] Contacts after merging: {merged}" + Style.RESET_ALL)
    def has_key_field(c):
        return c.get("email") or c.get("phone") or c.get("whatsapp") or c.get("telegram") or c.get("vk")
    merged = [c for c in merged if has_key_field(c)]
    if verbose:
        print(Fore.CYAN + f"\n[DEBUG] Contacts after key field filtering: {merged}" + Style.RESET_ALL)
    merged = [c for c in merged if not is_virtual_contact(c)]
    if verbose:
        print(Fore.BLUE + f"\n[DEBUG] Final contacts after virtual contact filtering: {merged}" + Style.RESET_ALL)
    return merged

def extract_contacts_from_content(content, local_only=True):
    regex_contacts = extract_contacts_regex(content)
    if not regex_contacts:
        return []  # Если ничего не найдено регулярками, не вызываем LLM и не добавляем контакты
    enriched_contacts = enrich_contacts_with_llm(content, local_only=local_only)
    return enriched_contacts

def merge_contacts(contact_lists):
    merged = []
    for contacts in contact_lists:
        for c in contacts:
            email = normalize_email(c.get("email"))
            phone = normalize_phone(c.get("phone"))
            whatsapp = normalize_whatsapp(c.get("whatsapp"))
            telegram = normalize_telegram(c.get("telegram"))
            vk = normalize_vk(c.get("vk"))
            full_name = (c.get("full_name") or '').strip().lower()
            role = (c.get("role") or '').strip().lower()
            found = None
            for m in merged:
                if (
                    (email and normalize_email(m.get("email")) == email) or
                    (phone and normalize_phone(m.get("phone")) == phone) or
                    (whatsapp and normalize_whatsapp(m.get("whatsapp")) == whatsapp) or
                    (telegram and normalize_telegram(m.get("telegram")) == telegram) or
                    (vk and normalize_vk(m.get("vk")) == vk)
                ):
                    m_full_name = (m.get("full_name") or '').strip().lower()
                    m_role = (m.get("role") or '').strip().lower()
                    if (full_name and m_full_name and full_name != m_full_name):
                        continue
                    if (role and m_role and role != m_role):
                        continue
                    found = m
                    break
            if found:
                for k, v in c.items():
                    if v and (not found.get(k)):
                        found[k] = v
            else:
                merged.append(c.copy())
    return merged

def finalize_contacts_format(contacts):
    finalized = []
    for c in contacts:
        new_c = c.copy()
        if new_c.get('whatsapp'):
            num = normalize_whatsapp(new_c['whatsapp'])
            if num:
                new_c['whatsapp'] = num
        if new_c.get('telegram'):
            tg = normalize_telegram(new_c['telegram'])
            if tg:
                new_c['telegram'] = tg
        if new_c.get('vk'):
            vk = normalize_vk(new_c['vk'])
            if vk:
                new_c['vk'] = vk
        finalized.append(new_c)
    return finalized

def crawl_and_extract_contacts(url, limit=10, local_only=True, verbose=False):
    crawl_status = app.crawl_url(
        url,
        limit=limit,
        scrape_options=ScrapeOptions(
            formats=['markdown', 'html'],
            onlyMainContent=False,
            exclude_paths=['blog/*'],
        ),
        poll_interval=5
    )
    data = crawl_status.data
    all_contacts = []
    os.makedirs('outputs', exist_ok=True)
    page_num = 1
    llm_requests = 0
    email_has_name = dict()
    phone_has_name = dict()
    whatsapp_has_name = dict()
    telegram_has_name = dict()
    vk_has_name = dict()
    for item in tqdm(data, desc="Извлечение контактов"):
        doc = item.model_dump() if hasattr(item, 'model_dump') else (item if isinstance(item, dict) else {})
        content = None
        for fmt in ['markdown', 'text', 'html']:
            if fmt in doc:
                content = doc[fmt]
                break
        if content:
            if verbose:
                print(Fore.CYAN + f"\n[DEBUG] Content from page {page_num}:" + Style.RESET_ALL)
                print(content[:1000])
            with open(f'outputs/page_{page_num}.md', 'w', encoding='utf-8') as f:
                f.write(content)
            page_num += 1
            regex_contacts = extract_contacts_regex(content)
            if verbose:
                print(Fore.YELLOW + f"\n[DEBUG] Regex found contacts: {regex_contacts}" + Style.RESET_ALL)
            def not_enriched(c):
                email_key = normalize_email(c.get("email"))
                phone_key = normalize_phone(c.get("phone"))
                whatsapp_key = normalize_whatsapp(c.get("whatsapp"))
                telegram_key = normalize_telegram(c.get("telegram"))
                vk_key = normalize_vk(c.get("vk"))
                res = (
                    (email_key and not email_has_name.get(email_key, False)) or
                    (phone_key and not phone_has_name.get(phone_key, False)) or
                    (whatsapp_key and not whatsapp_has_name.get(whatsapp_key, False)) or
                    (telegram_key and not telegram_has_name.get(telegram_key, False)) or
                    (vk_key and not vk_has_name.get(vk_key, False))
                )
                if verbose:
                    print(f"[DEBUG not_enriched] email={email_key}({email_has_name.get(email_key)}), phone={phone_key}({phone_has_name.get(phone_key)}), whatsapp={whatsapp_key}({whatsapp_has_name.get(whatsapp_key)}), telegram={telegram_key}({telegram_has_name.get(telegram_key)}), vk={vk_key}({vk_has_name.get(vk_key)}), => {res}")
                return res
            filtered_contacts = [c for c in regex_contacts if not_enriched(c)]
            if filtered_contacts:
                llm_requests += 1
                contacts = enrich_contacts_with_llm(content, local_only=local_only, verbose=verbose)
                if verbose:
                    print(Fore.GREEN + f"\n[DEBUG] LLM found contacts: {contacts}" + Style.RESET_ALL)
                llm_keys = set()
                for c in contacts:
                    if c.get("email"):
                        llm_keys.add(("email", normalize_email(c["email"])))
                    if c.get("phone"):
                        llm_keys.add(("phone", normalize_phone(c["phone"])))
                    if c.get("whatsapp"):
                        llm_keys.add(("whatsapp", normalize_whatsapp(c["whatsapp"])))
                    if c.get("telegram"):
                        llm_keys.add(("telegram", normalize_telegram(c["telegram"])))
                    if c.get("vk"):
                        llm_keys.add(("vk", normalize_vk(c["vk"])))
                for c in contacts:
                    has_name = bool(c.get("full_name") and c["full_name"].strip())
                    if has_name:
                        if c.get("email"):
                            k = normalize_email(c["email"])
                            email_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён email: {k}")
                        if c.get("phone"):
                            k = normalize_phone(c["phone"])
                            phone_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён phone: {k}")
                        if c.get("whatsapp"):
                            k = normalize_whatsapp(c["whatsapp"])
                            whatsapp_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён whatsapp: {k}")
                        if c.get("telegram"):
                            k = normalize_telegram(c["telegram"])
                            telegram_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён telegram: {k}")
                        if c.get("vk"):
                            k = normalize_vk(c["vk"])
                            vk_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён vk: {k}")
                for c in filtered_contacts:
                    if c.get("phone"):
                        k = normalize_phone(c["phone"])
                        if ("phone", k) not in llm_keys:
                            phone_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Игнорируемый phone: {k}")
                if verbose:
                    print("[DEBUG] Словари после обновления:")
                    print("  email_has_name:", email_has_name)
                    print("  phone_has_name:", phone_has_name)
                    print("  whatsapp_has_name:", whatsapp_has_name)
                    print("  telegram_has_name:", telegram_has_name)
                    print("  vk_has_name:", vk_has_name)
                all_contacts.append(contacts)
    if verbose:
        print(Fore.YELLOW + f"Количество запросов к LLM: {llm_requests}." + Style.RESET_ALL)
    merged_contacts = merge_contacts(all_contacts)
    merged_contacts = finalize_contacts_format(merged_contacts)
    merged_contacts = [c for c in merged_contacts if not is_virtual_contact(c)]
    return merged_contacts

def save_contacts_csv(contacts, filename):
    if not contacts:
        print(Fore.RED + "Нет контактов для сохранения." + Style.RESET_ALL)
        return
    keys = ["full_name", "role", "email", "phone", "whatsapp", "telegram", "vk", "comment"]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for c in contacts:
            writer.writerow({k: c.get(k, "") for k in keys})
    print(Fore.GREEN + f"Контакты сохранены в {filename}" + Style.RESET_ALL)

def clean_input(s):
    cleaned = ''.join(c for c in s if c.isprintable()).strip()
    # Диагностика: если были удалены символы, вывести их коды
    removed = [c for c in s if not c.isprintable()]
    if removed:
        print(f"[!] Ввод содержал неотображаемые символы: {[ord(c) for c in removed]}")
    return cleaned

def summarize_contacts_with_llm(contacts, local_only=True, verbose=False):
    prompt = (
        "Вот список всех уникальных контактов, найденных на сайте (ФИО, роль, email, телефон, мессенджеры, комментарии). "
        "Проанализируй их и выбери только действительно ценные, полезные и релевантные для связи с компанией. "
        "Если есть дубли, оставь только лучший вариант. Если есть ЛПР — выдели их в первую очередь. "
        "Если есть только общие контакты организации — выбери наиболее надёжные. "
        "Ответ верни строго в виде JSON-массива контактов с теми же полями, что и на входе (full_name, role, email, phone, whatsapp, telegram, vk, comment). "
        "Не добавляй новых контактов, не придумывай ничего. Не пиши пояснений вне JSON."
        "\n\nСписок контактов:\n" + json.dumps(contacts, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": "Ты — эксперт по анализу контактов и коммуникаций компаний."},
        {"role": "user", "content": prompt}
    ]
    response = sales_generate_completion(
        role="contact analyzer",
        task="Analyze and select the most valuable contacts",
        content=prompt,
        temperature=0.1,
        response_format=CONTACT_SCHEMA,
        is_final=True,
        local_only=local_only,
        messages=messages
    )
    if verbose:
        print(Fore.CYAN + "\n[DEBUG] Raw LLM response for contact summarization:" + Style.RESET_ALL)
        if isinstance(response, dict):
            debug_response = {k: v for k, v in response.items() if k != 'usage'}
            print(json.dumps(debug_response, indent=2, ensure_ascii=False))
        else:
            print(str(response))
    try:
        if isinstance(response, dict):
            content = response.get("content", "")
            if content:
                try:
                    data = json.loads(content)
                    best_contacts = data.get("contacts", [])
                except json.JSONDecodeError as e:
                    if verbose:
                        print(Fore.RED + f"[DEBUG] Error parsing content JSON: {e}" + Style.RESET_ALL)
                    best_contacts = response.get("contacts", [])
            else:
                best_contacts = response.get("contacts", [])
        elif isinstance(response, str):
            try:
                data = json.loads(response)
                best_contacts = data.get("contacts", [])
            except json.JSONDecodeError as e:
                if verbose:
                    print(Fore.RED + f"[DEBUG] Error parsing response string: {e}" + Style.RESET_ALL)
                best_contacts = []
        else:
            best_contacts = []
    except Exception as e:
        if verbose:
            print(Fore.RED + f"[DEBUG] Error parsing LLM response: {e}" + Style.RESET_ALL)
        match = re.search(r'\{.*\}', str(response), re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                best_contacts = data.get("contacts", [])
            except Exception as e:
                if verbose:
                    print(Fore.RED + f"[DEBUG] Error parsing regex match: {e}" + Style.RESET_ALL)
                best_contacts = []
        else:
            best_contacts = []
    if verbose:
        print(Fore.GREEN + f"\n[DEBUG] Parsed best contacts: {best_contacts}" + Style.RESET_ALL)
    return best_contacts

def select_best_contact_pages(links, limit, local_only=True, force_remote=False, verbose=False):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "contact_page_selection_result",
            "schema": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список наиболее релевантных для поиска контактов страниц (URL)"
                    }
                },
                "required": ["urls"]
            }
        }
    }
    prompt = (
        "Вот список страниц сайта. "
        "Выбери из них только те, на которых с наибольшей вероятностью могут быть контактные данные реальных сотрудников компании "
        "(например, страницы о компании, команде, руководстве, офисах, филиалах и т.п.). "
        "Блоги, новости, статьи, продукты, услуги, политики, пользовательские соглашения и любые нерелевантные страницы ВКЛЮЧАТЬ НЕЛЬЗЯ (не добавляй их в ответ). "
        f"Максимум {limit} ссылок. Если подходящих страниц меньше — верни только их. Если нет ни одной релевантной — верни пустой список.\n\n"
        "Ответ верни строго в виде JSON-объекта по схеме: urls — массив строк (URL).\n\n"
        "Список ссылок:\n" + "\n".join(links)
    )
    messages = [
        {"role": "system", "content": "Ты — эксперт по анализу сайтов."},
        {"role": "user", "content": prompt}
    ]
    response = sales_generate_completion(
        role="website analyzer",
        task="Select the most relevant pages for contact information",
        content=prompt,
        temperature=0.4,
        response_format=schema,
        is_final=True,
        local_only=local_only,
        force_remote=force_remote,
        messages=messages
    )
    if verbose:
        print(Fore.CYAN + "\n[DEBUG] Raw LLM response for page selection:" + Style.RESET_ALL)
        if isinstance(response, dict):
            debug_response = {k: v for k, v in response.items() if k != 'usage'}
            print(json.dumps(debug_response, indent=2, ensure_ascii=False))
        else:
            print(str(response))
    try:
        if isinstance(response, dict):
            content = response.get("content", "")
            if content:
                try:
                    data = json.loads(content)
                    urls = data.get("urls", [])
                except json.JSONDecodeError:
                    urls = response.get("urls", [])
            else:
                urls = response.get("urls", [])
        else:
            data = json.loads(response)
            urls = data.get("urls", [])
        if isinstance(urls, list):
            if verbose:
                print(Fore.GREEN + f"\n[DEBUG] Selected URLs: {urls}" + Style.RESET_ALL)
            return urls[:limit]
    except Exception as e:
        if verbose:
            print(Fore.RED + f"[DEBUG] Error parsing response: {e}" + Style.RESET_ALL)
        return [line.strip() for line in str(response).splitlines() if line.strip().startswith('http')][:limit]
    return []

def extract_links_from_text(text, base_url):
    # Ищет все http(s) ссылки и относительные пути
    links = set()
    # Абсолютные ссылки
    for m in re.findall(r'https?://[^\s\'"<>]+', text):
        links.add(m.split('#')[0].rstrip('/'))
    # Относительные ссылки ("/about", "contacts/")
    for m in re.findall(r'href=[\'"]?(/[^\'" >]+)', text):
        abs_url = urllib.parse.urljoin(base_url, m)
        links.add(abs_url.split('#')[0].rstrip('/'))
    # Фильтруем только те, что на том же домене
    domain = urllib.parse.urlparse(base_url).netloc
    links = [l for l in links if urllib.parse.urlparse(l).netloc == domain]
    return list(links)

def to_absolute(url, base):
    return urllib.parse.urljoin(base, url)

if __name__ == "__main__":
    init(autoreset=True)
    parser = argparse.ArgumentParser(
        description="Извлечение контактных данных с сайта. Пример: python contact.py -u example.com -l 5"
    )
    parser.add_argument('-u', '--url', type=str, help='URL сайта для анализа (можно с https:// или без).', required=True)
    parser.add_argument('-l', '--limit', type=int, help='Максимальное количество страниц для анализа (по умолчанию 10)', default=10)
    parser.add_argument('-L', '--local-only', action='store_true', help='Использовать только локальную модель, игнорировать OpenRouter')
    parser.add_argument('-o', '--output', type=str, help='Файл для сохранения контактов (CSV)')
    parser.add_argument('-F', '--force-remote', action='store_true', help='Сделать финальный запрос к провайдеру для отбора лучших контактов')
    parser.add_argument('-v', '--verbose', action='store_true', help='Выводить отладочную информацию')
    args = parser.parse_args()

    raw_url = clean_input(args.url)
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        url = f"https://{raw_url}"
    else:
        url = raw_url
    limit = args.limit
    local_only = args.local_only
    output_file = args.output
    force_remote = args.force_remote
    verbose = args.verbose

    if verbose:
        print(Fore.YELLOW + f"\nИспользуемый URL: {url}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Лимит страниц: {limit}" + Style.RESET_ALL)

    print(Fore.BLUE + "\nПолучаем карту сайта через Firecrawl..." + Style.RESET_ALL)
    site_map = app.map_url(url)
    if verbose:
        print(Fore.CYAN + "\n[DEBUG] map_url response:" + Style.RESET_ALL)
        print(json.dumps(site_map.model_dump() if hasattr(site_map, 'model_dump') else site_map, indent=2, ensure_ascii=False))
    all_links = site_map.links

    pages_to_parse = [url]
    used_playwright = False
    if not all_links or (len(all_links) == 1 and all_links[0].rstrip('/') == url.rstrip('/')):
        print(Fore.YELLOW + "[!] Карта сайта пуста, извлекаем ссылки из главной страницы..." + Style.RESET_ALL)
        crawl_status = app.crawl_url(
            url,
            limit=1,
            scrape_options=ScrapeOptions(formats=['markdown', 'html'], onlyMainContent=False),
            poll_interval=5
        )
        doc = crawl_status.data[0].model_dump() if hasattr(crawl_status.data[0], 'model_dump') else crawl_status.data[0]
        content = doc.get('markdown') or doc.get('html') or ''
        all_links = extract_links_from_text(content, url)
        print(Fore.YELLOW + f"Извлечено {len(all_links)} ссылок из главной страницы." + Style.RESET_ALL)
        regex_contacts = extract_contacts_regex(content)
        if not all_links and not regex_contacts:
            print(Fore.YELLOW + "[!] Нет ссылок и контактов на главной. Пробуем повторно с Playwright..." + Style.RESET_ALL)
            crawl_status_pw = app.crawl_url(
                url,
                limit=1,
                scrape_options=ScrapeOptions(formats=['markdown', 'html'], onlyMainContent=False, use_playwright=True),
                poll_interval=5
            )
            doc_pw = crawl_status_pw.data[0].model_dump() if hasattr(crawl_status_pw.data[0], 'model_dump') else crawl_status_pw.data[0]
            content_pw = doc_pw.get('markdown') or doc_pw.get('html') or ''
            all_links = extract_links_from_text(content_pw, url)
            regex_contacts_pw = extract_contacts_regex(content_pw)
            used_playwright = True
            if not all_links and not regex_contacts_pw:
                print(Fore.RED + "[!] Даже с Playwright не удалось найти ни ссылок, ни контактов. Нет контактов для сохранения." + Style.RESET_ALL)
                sys.exit(0)
            content = content_pw
            if not all_links:
                pages_to_parse = [url]
    else:
        contact_candidates = [link for link in all_links if any(
            word in link.lower() for word in ['contact', 'contacts', 'kontakty', 'kontakt', 'контакт', 'контакты']
        )]
        contact_url = contact_candidates[0] if contact_candidates else None
        other_links = [link for link in all_links if link != contact_url]
        select_remote = force_remote or not local_only
        print(Fore.BLUE + f"\nВыбор релевантных страниц для поиска контактов через LLM (remote={select_remote})..." + Style.RESET_ALL)
        n = limit - 1 if contact_url else limit
        best_links = select_best_contact_pages(other_links, n, local_only=local_only, force_remote=force_remote, verbose=verbose)
        if contact_url:
            abs_contact = to_absolute(contact_url, url)
            if abs_contact != url and abs_contact not in pages_to_parse:
                pages_to_parse.append(abs_contact)
        for link in best_links[:n]:
            abs_link = to_absolute(link, url)
            if abs_link != url and abs_link not in pages_to_parse:
                pages_to_parse.append(abs_link)
    print(Fore.BLUE + f"\nБудут проанализированы страницы:" + Style.RESET_ALL)
    for link in pages_to_parse:
        print("  ", link)
    print(Fore.BLUE + "\nПарсим выбранные страницы..." + Style.RESET_ALL)
    all_data = []
    for page_url in pages_to_parse:
        crawl_status = app.crawl_url(
            page_url,
            limit=1,
            scrape_options=ScrapeOptions(formats=['markdown'], onlyMainContent=False, exclude_paths=['blog/*']),
            poll_interval=5
        )
        all_data.extend(crawl_status.data)
    all_contacts = []
    os.makedirs('outputs', exist_ok=True)
    page_num = 1
    llm_requests = 0
    email_has_name = dict()
    phone_has_name = dict()
    whatsapp_has_name = dict()
    telegram_has_name = dict()
    vk_has_name = dict()
    for item in tqdm(all_data, desc="Извлечение контактов"):
        doc = item.model_dump() if hasattr(item, 'model_dump') else (item if isinstance(item, dict) else {})
        content = None
        for fmt in ['markdown', 'text', 'html']:
            if fmt in doc:
                content = doc[fmt]
                break
        if content:
            with open(f'outputs/page_{page_num}.md', 'w', encoding='utf-8') as f:
                f.write(content)
            page_num += 1
            regex_contacts = extract_contacts_regex(content)
            def not_enriched(c):
                email_key = normalize_email(c.get("email"))
                phone_key = normalize_phone(c.get("phone"))
                whatsapp_key = normalize_whatsapp(c.get("whatsapp"))
                telegram_key = normalize_telegram(c.get("telegram"))
                vk_key = normalize_vk(c.get("vk"))
                res = (
                    (email_key and not email_has_name.get(email_key, False)) or
                    (phone_key and not phone_has_name.get(phone_key, False)) or
                    (whatsapp_key and not whatsapp_has_name.get(whatsapp_key, False)) or
                    (telegram_key and not telegram_has_name.get(telegram_key, False)) or
                    (vk_key and not vk_has_name.get(vk_key, False))
                )
                if verbose:
                    print(f"[DEBUG not_enriched] email={email_key}({email_has_name.get(email_key)}), phone={phone_key}({phone_has_name.get(phone_key)}), whatsapp={whatsapp_key}({whatsapp_has_name.get(whatsapp_key)}), telegram={telegram_key}({telegram_has_name.get(telegram_key)}), vk={vk_key}({vk_has_name.get(vk_key)}), => {res}")
                return res
            filtered_contacts = [c for c in regex_contacts if not_enriched(c)]
            if filtered_contacts:
                llm_requests += 1
                contacts = enrich_contacts_with_llm(content, local_only=local_only, verbose=verbose)
                llm_keys = set()
                for c in contacts:
                    if c.get("email"):
                        llm_keys.add(("email", normalize_email(c["email"])))
                    if c.get("phone"):
                        llm_keys.add(("phone", normalize_phone(c["phone"])))
                    if c.get("whatsapp"):
                        llm_keys.add(("whatsapp", normalize_whatsapp(c["whatsapp"])))
                    if c.get("telegram"):
                        llm_keys.add(("telegram", normalize_telegram(c["telegram"])))
                    if c.get("vk"):
                        llm_keys.add(("vk", normalize_vk(c["vk"])))
                for c in contacts:
                    has_name = bool(c.get("full_name") and c["full_name"].strip())
                    if has_name:
                        if c.get("email"):
                            k = normalize_email(c["email"])
                            email_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён email: {k}")
                        if c.get("phone"):
                            k = normalize_phone(c["phone"])
                            phone_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён phone: {k}")
                        if c.get("whatsapp"):
                            k = normalize_whatsapp(c["whatsapp"])
                            whatsapp_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён whatsapp: {k}")
                        if c.get("telegram"):
                            k = normalize_telegram(c["telegram"])
                            telegram_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён telegram: {k}")
                        if c.get("vk"):
                            k = normalize_vk(c["vk"])
                            vk_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Обогащён vk: {k}")
                for c in filtered_contacts:
                    if c.get("phone"):
                        k = normalize_phone(c["phone"])
                        if ("phone", k) not in llm_keys:
                            phone_has_name[k] = True
                            if verbose:
                                print(f"[DEBUG] Игнорируемый phone: {k}")
                if verbose:
                    print("[DEBUG] Словари после обновления:")
                    print("  email_has_name:", email_has_name)
                    print("  phone_has_name:", phone_has_name)
                    print("  whatsapp_has_name:", whatsapp_has_name)
                    print("  telegram_has_name:", telegram_has_name)
                    print("  vk_has_name:", vk_has_name)
                all_contacts.append(contacts)
    if verbose:
        print(Fore.YELLOW + f"Количество запросов к LLM: {llm_requests}." + Style.RESET_ALL)
    merged_contacts = merge_contacts(all_contacts)
    merged_contacts = finalize_contacts_format(merged_contacts)
    merged_contacts = [c for c in merged_contacts if not is_virtual_contact(c)]
    print(Fore.MAGENTA + f"\nНайдено уникальных контактов: {len(merged_contacts)}" + Style.RESET_ALL)
    for i, c in enumerate(merged_contacts, 1):
        print(Fore.CYAN + f"\nКонтакт {i}:" + Style.RESET_ALL)
        for k, v in c.items():
            if v:
                print(f"  {k}: {v}")
    print(Fore.YELLOW + Style.BRIGHT + "\nФинальный запрос к LLM для отбора лучших контактов..." + Style.RESET_ALL)
    best_contacts = summarize_contacts_with_llm(merged_contacts, local_only=local_only and not force_remote, verbose=verbose)
    print(Fore.GREEN + f"\nЛучшие контакты после отбора:" + Style.RESET_ALL)
    if best_contacts:
        for i, c in enumerate(best_contacts, 1):
            print(Fore.CYAN + f"\nКонтакт {i}:" + Style.RESET_ALL)
            for k, v in c.items():
                if v:
                    print(f"  {k}: {v}")
        if output_file:
            save_contacts_csv(best_contacts, output_file)
    else:
        print(Fore.RED + "Нет контактов для сохранения." + Style.RESET_ALL)