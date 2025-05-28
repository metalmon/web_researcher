"""
Module for generating personalized outreach emails based on company analysis
"""

import json
from typing import Dict, Any, Tuple
from config import PRODUCT_CONFIG
from prompts_and_schemas import (
    OUTREACH_EMAIL_SCHEMA, OUTREACH_EMAIL_PROMPT,
    LEAD_SCORE_SCHEMA, LEAD_SCORE_PROMPT,
    EMAIL_SUBJECT_SCHEMA, EMAIL_SUBJECT_PROMPT
)
from llm_router import sales_generate_completion
from utils import extract_content, normalize_llm_response, is_russian
from progress import progress

def score_lead(
    company_analysis: Dict[str, Any],
    local_only: bool = True,
    force_remote: bool = False,
    debug: bool = False
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    Оценивает релевантность лида на основе анализа компании.
    
    Args:
        company_analysis: Dictionary containing company analysis results
        local_only: Whether to use only local model
        force_remote: Whether to force using remote model
        debug: Whether to enable debug output
        
    Returns:
        Tuple of (score, explanation, communication_scheme) where:
        - score: float between 0 and 1 representing lead relevance
        - explanation: dictionary with detailed scoring breakdown
        - communication_scheme: dictionary with recommended communication scheme
    """
    # Prepare context for the LLM
    context = {
        "company_analysis": company_analysis,
        "product_info": PRODUCT_CONFIG
    }
    
    # Generate lead score
    response = sales_generate_completion(
        role="lead scoring expert",
        task=LEAD_SCORE_PROMPT,
        content=json.dumps(context, ensure_ascii=False),
        response_format=LEAD_SCORE_SCHEMA,
        local_only=local_only,
        force_remote=force_remote,
        debug=debug,
        temperature=0.1  # Низкая температура для более точной оценки
    )
    
    # Extract and normalize the response
    score_content = extract_content(response)
    normalized_content = normalize_llm_response(score_content)
    
    # Check if any part is not in Russian and translate if needed
    if not is_russian(json.dumps(normalized_content, ensure_ascii=False)):
        translate_prompt = (
            "Переведи этот текст на русский язык, сохрани стиль и структуру ответа. Не добавляй ничего лишнего.\n" + 
            json.dumps(normalized_content, ensure_ascii=False)
        )
        translated = sales_generate_completion(
            role="lead scoring expert",
            task="Ты — профессиональный переводчик. Переведи на русский язык, сохрани стиль и структуру.",
            content=translate_prompt,
            response_format=LEAD_SCORE_SCHEMA,
            local_only=local_only,
            force_remote=force_remote,
            debug=debug,
            temperature=0.1
        )
        normalized_content = normalize_llm_response(extract_content(translated))
    
    # Extract score, explanation and communication scheme
    try:
        if isinstance(normalized_content, str):
            normalized_content = json.loads(normalized_content)
        score = float(normalized_content.get("score", 0))
        explanation = normalized_content.get("explanation", {})
        communication_scheme = normalized_content.get("communication_scheme", {})
        return score, explanation, communication_scheme
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0, {"error": "Не удалось распарсить оценку лида"}, {}

def generate_outreach_email(
    company_analysis: Dict[str, Any],
    local_only: bool = True,
    force_remote: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate a personalized outreach email based on company analysis.
    
    Args:
        company_analysis: Dictionary containing company analysis results
        local_only: Whether to use only local model
        force_remote: Whether to force using remote model
        debug: Whether to enable debug output
        
    Returns:
        Dictionary containing the generated email components
    """
    # Сначала получаем оценку лида и схему коммуникации
    score, explanation, communication_scheme = score_lead(company_analysis, local_only=local_only, force_remote=force_remote, debug=debug)
    
    # Prepare context for the LLM
    context = {
        "company_analysis": company_analysis,
        "product_info": PRODUCT_CONFIG,
        "communication_scheme": communication_scheme,
        "lead_score": {
            "score": score,
            "explanation": explanation
        },
        "industry": company_analysis.get("Отрасль", {})
    }
    
    # Обновляем прогресс перед генерацией письма
    progress.update(1, "Генерация письма")
    
    # Generate email body first
    response = sales_generate_completion(
        role="outreach email writer",
        task=OUTREACH_EMAIL_PROMPT,
        content=json.dumps(context, ensure_ascii=False),
        response_format=OUTREACH_EMAIL_SCHEMA,
        local_only=local_only,
        force_remote=force_remote,
        debug=debug,
        temperature=1
    )
    
    # Extract and normalize the response
    email_content = extract_content(response)
    normalized_content = normalize_llm_response(email_content)
    
    # Check if any part is not in Russian and translate if needed
    if not is_russian(json.dumps(normalized_content, ensure_ascii=False)):
        translate_prompt = (
            "Переведи этот текст на русский язык, сохрани стиль и структуру ответа. Не добавляй ничего лишнего.\n" + 
            json.dumps(normalized_content, ensure_ascii=False)
        )
        translated = sales_generate_completion(
            role="outreach email writer",
            task="Ты — профессиональный переводчик. Переведи на русский язык, сохрани стиль и структуру.",
            content=translate_prompt,
            response_format=OUTREACH_EMAIL_SCHEMA,
            local_only=local_only,
            force_remote=force_remote,
            debug=debug
        )
        normalized_content = normalize_llm_response(extract_content(translated))
    
    # Generate subject line based on the email body
    subject_context = {
        "email_body": normalized_content,
        "company_analysis": company_analysis,
        "lead_score": {
            "score": score,
            "explanation": explanation
        },
        "industry": company_analysis.get("Отрасль", {})
    }

    progress.update(1, "Генерация заголовка")    
    subject_response = sales_generate_completion(
        role="email subject writer",
        task=EMAIL_SUBJECT_PROMPT,
        content=json.dumps(subject_context, ensure_ascii=False),
        response_format=EMAIL_SUBJECT_SCHEMA,
        local_only=local_only,
        force_remote=force_remote,
        debug=debug,
        temperature=1  # Немного креативности для заголовка
    )
    
    # Extract and normalize the subject
    subject_content = extract_content(subject_response)
    normalized_subject = normalize_llm_response(subject_content)
    
    # Add subject to the email content
    if isinstance(normalized_subject, dict) and "subject" in normalized_subject:
        normalized_content["subject"] = normalized_subject["subject"]

    return normalized_content

def format_email_for_sending(email_content: Dict[str, Any]) -> str:
    """
    Format the email content into a ready-to-send email.
    
    Args:
        email_content: Dictionary containing email components
        
    Returns:
        Formatted email text
    """
    # Форматируем story как часть повествования
    story = email_content.get("story", "")
    if isinstance(story, str):
        story_lines = story.split('\n')
        formatted_story = []
        for line in story_lines:
            line = line.strip()
            if line:
                formatted_story.append(line)
        story = ' '.join(formatted_story)
    
    # Форматируем insight как часть повествования
    insight = email_content.get("insight", "")
    if isinstance(insight, str):
        insight_lines = insight.split('\n')
        formatted_insight = []
        for line in insight_lines:
            line = line.strip()
            if line:
                formatted_insight.append(line)
        insight = ' '.join(formatted_insight)
    
    # Форматируем proposal как часть повествования
    proposal = email_content.get("proposal", "")
    if isinstance(proposal, str):
        proposal_lines = proposal.split('\n')
        formatted_proposal = []
        for line in proposal_lines:
            line = line.strip()
            if line:
                formatted_proposal.append(line)
        proposal = ' '.join(formatted_proposal)
    
    # Собираем письмо с правильными отступами
    email = f"{email_content.get('greeting', '')}\n\n{story}\n\n{insight}\n\n{proposal}\n\n{email_content.get('call_to_action', '')}\n\n{email_content.get('signature', '')}"
    
    # Убираем лишние пробелы и переносы строк
    email = '\n'.join(line for line in email.split('\n') if line.strip())
    
    return email.strip()

def save_email_to_file(email_content, output_file):
    """Save the generated email to a file."""
    try:
        # Парсим JSON, если получили строку
        if isinstance(email_content, str):
            try:
                email_content = json.loads(email_content)
            except json.JSONDecodeError:
                # Если не удалось распарсить JSON, создаем базовую структуру
                email_content = {
                    "subject": "Письмо для первого контакта",
                    "body": email_content
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Тема: {email_content.get('subject', '')}\n\n")
            f.write(email_content.get('body', ''))
    except Exception as e:
        print(f"[!] Ошибка при сохранении письма: {e}")
        # Fallback: сохраняем как есть
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(email_content)) 