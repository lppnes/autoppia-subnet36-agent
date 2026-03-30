"""
Autoppia Subnet 36 - LLM-Powered Web Agent
Uses sandbox gateway LLM API to analyze HTML and produce precise action sequences.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Autoppia SN36 LLM Agent")

# Sandbox gateway endpoint (provided by validator environment)
GATEWAY_URL = os.getenv("SANDBOX_GATEWAY_URL", "http://sandbox-gateway:9000")
GATEWAY_ADMIN_TOKEN = os.getenv("SANDBOX_GATEWAY_ADMIN_TOKEN", "")
LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4.1-mini")

# Max HTML bytes to send to LLM (to stay within token limits and cost)
MAX_HTML_CHARS = 12000


class ActRequest(BaseModel):
    task_id: str | None = None
    prompt: str
    url: str | None = None
    snapshot_html: str | None = None
    screenshot: str | None = None
    step_index: int = 0
    web_project_id: str | None = None
    history: list[dict[str, Any]] | None = None


def _truncate_html(html: str, max_chars: int = MAX_HTML_CHARS) -> str:
    """Truncate HTML but keep structure visible."""
    if len(html) <= max_chars:
        return html
    return html[:max_chars] + "\n... [truncated]"


def _clean_html(html: str) -> str:
    """Extract relevant interactive elements from HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts and styles
        for tag in soup.find_all(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()

        # Focus on interactive elements + headings + visible text
        important = []
        for tag in soup.find_all(
            ["input", "button", "select", "textarea", "a", "form", "h1", "h2", "h3", "label", "p", "span", "div"]
        ):
            text = tag.get_text(strip=True)
            attrs = {k: v for k, v in tag.attrs.items() if k in ("id", "name", "class", "type", "href", "placeholder", "value", "action", "method")}
            if text or attrs:
                line = f"<{tag.name}"
                for k, v in attrs.items():
                    v_str = " ".join(v) if isinstance(v, list) else str(v)
                    line += f' {k}="{v_str}"'
                if text:
                    line += f">{text[:200]}</{tag.name}>"
                else:
                    line += "/>"
                important.append(line)

        cleaned = "\n".join(important)
        if len(cleaned) > MAX_HTML_CHARS:
            cleaned = cleaned[:MAX_HTML_CHARS] + "\n... [truncated]"
        return cleaned
    except Exception:
        return _truncate_html(html)


def _call_llm(messages: list[dict], task_id: str | None = None) -> str | None:
    """Call LLM via sandbox gateway."""
    headers = {"Content-Type": "application/json"}
    if GATEWAY_ADMIN_TOKEN:
        headers["Authorization"] = f"Bearer {GATEWAY_ADMIN_TOKEN}"

    # Include task_id header if available (for gateway billing tracking)
    if task_id:
        headers["X-Task-Id"] = task_id

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.1,
    }

    try:
        url = f"{GATEWAY_URL}/openai/v1/chat/completions"
        resp = httpx.post(url, json=payload, headers=headers, timeout=20.0)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            logger.warning(f"LLM gateway returned {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


SYSTEM_PROMPT = """You are a web automation agent. Given a task description and the current HTML state of a webpage, return a JSON array of actions to perform.

Available action types:
- NavigateAction: {"type": "NavigateAction", "url": "https://..."}
- ClickAction: {"type": "ClickAction", "selector": {"type": "cssSelector", "value": "css-selector"}}
- TypeAction: {"type": "TypeAction", "selector": {"type": "cssSelector", "value": "css-selector"}, "text": "text to type"}
- SelectAction: {"type": "SelectAction", "selector": {"type": "cssSelector", "value": "css-selector"}, "value": "option value"}
- ScrollAction: {"type": "ScrollAction", "x": 0, "y": 300}
- WaitAction: {"type": "WaitAction", "milliseconds": 1000}

Selector types: "cssSelector", "xpathSelector", "textSelector"
For text-based selection: {"type": "textSelector", "value": "button text"}
For xpath: {"type": "xpathSelector", "value": "//button[@id='submit']"}

IMPORTANT RULES:
1. Return ONLY a JSON array of actions, no explanation
2. Be precise - use specific selectors based on the HTML
3. For forms: navigate -> fill fields -> submit
4. Return empty array [] if task is already complete or impossible
5. Prefer cssSelector when IDs are available (#id), use xpathSelector for complex cases
6. For step 0, start with NavigateAction to the URL
7. Each step returns 1-3 actions maximum
"""


def _parse_actions(llm_response: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    if not llm_response:
        return []

    # Try to find JSON array
    text = llm_response.strip()

    # Remove markdown code blocks if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Find the first [ ... ] block
    try:
        start = text.index("[")
        # Find matching ]
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > start:
            json_str = text[start : end + 1]
            actions = json.loads(json_str)
            if isinstance(actions, list):
                return actions
    except Exception as e:
        logger.warning(f"Failed to parse LLM actions: {e}, response: {text[:200]}")

    return []


def _heuristic_actions(req: ActRequest) -> list[dict[str, Any]]:
    """Rule-based fallback when LLM is unavailable."""
    html = req.snapshot_html or ""
    prompt_lower = req.prompt.lower()

    if req.step_index == 0 and req.url:
        return [{"type": "NavigateAction", "url": req.url}]

    soup = BeautifulSoup(html, "html.parser") if html else None

    if req.step_index == 1:
        # Try to find the most relevant input and interact with it
        if soup:
            # Search tasks
            if any(k in prompt_lower for k in ("search", "find", "look for", "query")):
                search_input = soup.find("input", {"type": ["search", "text"]}) or soup.find("input")
                if search_input:
                    query = _extract_query_from_prompt(req.prompt)
                    sel_id = search_input.get("id", "")
                    sel_name = search_input.get("name", "")
                    if sel_id:
                        selector = {"type": "cssSelector", "value": f"#{sel_id}"}
                    elif sel_name:
                        selector = {"type": "cssSelector", "value": f"[name='{sel_name}']"}
                    else:
                        selector = {"type": "xpathSelector", "value": "(//input)[1]"}
                    return [{"type": "TypeAction", "selector": selector, "text": query}]

            # Login tasks
            if any(k in prompt_lower for k in ("login", "sign in", "log in", "authenticate")):
                email_input = soup.find("input", {"type": "email"}) or soup.find("input", {"name": re.compile(r"email|user", re.I)})
                if email_input:
                    sel_id = email_input.get("id", "")
                    selector = {"type": "cssSelector", "value": f"#{sel_id}"} if sel_id else {"type": "xpathSelector", "value": "(//input[@type='email' or @name='email' or @name='username'])[1]"}
                    username = _extract_field(req.prompt, "email") or "user@example.com"
                    return [{"type": "TypeAction", "selector": selector, "text": username}]

    if req.step_index == 2:
        # Try to submit
        if soup:
            submit_btn = (
                soup.find("button", {"type": "submit"})
                or soup.find("input", {"type": "submit"})
                or soup.find("button", string=re.compile(r"submit|search|sign|log|send|save|continue|ok|confirm", re.I))
            )
            if submit_btn:
                text_content = submit_btn.get_text(strip=True)
                if text_content:
                    return [{"type": "ClickAction", "selector": {"type": "textSelector", "value": text_content}}]
                sel_id = submit_btn.get("id", "")
                if sel_id:
                    return [{"type": "ClickAction", "selector": {"type": "cssSelector", "value": f"#{sel_id}"}}]
            return [{"type": "ClickAction", "selector": {"type": "xpathSelector", "value": "(//button[@type='submit'] | //input[@type='submit'])[1]"}}]

    return []


def _extract_query_from_prompt(prompt: str) -> str:
    """Try to extract a search query from the prompt."""
    patterns = [
        r'search (?:for )?["\']?([^"\']+)["\']?',
        r'find ["\']?([^"\']+)["\']?',
        r'look for ["\']?([^"\']+)["\']?',
        r'query ["\']?([^"\']+)["\']?',
    ]
    for pattern in patterns:
        m = re.search(pattern, prompt, re.I)
        if m:
            return m.group(1).strip()
    # Return last few words as query
    words = prompt.split()
    return " ".join(words[-3:]) if len(words) > 3 else prompt


def _extract_field(prompt: str, field: str) -> str | None:
    """Extract a specific field value from prompt."""
    patterns = {
        "email": r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}',
        "username": r'(?:username|user):?\s+([^\s,]+)',
        "password": r'(?:password|pass):?\s+([^\s,]+)',
    }
    if field in patterns:
        m = re.search(patterns[field], prompt, re.I)
        if m:
            return m.group(0) if field == "email" else m.group(1)
    return None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/act")
async def act(req: ActRequest) -> dict[str, list[dict[str, Any]]]:
    logger.info(f"[Step {req.step_index}] task_id={req.task_id} url={req.url} prompt={req.prompt[:80]}")

    html = req.snapshot_html or ""
    history_str = ""

    if req.history:
        history_str = "\nPrevious actions:\n"
        for h in req.history[-3:]:  # Last 3 steps
            history_str += f"- Step {h.get('step_index', '?')}: {json.dumps(h.get('actions', []))[:200]}\n"

    # Step 0: Always navigate first
    if req.step_index == 0 and req.url:
        return {"actions": [{"type": "NavigateAction", "url": req.url}]}

    # Clean HTML for LLM
    cleaned_html = _clean_html(html) if html else ""

    # Build LLM prompt
    user_message = f"""Task: {req.prompt}

Current URL: {req.url or "unknown"}
Step: {req.step_index}
{history_str}

Current page HTML (relevant elements):
{cleaned_html}

Return a JSON array of 1-3 actions to perform on this step to progress toward completing the task.
If the task appears already complete, return [].
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Try LLM first
    llm_response = _call_llm(messages, task_id=req.task_id)

    if llm_response:
        actions = _parse_actions(llm_response)
        if actions:
            logger.info(f"[Step {req.step_index}] LLM actions: {json.dumps(actions)[:300]}")
            return {"actions": actions}
        logger.warning(f"[Step {req.step_index}] LLM returned empty/invalid actions, using heuristic fallback")
    else:
        logger.warning(f"[Step {req.step_index}] LLM unavailable, using heuristic fallback")

    # Fallback to heuristic
    actions = _heuristic_actions(req)
    logger.info(f"[Step {req.step_index}] Heuristic actions: {json.dumps(actions)[:300]}")
    return {"actions": actions}
