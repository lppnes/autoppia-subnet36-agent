from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Autoppia Subnet 36 Agent")


class ActRequest(BaseModel):
    task_id: str | None = None
    prompt: str
    url: str | None = None
    snapshot_html: str | None = None
    screenshot: str | None = None
    step_index: int = 0
    web_project_id: str | None = None
    history: list[dict[str, Any]] | None = None


def _looks_like_input(prompt: str, html: str) -> bool:
    text = f"{prompt}\n{html}".lower()
    return any(token in text for token in ("input", "email", "password", "search", "message", "name"))


def _looks_like_submit(prompt: str, html: str) -> bool:
    text = f"{prompt}\n{html}".lower()
    return any(token in text for token in ("submit", "sign in", "log in", "search", "send", "checkout", "save"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/act")
async def act(req: ActRequest) -> dict[str, list[dict[str, Any]]]:
    html = req.snapshot_html or ""

    # Very small baseline behavior:
    # 1. On the first step, navigate to the provided URL if one exists.
    # 2. Then try a conservative type/click sequence using generic selectors.
    # 3. If nothing obvious matches, return no actions.
    if req.step_index == 0 and req.url:
        return {"actions": [{"type": "NavigateAction", "url": req.url}]}

    if req.step_index == 1 and _looks_like_input(req.prompt, html):
        return {
            "actions": [
                {
                    "type": "TypeAction",
                    "selector": {
                        "type": "xpathSelector",
                        "value": "(//input | //textarea)[1]",
                    },
                    "text": "<username>",
                }
            ]
        }

    if req.step_index == 2 and _looks_like_submit(req.prompt, html):
        return {
            "actions": [
                {
                    "type": "ClickAction",
                    "selector": {
                        "type": "xpathSelector",
                        "value": "(//button | //input[@type='submit'])[1]",
                    },
                }
            ]
        }

    return {"actions": []}
