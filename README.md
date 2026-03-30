# autoppia-subnet36-agent

LLM-powered web agent for Autoppia Subnet 36 (Bittensor SN36).

## Architecture

This agent uses:
1. **LLM reasoning** via sandbox gateway (OpenAI-compatible) to analyze HTML and plan actions
2. **BeautifulSoup** for HTML parsing and element extraction
3. **Heuristic fallback** when LLM is unavailable

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /act` → `{"actions": [...]}`

## Action Types Supported

- `NavigateAction` - navigate to URL
- `ClickAction` - click element by CSS/XPath/text selector
- `TypeAction` - type text into input
- `SelectAction` - select dropdown option
- `ScrollAction` - scroll page
- `WaitAction` - wait milliseconds

## Strategy

For each step:
1. Step 0: NavigateAction to task URL
2. Steps 1+: LLM analyzes cleaned HTML + task prompt → returns optimal action sequence
3. Fallback: heuristic rules for common patterns (search, login, submit)

## Local Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
