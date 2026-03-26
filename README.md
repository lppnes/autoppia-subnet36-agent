# autoppia-subnet36-agent

Minimal FastAPI agent for Autoppia Subnet 36.

## Endpoints

- `GET /health`
- `POST /act`

## Local run

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

This repository is intentionally minimal so it matches the validator sandbox
contract used by `autoppia_web_agents_subnet`.
