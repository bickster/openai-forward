# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

OpenAI Forward is a FastAPI-based reverse proxy service for the OpenAI API that enables access in regions where OpenAI services are restricted. Key architectural components:

### Core Components
- **OpenaiBase** (`base.py`): Base class handling proxy logic, API key rotation, authentication, and request forwarding
- **Openai** (`openai.py`): Main service class extending OpenaiBase with IP validation and routing
- **FastAPI App** (`app.py`): Main application using sparrow-python framework with route configuration
- **Configuration** (`config.py`): Startup configuration, logging setup, and environment handling

### Key Features
- API key rotation/pooling from multiple OpenAI keys
- Custom forward keys for secure API key distribution
- Request logging and chat conversation tracking
- IP whitelist/blacklist validation
- Image generation platform integration (including Flux)
- Content moderation error handling

### Router Structure
- `routers/openai_v1.py`: OpenAI v1 API endpoint handlers
- `routers/image_gen_platform.py`: Image generation platform routing
- `routers/schemas.py`: Request/response data models

## Development Commands

### Running the Service
```bash
# Development server
openai-forward run --port=8000 --workers=1 --log_chat=true

# Production with custom config
openai-forward run --port=8000 --workers=4 --api_key=sk-xxx --forward_key=fk-xxx
```

### Testing
```bash
# Run all tests
pytest -v tests
# or 
make test

# Run with coverage and doctest
pytest --doctest-modules --doctest-glob=README.md --doctest-glob=*.py --ignore=setup.py
```

### Code Quality
```bash
# Format code (uses black)
black -S openai_forward/

# Check formatting the way CI does (whole repo), or fix it in place
./scripts/black.sh
./scripts/black.sh --fix
```

### Docker Development
```bash
# Build and run
make build
make start

# Development with compose
make up
make down

# Interactive container
make run
make exec
```

### Log Management
```bash
# Convert chat logs to JSON
openai-forward convert

# Clean logs
./scripts/logclean.sh

# View container logs
make log
```

## Environment Configuration

Key environment variables (can be set in `.env` file):
- `OPENAI_BASE_URL`: Target OpenAI API base URL
- `OPENAI_API_KEY`: Space-separated OpenAI API keys for rotation
- `FORWARD_KEY`: Space-separated custom keys for API access
- `ROUTE_PREFIX`: Custom route prefix
- `LOG_CHAT`: Enable chat logging (true/false)
- `OPENAI_IMAGE_MODEL`: pins the model used for `/v1/images/generations` and `/v1/images/edits` when they route to OpenAI, overriding whatever the client asked for. This is how a new OpenAI image model is rolled out without an app release — set it, restart, confirm the `Image model pin:` line in the log. Unset means the client's own value is used. There is deliberately no chat equivalent: clients shape chat params by model family (`max_tokens` vs `max_completion_tokens`, `system` vs `developer` role), so swapping the id server-side would send old-family params to a new-family endpoint.
- `OPENAI_IMAGE_FLEXIBLE_SIZE_MODELS`: comma-separated model id prefixes that accept an arbitrary `size` (default `gpt-image-2`, which also matches dated snapshots such as `gpt-image-2-2026-04-21`). For these, an aspect ratio from the client (`16:9`) is converted to exact dimensions honouring the documented constraints — both edges divisible by 16, ratio within 1:3–3:1, max 3840x2160 — and `size: "auto"` is passed through. Any other model gets the three legacy sizes (`1024x1024`, `1536x1024`, `1024x1536`) chosen by orientation, because a legacy size is valid for every model while an arbitrary one is a 400 on models that predate it.
- `OPENAI_IMAGE_MODEL_FALLBACK`: what a `flux*` model id is rewritten to when no pin is set (default `gpt-image-1.5`). Applies when a client asks for Flux but the request routes to OpenAI — otherwise OpenAI answers 400 `The model 'flux-kontext' does not exist.`
- `IP_WHITELIST`/`IP_BLACKLIST`: IP access control (space-separated)
- `UA_WHITELIST`/`UA_BLACKLIST`: User-Agent access control (comma-separated glob patterns, e.g. `UA_WHITELIST="okhttp/3.9.*"` with `UA_BLACKLIST="okhttp/*"`). Whitelist match allows, then blacklist match blocks, otherwise allowed; add `*` to the blacklist for strict whitelist-only mode. Patterns must match the full UA string (case-insensitive). When either list is set, requests with a missing/empty User-Agent are blocked. Blocked requests get a generic 403 identical to the HMAC-failure response (no hint that UA filtering exists).

## Testing Notes

- Tests use pytest with timeout and repeat markers
- Includes doctests from README.md and Python files
- `tests/test_http.py` are integration tests: they launch the real server and talk to it over localhost. They skip unless the `openai-forward` console script is on `PATH`, so run `pip install -e .` in your venv to exercise them.
- Test configuration in `pytest.ini` with 180s timeout
- Uses markers: `slow`, `timeout`, `repeat`

## Remembered CLI Parameters and Environment Variables
- Remembered all CLI parameters and environment variables from previous context
- To run locally in basic configuration: `python3 -m openai_forward run --port=8000 --app_secret=<your-secret> --api_key=<your-key>`