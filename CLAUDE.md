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

# Run formatter script
./scripts/black.sh
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
- `IP_WHITELIST`/`IP_BLACKLIST`: IP access control

## Testing Notes

- Tests use pytest with timeout and repeat markers
- Includes doctests from README.md and Python files
- Test configuration in `pytest.ini` with 180s timeout
- Uses markers: `slow`, `timeout`, `repeat`

## Remembered CLI Parameters and Environment Variables
- Remembered all CLI parameters and environment variables from previous context
- To run locally in basic configuration: `python3 -m openai_forward run --port=8000 --app_secret=<your-secret> --api_key=<your-key>`