# Chorus Agent Guidelines

## Agent Rules
- Agents MUST look for `AGENTS.md` in the project root.
- If found, include its contents in the agent's context.
- Parse this file and any `AGENTS.md` in the current working directory as plain text instructions; no special metadata is required.
- Agents MAY combine instructions from multiple `AGENTS.md` files when present.

## Project Principles
- Chorus is a local-first LangGraph system for long-form fiction.
- PostgreSQL is the canonical store.
- Code is Python 3.12, async-first, and relies on Pydantic models.
- Review the files in `memory-bank/` before making changes.

## Repository Layout
- `src/chorus/` – core application code
  - `agents/` – agent implementations and orchestration helpers
  - `canon/` – canonical data layer and graph synchronization
  - `core/` – shared utilities for logging, queueing, and environment
  - `langgraph/` – LangGraph workflow definitions
  - `models/` – Pydantic data structures
  - `modes/` – CLI and interactive modes
  - `ner/` – named entity recognition components
  - `web/` – FastAPI web interface
- `tests/` – test suite mirroring `src/`
- `memory-bank/` – persistent project context for agents
- `docs/` – API and user documentation
- `scripts/` – setup and maintenance scripts
- `alembic/` – database migration scripts
- `config/` – deployment and runtime configuration

## Development Workflow

### Setup
1. `cp .env.example .env` and fill in values.
2. `docker compose up -d` to start PostgreSQL.
3. `poetry install` then `poetry shell`.

### Quality Checks (run before committing)
- `poetry run ruff check src`
- `poetry run mypy src`
- `poetry run pytest`

### Coding Conventions
- Start each source file with a comment of its relative path.
- Format and lint with `ruff`; use explicit type hints and Google-style docstrings.
- Use `async/await` for all I/O and keep side effects idempotent.
- Read configuration from `.env`; update `.env.example` when adding variables.
- Apply schema changes through Alembic migrations.

### Testing
- Place tests in `tests/` mirroring `src/`.
- New features require tests; bug fixes need regression tests.
- Use fixtures like `monkeypatch` to manage environment variables.

### Commit and PR Guidelines
- Use Conventional Commits.
- PR titles: `[COMPONENT] concise summary`.
- PR description sections: Summary, Implementation Details, Testing, Performance Considerations, Related Issue.

## Documentation and Memory
- Significant changes or insights SHOULD update files in `memory-bank/`.
